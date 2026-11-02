"""
Детектор участков сварного шва на основе Ultralytics YOLO.

Использует ultralytics для инференса. Поддерживает:
- Загрузку любой .pt модели (YOLOv8/v9/v10/v11)
- Настраиваемые пороги confidence/IoU
- TTA (Test-Time Augmentation) для повышения точности
- Аннотацию изображений с цветовой схемой по классам
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "Не установлен пакет ultralytics. Выполните: pip install ultralytics"
    ) from e


DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "best.pt"

# Цвета BGR для классов дефектов (контрастные)
CLASS_COLORS_BGR = {
    "Bad Weld":  (60, 60, 230),    # красный
    "Good Weld": (60, 200, 80),    # зелёный
    "Defect":    (40, 140, 250),   # оранжевый
    # Резервные цвета для других классов
    "porosity":  (200, 80, 200),
    "crack":     (40, 40, 200),
    "spatter":   (180, 180, 60),
    "undercut":  (80, 200, 200),
}

DEFAULT_PALETTE = [
    (60, 60, 230), (60, 200, 80), (40, 140, 250),
    (200, 80, 200), (180, 180, 60), (80, 200, 200),
    (250, 100, 40), (140, 40, 200),
]


@dataclass
class Detection:
    """Единичный объект детекции."""
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]  # абсолютные координаты в пикселях

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class InferenceResult:
    """Результат инференса для одного кадра/изображения."""
    detections: list[Detection] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None  # RGB
    original_shape: tuple[int, int] = (0, 0)      # (H, W)
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    @property
    def count_by_class(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts


class WeldingDetector:
    """
    Высокоуровневая обёртка над YOLO для детекции дефектов сварки.

    Параметры
    ---------
    model_path : str | Path
        Путь к файлу весов модели (.pt). По умолчанию используется
        обученная модель из каталога models.
    device : str
        'cpu', 'cuda', '0', '1' и т.п.
    conf : float
        Порог уверенности для отбора детекций.
    iou : float
        Порог IoU для NMS.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        use_tta: bool = False,
        class_thresholds: Optional[dict[str, float]] = None,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.use_tta = use_tta
        # Class-specific confidence thresholds (key = class name OR class id).
        # Используется при дефектоскопии: например, для "Defect" ставим
        # низкий порог (не пропускать), а для "Good Weld" — высокий
        # (требуем большей уверенности, что шов действительно норма).
        self.class_thresholds: dict = dict(class_thresholds or {})
        self.model: Optional[YOLO] = None
        self.class_names: dict[int, str] = {}
        self._color_cache: dict[int, tuple[int, int, int]] = {}

    # ------------------------------------------------------------------
    # Загрузка модели
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Загрузить веса. Может занять время при первом запуске."""
        self.model = YOLO(self.model_path)
        # ultralytics хранит имена классов в model.names (dict[int, str])
        names = getattr(self.model, "names", None) or {}
        if isinstance(names, dict):
            self.class_names = {int(k): str(v) for k, v in names.items()}
        else:
            self.class_names = {i: str(n) for i, n in enumerate(names)}

    def is_loaded(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    # Инференс
    # ------------------------------------------------------------------
    def predict_image(
        self,
        image: np.ndarray | str | Path | Image.Image,
        draw: bool = True,
    ) -> InferenceResult:
        """
        Выполнить детекцию на одном изображении.

        Возвращает InferenceResult с детекциями и (опционально) RGB-картинкой
        с отрисованными bbox'ами.
        """
        if not self.is_loaded():
            self.load()
        assert self.model is not None

        # Приводим вход к BGR ndarray, как ожидает OpenCV
        bgr = self._to_bgr(image)
        h, w = bgr.shape[:2]

        # Если заданы per-class thresholds — для NMS используем минимальный
        # порог, а реальную фильтрацию делаем в _extract_detections.
        effective_conf = self.conf
        if self.class_thresholds:
            effective_conf = min(
                [self.conf] + [float(v) for v in self.class_thresholds.values()]
            )

        t0 = time.perf_counter()
        # ultralytics возвращает список Result-объектов
        results = self.model.predict(
            source=bgr,
            conf=effective_conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            augment=self.use_tta,
            verbose=False,
        )
        t1 = time.perf_counter()

        result = results[0]
        detections = self._extract_detections(result)
        t2 = time.perf_counter()

        annotated = None
        if draw:
            annotated_bgr = self._draw(bgr.copy(), detections)
            annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # ultralytics возвращает свои тайминги через result.speed (мс)
        speed = getattr(result, "speed", None) or {}
        pre_ms = float(speed.get("preprocess", 0.0))
        inf_ms = float(speed.get("inference", (t1 - t0) * 1000.0))
        post_ms = float(speed.get("postprocess", (t2 - t1) * 1000.0))

        return InferenceResult(
            detections=detections,
            annotated_image=annotated,
            original_shape=(h, w),
            preprocess_ms=pre_ms,
            inference_ms=inf_ms,
            postprocess_ms=post_ms,
        )

    def predict_batch(
        self,
        paths: list[str | Path],
        draw: bool = False,
    ) -> list[InferenceResult]:
        """Пакетный инференс по списку путей."""
        return [self.predict_image(p, draw=draw) for p in paths]

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------
    def _to_bgr(self, image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise FileNotFoundError(f"Не удалось прочитать изображение: {image}")
            return img
        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            return image
        raise TypeError(f"Неподдерживаемый тип входа: {type(image)}")

    def _threshold_for(self, cid: int, name: str) -> float:
        """Порог для конкретного класса с фолбэком на общий self.conf."""
        if name in self.class_thresholds:
            return float(self.class_thresholds[name])
        if cid in self.class_thresholds:
            return float(self.class_thresholds[cid])
        return float(self.conf)

    def _extract_detections(self, result) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        # Фильтр class_id, которых нет в модели (защита от загрузки
        # COCO-pretrained .pt без fine-tune на сварке).
        max_cid = max(self.class_names) if self.class_names else -1

        out: list[Detection] = []
        for i in range(len(boxes)):
            cid = int(cls[i])
            if self.class_names and cid > max_cid:
                continue
            name = self.class_names.get(cid, f"class_{cid}")
            c = float(conf[i])
            # Per-class threshold (если задан)
            if c < self._threshold_for(cid, name):
                continue
            out.append(Detection(
                class_id=cid,
                class_name=name,
                confidence=c,
                box_xyxy=tuple(xyxy[i].tolist()),
            ))
        return out

    def _color_for(self, det: Detection) -> tuple[int, int, int]:
        if det.class_name in CLASS_COLORS_BGR:
            return CLASS_COLORS_BGR[det.class_name]
        if det.class_id in self._color_cache:
            return self._color_cache[det.class_id]
        color = DEFAULT_PALETTE[det.class_id % len(DEFAULT_PALETTE)]
        self._color_cache[det.class_id] = color
        return color

    def _draw(self, bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
        h, w = bgr.shape[:2]
        thickness = max(2, int(round(min(h, w) / 400)))
        font_scale = max(0.5, min(h, w) / 1200)

        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.box_xyxy]
            color = self._color_for(det)

            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness)

            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), bl = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            ty1 = max(0, y1 - th - bl - 4)
            cv2.rectangle(bgr, (x1, ty1), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                bgr, label, (x1 + 2, y1 - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                thickness, lineType=cv2.LINE_AA,
            )
        return bgr


def list_models_in_dir(directory: str | Path) -> list[Path]:
    """Удобная функция для поиска .pt файлов в директории."""
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(d.rglob("*.pt"))
