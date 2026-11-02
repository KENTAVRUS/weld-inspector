"""
Обучение YOLO-модели на локальном датасете Welding Defect Dataset (v2).

Простейший запуск (CPU, пресет balanced):
    python train.py

Полный YOLOv10x на GPU:
    python train.py --preset best --device 0

Кастом:
    python train.py --model yolov10s.pt --epochs 80 --batch 8 --imgsz 640

Пресеты:
    fast     — YOLOv10n,  30 эпох, imgsz=480  — для быстрой проверки пайплайна.
    balanced — YOLOv10s,  80 эпох, imgsz=640  — разумный baseline на CPU за ночь.
    best     — YOLOv10x, 150 эпох, imgsz=640  — максимум точности (НУЖЕН GPU).
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = PROJECT_DIR / "data.yaml"


PRESETS = {
    "fast": dict(
        model="yolov10n.pt",
        epochs=30,
        imgsz=480,
        batch=16,
        mosaic=0.8,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=5,
        patience=15,
    ),
    "balanced": dict(
        model="yolov10s.pt",
        epochs=80,
        imgsz=640,
        batch=8,
        mosaic=1.0,
        mixup=0.10,
        copy_paste=0.20,
        close_mosaic=10,
        patience=25,
    ),
    "best": dict(
        model="yolov10x.pt",
        epochs=150,
        imgsz=640,
        batch=8,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.30,
        close_mosaic=15,
        patience=30,
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLOv10 training for welding defect detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML,
                   help="Путь к data.yaml (по умолчанию — data.yaml в папке проекта).")
    p.add_argument("--preset", choices=list(PRESETS), default="balanced",
                   help="Готовый набор гиперпараметров. Можно переопределить флагами ниже.")
    p.add_argument("--model", default=None, help="Стартовые веса (.pt). Переопределяет preset.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None,
                   help="'cpu', 'cuda', '0', '1'. По умолчанию — автоопределение.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--name", default="welding_yolov10",
                   help="Имя эксперимента в runs/detect/.")
    p.add_argument("--project", default=None,
                   help="Куда сохранять результаты. По умолчанию ultralytics использует runs/detect/.")
    p.add_argument("--resume", action="store_true",
                   help="Возобновить с последнего last.pt.")

    # Аугментации (можно перекрыть пресет)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--box", type=float, default=7.5)
    p.add_argument("--cls", type=float, default=0.5)
    p.add_argument("--dfl", type=float, default=1.5)
    p.add_argument("--mosaic", type=float, default=None)
    p.add_argument("--mixup", type=float, default=None)
    p.add_argument("--copy-paste", type=float, default=None)
    p.add_argument("--hsv-h", type=float, default=0.015)
    p.add_argument("--hsv-s", type=float, default=0.7)
    p.add_argument("--hsv-v", type=float, default=0.4)
    p.add_argument("--degrees", type=float, default=10.0)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--shear", type=float, default=2.0)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--flipud", type=float, default=0.1)
    p.add_argument("--close-mosaic", type=int, default=None)

    p.add_argument("--no-amp", action="store_true",
                   help="Отключить mixed precision (нужно на старых CPU/GPU).")
    return p.parse_args()


def auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def estimate_time(model: str, epochs: int, n_train: int, device: str) -> str:
    """Грубая оценка времени обучения на CPU/GPU."""
    # Время forward на CPU (мс/изобр.) при imgsz=640, эмпирически
    cpu_fwd_ms = {
        "yolov10n.pt": 35, "yolov10s.pt": 60, "yolov10m.pt": 110,
        "yolov10l.pt": 180, "yolov10x.pt": 280,
        "yolov8n.pt": 30, "yolov8s.pt": 55, "yolov8m.pt": 95,
        "yolov8l.pt": 160, "yolov8x.pt": 260,
    }.get(model, 100)

    if device == "cpu":
        # Backward ≈ 2.5×forward, +валидация, аугментация → ≈4×
        sec_per_epoch = n_train * cpu_fwd_ms * 4 / 1000
    else:
        # GPU грубо в 25× быстрее
        sec_per_epoch = n_train * cpu_fwd_ms * 4 / 1000 / 25

    total_sec = sec_per_epoch * epochs
    if total_sec < 3600:
        return f"≈ {total_sec/60:.0f} мин"
    if total_sec < 86400:
        return f"≈ {total_sec/3600:.1f} ч"
    return f"≈ {total_sec/86400:.1f} суток"


def main() -> None:
    args = parse_args()

    # ── Сборка финальных гиперпараметров ──
    preset = PRESETS[args.preset].copy()

    def pick(name: str, preset_key: str | None = None):
        v = getattr(args, name)
        if v is not None:
            return v
        return preset.get(preset_key or name)

    model_name = pick("model")
    epochs = pick("epochs")
    imgsz = pick("imgsz")
    batch = pick("batch")
    mosaic = pick("mosaic")
    mixup = pick("mixup")
    copy_paste = pick("copy_paste")
    close_mosaic = pick("close_mosaic")
    patience = pick("patience")
    device = args.device or auto_device()

    # ── Проверки ──
    if not args.data.exists():
        raise SystemExit(
            f"data.yaml не найден: {args.data}\n"
            f"Ожидается рядом со скриптом: {DEFAULT_DATA_YAML}"
        )

    # Импортируем здесь, чтобы успеть показать справку до тяжёлого импорта
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    print()
    print("═" * 70)
    print("  Welding Defect Detection · Training")
    print("═" * 70)
    print(f"  Preset    : {args.preset}")
    print(f"  Model     : {model_name}")
    print(f"  Data      : {args.data}")
    print(f"  Epochs    : {epochs}    Batch: {batch}    Imgsz: {imgsz}")
    print(f"  Device    : {device}    Workers: {args.workers}")
    print(f"  Aug       : mosaic={mosaic}  mixup={mixup}  copy_paste={copy_paste}")
    project_display = args.project if args.project else "runs/detect"
    print(f"  Run       : {project_display}/{args.name}")
    print("═" * 70)

    # Проверка датасета (быстрая)
    print("\n▶ Проверка датасета…")
    info = check_det_dataset(str(args.data))
    n_train = len(list(Path(info["train"]).iterdir())) if Path(info["train"]).exists() else 0
    n_val = len(list(Path(info["val"]).iterdir())) if Path(info["val"]).exists() else 0
    print(f"  train : {n_train} изображений")
    print(f"  val   : {n_val} изображений")
    print(f"  classes: {info['names']}\n")

    # Оценка времени
    eta = estimate_time(model_name, epochs, n_train, device)
    print(f"⏱  Оценка времени обучения: {eta}\n")

    if device == "cpu" and args.preset == "best":
        print("⚠  ВНИМАНИЕ: вы выбрали пресет 'best' (YOLOv10x) на CPU.")
        print("   Реальное время может превысить несколько суток.")
        print("   Рекомендуется:")
        print("     • запустить пресет 'balanced' (yolov10s, ≈ 6–10 ч)")
        print("     • либо обучить на GPU (Google Colab, Kaggle)\n")

    # ── Запуск ──
    print(f"▶ Загрузка стартовых весов {model_name}…")
    model = YOLO(model_name)

    print(f"\n▶ Старт обучения…\n")
    train_kwargs = dict(
        data=str(args.data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=args.workers,
        name=args.name,
        patience=patience,
        resume=args.resume,
    )
    if args.project:
        train_kwargs["project"] = args.project
    model.train(
        **train_kwargs,
        optimizer="auto",
        cos_lr=True,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        # Аугментации
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        fliplr=args.fliplr,
        flipud=args.flipud,
        close_mosaic=close_mosaic,
        amp=not args.no_amp,
        cache=False,
        rect=False,
        plots=True,
        save=True,
        save_period=-1,
        val=True,
        verbose=True,
    )

    # Финальная валидация
    print("\n▶ Валидация лучшей модели…")
    metrics = model.val(data=str(args.data), imgsz=imgsz, split="val")
    print(f"\n  val  mAP50      = {metrics.box.map50:.4f}")
    print(f"  val  mAP50-95   = {metrics.box.map:.4f}")
    print(f"  val  precision  = {metrics.box.mp:.4f}")
    print(f"  val  recall     = {metrics.box.mr:.4f}\n")

    project_root = Path(args.project) if args.project else Path("runs/detect")
    best_pt = project_root / args.name / "weights" / "best.pt"
    results_csv = project_root / args.name / "results.csv"
    print(f"✓ Готово.")
    print(f"  Веса:        {best_pt}")
    print(f"  results.csv: {results_csv}")
    print(f"\nДальше:")
    print(f"  1) Десктоп-приложение:")
    print(f"     python welding_app.py")
    print(f"     → загрузить модель: {best_pt}")


if __name__ == "__main__":
    main()
