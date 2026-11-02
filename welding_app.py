"""Просмотр снимков сварных швов и результатов детектора."""

from __future__ import annotations

import csv
import queue
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageTk

from detector import (
    DEFAULT_MODEL,
    DEFAULT_PALETTE,
    CLASS_COLORS_BGR,
    Detection,
    InferenceResult,
    WeldingDetector,
)


# -----------------------------------------------------------------------------
# Тема / стили
# -----------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

BG_PRIMARY = "#1B1E20"
BG_PANEL = "#24282A"
BG_PANEL_2 = "#323739"
PREVIEW_BG = "#141718"
ACCENT = "#E6AB75"
ACCENT_HOVER = "#F0BB8F"
TEXT_PRIMARY = "#F2EFE9"
TEXT_MUTED = "#B1B9B8"
DANGER = "#F29683"
WARN = ACCENT
OK = "#9AC1AA"

CLASS_LABELS = {
    "Bad Weld": "Некачественный шов",
    "Good Weld": "Качественный шов",
    "Defect": "Дефект",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02X}{g:02X}{b:02X}"


# -----------------------------------------------------------------------------
# Главное окно
# -----------------------------------------------------------------------------
class WeldingApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Контроль сварных швов")
        width = min(1440, self.winfo_screenwidth() - 40)
        height = min(860, self.winfo_screenheight() - 90)
        self.geometry(f"{width}x{height}")
        self.minsize(960, 540)
        self.configure(fg_color=BG_PRIMARY)

        # Состояние
        self.detector: WeldingDetector | None = None
        self.model_path_var = ctk.StringVar(value=str(DEFAULT_MODEL))
        self.device_var = ctk.StringVar(value="cpu")
        self.conf_var = ctk.DoubleVar(value=0.25)
        self.iou_var = ctk.DoubleVar(value=0.50)
        self.imgsz_var = ctk.IntVar(value=640)
        self.tta_var = ctk.BooleanVar(value=False)

        self.current_image_path: Path | None = None
        self.current_result: InferenceResult | None = None
        self.batch_results: list[tuple[Path, InferenceResult]] = []
        self.batch_paths: list[Path] = []
        self.batch_index = 0

        self.preview_pil: Image.Image | None = None  # текущая PIL для предпросмотра
        self._preview_tk: ImageTk.PhotoImage | None = None

        # Очередь для безопасных обновлений UI из фоновых потоков
        self._ui_queue: queue.Queue = queue.Queue()

        self.grid_columnconfigure(0, weight=0, minsize=260)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_area()
        self._build_statusbar()

        self.after(50, self._poll_ui_queue)

    # ------------------------------------------------------------------
    # БОКОВАЯ ПАНЕЛЬ
    # ------------------------------------------------------------------
    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkScrollableFrame(self, width=240, fg_color=BG_PANEL, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            sidebar, text="Контроль сварки", anchor="w",
            font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_PRIMARY,
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(18, 0))
        ctk.CTkLabel(
            sidebar, text="Визуальный осмотр швов", anchor="w",
            font=ctk.CTkFont(size=12), text_color=TEXT_MUTED,
        ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))

        # ----- Модель -----
        self._section_label(sidebar, "Модель", row=2)
        model_frame = ctk.CTkFrame(sidebar, fg_color=BG_PANEL_2)
        model_frame.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 8))
        model_frame.grid_columnconfigure(0, weight=1)

        model_entry = ctk.CTkEntry(
            model_frame, textvariable=self.model_path_var,
            placeholder_text="путь к .pt",
            fg_color=BG_PANEL, border_width=0, height=32,
        )
        model_entry.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        self.after_idle(lambda: model_entry.xview_moveto(1))
        ctk.CTkButton(
            model_frame, text="…", width=36, height=32,
            command=self._browse_model,
            fg_color=BG_PANEL_2, hover_color="#42484A",
        ).grid(row=0, column=1, padx=(0, 8), pady=8)

        device_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        device_frame.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 8))
        device_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            device_frame, text="Устройство", text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=0, padx=(4, 8))
        ctk.CTkOptionMenu(
            device_frame, variable=self.device_var,
            values=["cpu", "cuda", "0", "1"],
            fg_color=BG_PANEL_2, button_color="#505859", button_hover_color="#626969",
            height=30,
        ).grid(row=0, column=1, sticky="ew")

        self.load_model_btn = ctk.CTkButton(
            sidebar, text="Загрузить модель",
            command=self._load_model_clicked,
            height=38, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=BG_PANEL_2, hover_color="#42484A",
        )
        self.load_model_btn.grid(row=5, column=0, sticky="ew", padx=14, pady=(2, 14))

        # ----- Пороги -----
        self._section_label(sidebar, "Параметры анализа", row=6)
        params_frame = ctk.CTkFrame(sidebar, fg_color=BG_PANEL_2)
        params_frame.grid(row=7, column=0, sticky="ew", padx=14, pady=(0, 14))
        params_frame.grid_columnconfigure(0, weight=1)

        self.conf_label = self._slider_with_label(
            params_frame, "Порог уверенности", self.conf_var, 0.05, 0.95, row=0,
        )
        self.iou_label = self._slider_with_label(
            params_frame, "Перекрытие рамок (IoU)", self.iou_var, 0.1, 0.9, row=1,
        )
        # imgsz
        imgsz_box = ctk.CTkFrame(params_frame, fg_color="transparent")
        imgsz_box.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))
        imgsz_box.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            imgsz_box, text="Размер кадра, пикс.", text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkOptionMenu(
            imgsz_box, variable=self.imgsz_var,
            values=["480", "640", "800", "960", "1280"],
            command=lambda v: self.imgsz_var.set(int(v)),
            fg_color=BG_PANEL, button_color="#505859", button_hover_color="#626969",
            height=28, width=110,
        ).grid(row=0, column=1, sticky="e")

        ctk.CTkSwitch(
            params_frame, text="Аугментация (TTA)",
            variable=self.tta_var,
            progress_color=ACCENT,
            font=ctk.CTkFont(size=11),
        ).grid(row=3, column=0, sticky="w", padx=10, pady=(2, 10))

        # ----- Действия -----
        self._section_label(sidebar, "Снимки и результаты", row=8)
        actions = ctk.CTkFrame(sidebar, fg_color="transparent")
        actions.grid(row=9, column=0, sticky="ew", padx=14, pady=(0, 14))
        actions.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            actions, text="Открыть снимок", command=self._open_image,
            height=36, text_color=PREVIEW_BG, fg_color=ACCENT, hover_color=ACCENT_HOVER,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, sticky="ew", pady=3)

        ctk.CTkButton(
            actions, text="Обработать папку", command=self._open_folder,
            height=36, fg_color=BG_PANEL_2, hover_color="#42484A",
            text_color=TEXT_PRIMARY, font=ctk.CTkFont(size=12),
        ).grid(row=1, column=0, sticky="ew", pady=3)

        ctk.CTkButton(
            actions, text="Сохранить разметку", command=self._save_result,
            height=36, fg_color=BG_PANEL_2, hover_color="#42484A",
            text_color=TEXT_PRIMARY, font=ctk.CTkFont(size=12),
        ).grid(row=2, column=0, sticky="ew", pady=3)

        ctk.CTkButton(
            actions, text="Сохранить отчёт CSV", command=self._export_csv,
            height=36, fg_color=BG_PANEL_2, hover_color="#42484A",
            text_color=TEXT_PRIMARY, font=ctk.CTkFont(size=12),
        ).grid(row=3, column=0, sticky="ew", pady=3)

    def _section_label(self, parent, text: str, row: int) -> None:
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_MUTED, anchor="w",
        ).grid(row=row, column=0, sticky="ew", padx=18, pady=(4, 4))

    def _slider_with_label(self, parent, name: str, var, vmin: float, vmax: float, row: int):
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=row, column=0, sticky="ew", padx=10, pady=(8, 0))
        box.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(box, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            top, text=name, text_color=TEXT_MUTED, font=ctk.CTkFont(size=11),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        value_lbl = ctk.CTkLabel(
            top, text=f"{var.get():.2f}", text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        value_lbl.grid(row=0, column=1, sticky="e")

        slider = ctk.CTkSlider(
            box, from_=vmin, to=vmax, variable=var,
            progress_color=ACCENT, button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            command=lambda v: value_lbl.configure(text=f"{float(v):.2f}"),
        )
        slider.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        return value_lbl

    # ------------------------------------------------------------------
    # ОСНОВНАЯ ОБЛАСТЬ
    # ------------------------------------------------------------------
    def _build_main_area(self) -> None:
        main = ctk.CTkFrame(self, fg_color=BG_PRIMARY, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=2, uniform="inspection")
        main.grid_columnconfigure(1, weight=1, uniform="inspection")
        main.grid_rowconfigure(0, weight=0)
        main.grid_rowconfigure(1, weight=1)

        # Топ-бар с метриками
        top = ctk.CTkFrame(main, fg_color=BG_PANEL, corner_radius=5)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(18, 12))
        for i in range(4):
            top.grid_columnconfigure(i, weight=1, uniform="cards")

        self.card_count = self._stat_card(top, "Найдено областей", "—", col=0)
        self.card_defects = self._stat_card(top, "С признаками дефекта", "—", col=1)
        self.card_conf = self._stat_card(top, "Средняя уверенность", "—", col=2)
        self.card_speed = self._stat_card(top, "Обработка кадра", "—", col=3)

        # Канвас для изображения
        canvas_card = ctk.CTkFrame(main, fg_color=BG_PANEL, corner_radius=5)
        canvas_card.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=(0, 20))
        canvas_card.grid_rowconfigure(1, weight=1)
        canvas_card.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(canvas_card, fg_color="transparent", height=44)
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 4))
        header.grid_columnconfigure(0, weight=1)
        self.img_title = ctk.CTkLabel(
            header, text="Снимок шва", width=1,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY, anchor="w",
        )
        self.img_title.grid(row=0, column=0, sticky="ew")

        # Кнопки навигации для батча
        nav = ctk.CTkFrame(header, fg_color="transparent")
        nav.grid(row=0, column=1, sticky="e")
        self.batch_nav = nav
        nav.grid_remove()
        self.prev_btn = ctk.CTkButton(
            nav, text="◀", width=36, height=28,
            command=self._batch_prev, fg_color=BG_PANEL_2, hover_color="#42484A",
            state="disabled",
        )
        self.prev_btn.pack(side="left", padx=2)
        self.batch_pos_lbl = ctk.CTkLabel(
            nav, text="—", text_color=TEXT_MUTED, font=ctk.CTkFont(size=11),
            width=60,
        )
        self.batch_pos_lbl.pack(side="left", padx=4)
        self.next_btn = ctk.CTkButton(
            nav, text="▶", width=36, height=28,
            command=self._batch_next, fg_color=BG_PANEL_2, hover_color="#42484A",
            state="disabled",
        )
        self.next_btn.pack(side="left", padx=2)

        self.canvas = ctk.CTkCanvas(
            canvas_card, bg=PREVIEW_BG, highlightthickness=0,
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.canvas.bind("<Configure>", lambda e: self._redraw_preview())

        # Правая колонка: детекции + классы
        right = ctk.CTkFrame(main, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=(0, 20))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=2)
        right.grid_rowconfigure(1, weight=1)

        # ── Таблица детекций ──
        det_card = ctk.CTkFrame(right, fg_color=BG_PANEL, corner_radius=5)
        det_card.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        det_card.grid_rowconfigure(1, weight=1)
        det_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            det_card, text="Отметки на снимке",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY, anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))

        self.det_list = ctk.CTkScrollableFrame(
            det_card, fg_color="transparent",
            scrollbar_button_color=BG_PANEL_2,
        )
        self.det_list.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 10))
        self.det_list.grid_columnconfigure(0, weight=1)

        self._show_empty_state()

        # ── Сводка по классам ──
        sum_card = ctk.CTkFrame(right, fg_color=BG_PANEL, corner_radius=5)
        sum_card.grid(row=1, column=0, sticky="nsew")
        sum_card.grid_columnconfigure(0, weight=1)
        sum_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            sum_card, text="По типу отметки",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY, anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))

        self.summary_frame = ctk.CTkFrame(sum_card, fg_color="transparent")
        self.summary_frame.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 12))
        self.summary_frame.grid_columnconfigure(0, weight=1)
        self._fill_summary({})

    def _stat_card(
        self, parent, title: str, value: str, col: int,
        accent: str = TEXT_PRIMARY,
    ):
        card = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=0, height=70)
        card.grid(row=0, column=col, sticky="ew", padx=4)
        card.grid_propagate(False)
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card, text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_MUTED, anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 0))

        val_lbl = ctk.CTkLabel(
            card, text=value,
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=accent, anchor="w",
        )
        val_lbl.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))
        return val_lbl

    # ------------------------------------------------------------------
    # Статус-бар
    # ------------------------------------------------------------------
    def _build_statusbar(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=BG_PANEL, height=28, corner_radius=0)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(1, weight=1)

        self.status_dot = ctk.CTkLabel(
            bar, text="●", text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=12),
        )
        self.status_dot.grid(row=0, column=0, sticky="w", padx=(12, 6))

        self.status_text = ctk.CTkLabel(
            bar, text="Модель не загружена",
            text_color=TEXT_MUTED, font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self.status_text.grid(row=0, column=1, sticky="ew")

        self.progress = ctk.CTkProgressBar(
            bar, width=200, height=6, mode="indeterminate",
            progress_color=ACCENT,
        )
        self.progress.grid(row=0, column=2, sticky="e", padx=12)
        self.progress.set(0)
        self.progress.grid_remove()

    def _set_status(self, text: str, color: str = TEXT_MUTED, dot: str | None = None) -> None:
        self.status_text.configure(text=text, text_color=color)
        if dot is not None:
            self.status_dot.configure(text_color=dot)

    # ------------------------------------------------------------------
    # Загрузка модели
    # ------------------------------------------------------------------
    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите файл весов модели",
            filetypes=[("PyTorch weights", "*.pt *.pth"), ("All", "*.*")],
        )
        if path:
            self.model_path_var.set(path)

    def _load_model_clicked(self) -> None:
        path = self.model_path_var.get().strip()
        if not path:
            messagebox.showwarning("Модель", "Укажите путь к файлу весов.")
            return
        self._set_status("Загрузка модели…", color=TEXT_PRIMARY, dot=WARN)
        self.progress.grid()
        self.progress.start()
        self.load_model_btn.configure(state="disabled")

        det = WeldingDetector(
            model_path=path, device=self.device_var.get(),
            conf=self.conf_var.get(), iou=self.iou_var.get(),
            imgsz=self.imgsz_var.get(), use_tta=self.tta_var.get(),
        )

        def worker() -> None:
            try:
                det.load()
                self._ui_queue.put(("model_loaded", det))
            except Exception as exc:  # noqa: BLE001
                self._ui_queue.put(("model_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_model_loaded(self, det: WeldingDetector) -> None:
        self.detector = det
        cls_str = ", ".join(CLASS_LABELS.get(name, name) for name in det.class_names.values()) if det.class_names else "—"
        self._set_status(
            f"Модель загружена · классы: {cls_str}",
            color=OK, dot=OK,
        )
        self.progress.stop()
        self.progress.grid_remove()
        self.load_model_btn.configure(state="normal", text="Загрузить модель")

    def _on_model_error(self, err: str) -> None:
        self._set_status(f"Ошибка загрузки модели: {err}", color=DANGER, dot=DANGER)
        self.progress.stop()
        self.progress.grid_remove()
        self.load_model_btn.configure(state="normal")
        messagebox.showerror("Модель", f"Не удалось загрузить модель:\n{err}")

    # ------------------------------------------------------------------
    # Открытие изображения или папки
    # ------------------------------------------------------------------
    def _ensure_model(self) -> bool:
        if self.detector is None or not self.detector.is_loaded():
            messagebox.showinfo(
                "Модель не загружена",
                "Загрузите модель перед анализом снимка\n(кнопка «Загрузить модель»).",
            )
            return False
        # синхронизируем пороги
        self.detector.conf = float(self.conf_var.get())
        self.detector.iou = float(self.iou_var.get())
        self.detector.imgsz = int(self.imgsz_var.get())
        self.detector.use_tta = bool(self.tta_var.get())
        return True

    def _open_image(self) -> None:
        if not self._ensure_model():
            return
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")],
        )
        if not path:
            return
        self.batch_paths = []
        self.batch_results = []
        self._update_batch_nav()
        self._run_single(Path(path))

    def _open_folder(self) -> None:
        if not self._ensure_model():
            return
        folder = filedialog.askdirectory(title="Выберите папку с изображениями")
        if not folder:
            return
        paths = sorted(
            p for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not paths:
            messagebox.showinfo("Пакет", "В папке не найдено изображений.")
            return
        self.batch_paths = paths
        self.batch_results = []
        self.batch_index = 0
        self._run_batch()

    def _run_single(self, path: Path) -> None:
        self._set_status("Анализ снимка…", color=TEXT_PRIMARY, dot=WARN)
        self.progress.grid()
        self.progress.start()

        def worker() -> None:
            try:
                assert self.detector is not None
                result = self.detector.predict_image(path, draw=True)
                self._ui_queue.put(("single_done", (path, result)))
            except Exception as exc:  # noqa: BLE001
                self._ui_queue.put(("single_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _run_batch(self) -> None:
        self._set_status(
            f"Пакетная обработка: 0 / {len(self.batch_paths)}",
            color=TEXT_PRIMARY, dot=WARN,
        )
        self.progress.grid()
        self.progress.start()

        paths = list(self.batch_paths)

        def worker() -> None:
            try:
                assert self.detector is not None
                results: list[tuple[Path, InferenceResult]] = []
                for i, p in enumerate(paths, start=1):
                    res = self.detector.predict_image(p, draw=True)
                    results.append((p, res))
                    self._ui_queue.put((
                        "batch_progress",
                        (i, len(paths)),
                    ))
                self._ui_queue.put(("batch_done", results))
            except Exception as exc:  # noqa: BLE001
                self._ui_queue.put(("batch_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Очередь UI
    # ------------------------------------------------------------------
    def _poll_ui_queue(self) -> None:
        try:
            while True:
                kind, payload = self._ui_queue.get_nowait()
                self._handle_ui_event(kind, payload)
        except queue.Empty:
            pass
        self.after(40, self._poll_ui_queue)

    def _handle_ui_event(self, kind: str, payload) -> None:
        if kind == "model_loaded":
            self._on_model_loaded(payload)
        elif kind == "model_error":
            self._on_model_error(payload)
        elif kind == "single_done":
            path, result = payload
            self._show_result(path, result)
            self._set_status(
                f"Областей на снимке: {len(result.detections)}",
                color=OK, dot=OK,
            )
            self.progress.stop()
            self.progress.grid_remove()
        elif kind == "single_error":
            self._set_status(f"Ошибка: {payload}", color=DANGER, dot=DANGER)
            self.progress.stop()
            self.progress.grid_remove()
            messagebox.showerror("Анализ снимка", str(payload))
        elif kind == "batch_progress":
            i, n = payload
            self._set_status(
                f"Пакетная обработка: {i} / {n}",
                color=TEXT_PRIMARY, dot=WARN,
            )
        elif kind == "batch_done":
            self.batch_results = payload
            self.batch_index = 0
            self.progress.stop()
            self.progress.grid_remove()
            total_objs = sum(len(r.detections) for _, r in payload)
            self._set_status(
                f"Пакет готов: {len(payload)} изображений, {total_objs} объектов",
                color=OK, dot=OK,
            )
            if payload:
                p, r = payload[0]
                self._show_result(p, r)
            self._update_batch_nav()
        elif kind == "batch_error":
            self.progress.stop()
            self.progress.grid_remove()
            self._set_status(f"Ошибка обработки папки: {payload}", color=DANGER, dot=DANGER)
            messagebox.showerror("Обработка папки", str(payload))

    # ------------------------------------------------------------------
    # Отображение результата
    # ------------------------------------------------------------------
    def _show_result(
        self,
        path: Path,
        result: InferenceResult,
    ) -> None:
        self.current_result = result
        self.current_image_path = path
        self.img_title.configure(text=path.name)
        if result.annotated_image is not None:
            self.preview_pil = Image.fromarray(result.annotated_image)
            self._redraw_preview()

        # Статистика
        n_total = len(result.detections)
        n_defects = sum(
            1 for d in result.detections
            if d.class_name.lower() in {"defect", "bad weld", "porosity", "crack", "spatter", "undercut"}
        )
        if n_total:
            avg_conf = sum(d.confidence for d in result.detections) / n_total
            self.card_conf.configure(text=f"{avg_conf:.0%}")
        else:
            self.card_conf.configure(text="—")

        self.card_count.configure(text=str(n_total))
        self.card_defects.configure(text=str(n_defects), text_color=DANGER if n_defects else TEXT_PRIMARY)
        self.card_speed.configure(
            text=f"{result.total_ms:.0f} мс",
        )

        self._fill_detections(result.detections)
        self._fill_summary(result.count_by_class)

    def _redraw_preview(self) -> None:
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 4 or h <= 4:
            return
        self.canvas.delete("all")
        if self.preview_pil is None:
            self.canvas.create_text(
                w // 2, h // 2, text="Откройте снимок для осмотра",
                fill=TEXT_MUTED, font=("Segoe UI", 13),
            )
            return
        img = self.preview_pil
        iw, ih = img.size
        scale = min(w / iw, h / ih)
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        self._preview_tk = ImageTk.PhotoImage(resized)

        self.canvas.create_image(w // 2, h // 2, image=self._preview_tk, anchor="center")

    def _show_empty_state(self, text: str = "Снимок ещё не обработан") -> None:
        for w in self.det_list.winfo_children():
            w.destroy()
        ph = ctk.CTkLabel(
            self.det_list, text=text,
            text_color=TEXT_MUTED, font=ctk.CTkFont(size=11),
        )
        ph.grid(row=0, column=0, sticky="ew", padx=10, pady=24)

    def _fill_detections(self, detections: list[Detection]) -> None:
        for w in self.det_list.winfo_children():
            w.destroy()
        if not detections:
            self._show_empty_state("Нет отметок при выбранном пороге")
            return

        # Сортируем по уверенности (по убыванию)
        for i, det in enumerate(sorted(detections, key=lambda d: -d.confidence)):
            row = ctk.CTkFrame(self.det_list, fg_color=BG_PANEL_2, corner_radius=8)
            row.grid(row=i, column=0, sticky="ew", padx=4, pady=3)
            row.grid_columnconfigure(1, weight=1)

            # Цветной индикатор
            color_hex = bgr_to_hex(CLASS_COLORS_BGR.get(
                det.class_name,
                DEFAULT_PALETTE[det.class_id % len(DEFAULT_PALETTE)],
            ))
            dot = ctk.CTkFrame(row, fg_color=color_hex, width=3, height=1, corner_radius=1)
            dot.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(8, 8), pady=8)
            dot.grid_propagate(False)

            ctk.CTkLabel(
                row, text=CLASS_LABELS.get(det.class_name, det.class_name),
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=TEXT_PRIMARY, anchor="w",
            ).grid(row=0, column=1, sticky="ew", pady=(8, 0))

            x1, y1, x2, y2 = det.box_xyxy
            sub = (f"Уверенность {det.confidence:.0%}   ·   "
                   f"{int(x2 - x1)} × {int(y2 - y1)} пикс.")
            ctk.CTkLabel(
                row, text=sub,
                font=ctk.CTkFont(size=10),
                text_color=TEXT_MUTED, anchor="w",
            ).grid(row=1, column=1, sticky="ew", pady=(0, 8))

    def _fill_summary(self, counts: dict[str, int]) -> None:
        for w in self.summary_frame.winfo_children():
            w.destroy()
        if not counts:
            ctk.CTkLabel(
                self.summary_frame, text="Нет данных",
                text_color=TEXT_MUTED, font=ctk.CTkFont(size=11),
            ).grid(row=0, column=0, sticky="ew", pady=20)
            return

        total = sum(counts.values()) or 1
        for i, (cls, n) in enumerate(sorted(counts.items(), key=lambda kv: -kv[1])):
            color_hex = bgr_to_hex(CLASS_COLORS_BGR.get(
                cls, DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)],
            ))

            row = ctk.CTkFrame(self.summary_frame, fg_color="transparent")
            row.grid(row=i, column=0, sticky="ew", pady=4)
            row.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(
                row, text=CLASS_LABELS.get(cls, cls),
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=TEXT_PRIMARY, anchor="w", width=125,
            ).grid(row=0, column=0, sticky="w")

            bar_bg = ctk.CTkFrame(row, fg_color=BG_PANEL_2, height=8, corner_radius=4)
            bar_bg.grid(row=0, column=1, sticky="ew", padx=8)
            bar_bg.grid_propagate(False)
            row.update_idletasks()
            bar_bg.update_idletasks()
            # Используем относительную ширину через place
            ctk.CTkFrame(
                bar_bg, fg_color=color_hex, height=8, corner_radius=4,
            ).place(relx=0, rely=0, relwidth=n / total, relheight=1.0)

            ctk.CTkLabel(
                row, text=str(n),
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=color_hex, width=30,
            ).grid(row=0, column=2, sticky="e")

    # ------------------------------------------------------------------
    # Навигация по батчу
    # ------------------------------------------------------------------
    def _update_batch_nav(self) -> None:
        if not self.batch_results:
            self.batch_nav.grid_remove()
            return
        self.batch_nav.grid()
        n = len(self.batch_results)
        self.batch_pos_lbl.configure(text=f"{self.batch_index + 1} / {n}")
        self.prev_btn.configure(state="normal" if self.batch_index > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.batch_index < n - 1 else "disabled")

    def _batch_prev(self) -> None:
        if self.batch_index > 0:
            self.batch_index -= 1
            self._show_batch_current()

    def _batch_next(self) -> None:
        if self.batch_index < len(self.batch_results) - 1:
            self.batch_index += 1
            self._show_batch_current()

    def _show_batch_current(self) -> None:
        p, r = self.batch_results[self.batch_index]
        self._show_result(p, r)
        self._update_batch_nav()

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------
    def _save_result(self) -> None:
        if self.current_result is None or self.current_result.annotated_image is None:
            messagebox.showinfo("Сохранение", "Сначала обработайте снимок.")
            return
        default_name = "result.jpg"
        if self.current_image_path:
            default_name = f"{self.current_image_path.stem}_annotated.jpg"
        path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
        )
        if not path:
            return
        try:
            Image.fromarray(self.current_result.annotated_image).save(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Сохранение", str(exc))
            return
        self._set_status(f"Сохранено: {path}", color=OK, dot=OK)

    def _export_csv(self) -> None:
        if not self.batch_results:
            messagebox.showinfo(
                "Экспорт",
                "Нет данных пакета. Сначала запустите пакетную обработку.",
            )
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить CSV",
            defaultextension=".csv",
            initialfile="detections.csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "file", "class_id", "class_name", "confidence",
                "x1", "y1", "x2", "y2", "area_px",
                "inference_ms", "total_ms",
            ])
            for img_path, res in self.batch_results:
                if not res.detections:
                    w.writerow([
                        img_path.name, "", "", "",
                        "", "", "", "", "",
                        f"{res.inference_ms:.2f}", f"{res.total_ms:.2f}",
                    ])
                for det in res.detections:
                    x1, y1, x2, y2 = det.box_xyxy
                    w.writerow([
                        img_path.name, det.class_id, det.class_name,
                        f"{det.confidence:.4f}",
                        f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                        f"{det.area:.0f}",
                        f"{res.inference_ms:.2f}", f"{res.total_ms:.2f}",
                    ])
        self._set_status(f"CSV сохранён: {path}", color=OK, dot=OK)


def main() -> None:
    app = WeldingApp()
    app.mainloop()


if __name__ == "__main__":
    main()
