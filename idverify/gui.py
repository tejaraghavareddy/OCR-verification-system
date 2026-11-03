"""Desktop interface (Tkinter) for IDVerify.

Runs the same :class:`~idverify.engine.IdentitySystem` as the CLI, on the local
machine: camera frames, card images and the SQLite audit log all stay on disk.
Heavy work (OCR + face models) happens on a worker thread so the window never
freezes; results are posted back with ``root.after``.

The window opens with a short intro, a clear step-by-step guide for the Verify
and Enroll tabs, a live camera preview when you ask for one, and a status bar
that actually names what the app is doing.

If the face models are missing, the app tells you at startup and points at the
one command that fixes it. If ``tkinter`` itself is missing, the CLI owns that
message and points at the package manager command.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from . import __version__, config
from .camera import Camera, CameraError
from .engine import (
    REJECTED,
    REVIEW,
    VERIFIED,
    EnrollError,
    IdentitySystem,
    VERDICT_LABELS,
)
from .face import FaceError, ModelMissingError
from .idcard import FIELD_LABELS
from .util import load_image

VERDICT_COLORS = {
    VERIFIED: ("#0b6b2f", "#d9f2e3"),
    REJECTED: ("#8f1d1d", "#fbdede"),
    REVIEW: ("#8a5a00", "#fdf0d5"),
}

PREVIEW_W, PREVIEW_H = 260, 174
IMAGE_TYPES = [
    ("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
    ("All files", "*.*"),
]

LOGO = r"""
   ___               _
  |_ _|_ __ ___  ___| | __
   | || '_ ` _ \/ __| |/ /  offline identity verification
   | || | | | | \__ \   <
  |___|_| |_| |_|___/_|\_\
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_photo(
    bgr: np.ndarray, size: tuple[int, int] = (PREVIEW_W, PREVIEW_H)
) -> ImageTk.PhotoImage:
    """Best-effort conversion of a BGR array into a Tk photo for a preview."""
    if bgr is None:
        return _blank_photo(size)
    if bgr.size == 0:
        return _blank_photo(size)
    rgb = bgr[:, :, ::-1] if bgr.ndim == 3 else np.stack([bgr] * 3, axis=-1)
    rgb = np.ascontiguousarray(rgb)
    image = Image.fromarray(rgb)
    image.thumbnail(size, Image.LANCZOS)
    return ImageTk.PhotoImage(image)


def _blank_photo(size: tuple[int, int] = (PREVIEW_W, PREVIEW_H)) -> ImageTk.PhotoImage:
    blank = np.full((size[1], size[0], 3), (16, 20, 24), dtype=np.uint8)
    return _to_photo(blank, size)


def _safe_filename(path: str) -> str:
    if not path:
        return ""
    return path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


# --------------------------------------------------------------------------- #
# Live camera preview panel
# --------------------------------------------------------------------------- #
class LiveCameraPanel(ttk.LabelFrame):
    """A camera preview that starts when you click "Start camera" and stops when
    you click "Stop", or when the tab changes. Useful for aiming the card and
    the face before capturing them.

    Under the hood this keeps one :class:`Camera` open and feeds frames from a
    Tkinter ``after()`` timer so the preview actually moves.
    """

    REFRESH_MS = 60

    def __init__(self, master, app: "App") -> None:
        super().__init__(master, text="Live camera", padding=8)
        self.app = app
        self.camera: Optional[Camera] = None
        self._timer_id: Optional[str] = None

        self.status_label = ttk.Label(self, text="Camera is off", foreground="#555")
        self.status_label.pack(anchor="w")

        self.preview = tk.Label(
            self,
            width=PREVIEW_W,
            height=PREVIEW_H,
            relief="sunken",
            bg="#101418",
        )
        self.preview.pack(pady=6)
        self._show_blank()

        buttons = ttk.Frame(self)
        buttons.pack(fill="x")
        self.start_btn = ttk.Button(
            buttons, text="Start camera", command=self.start
        )
        self.start_btn.pack(side="left", expand=True, fill="x")
        self.stop_btn = ttk.Button(
            buttons, text="Stop", command=self.stop, state="disabled"
        )
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=(6, 0))

    # -- preview lifecycle ----------------------------------------------\
    def start(self) -> None:
        if self.camera is not None:
            self.stop()
        index = self.app.camera_index.get()
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = 0
        self.app.run_async(
            lambda: Camera(index).start_stream(),
            self._on_stream_ready,
            busy_msg=f"Opening camera {index}...",
        )

    def _on_stream_ready(self, value: Any, error: Optional[BaseException]) -> None:
        if error:
            self._set_status(str(error), "#8f1d1d")
            return
        camera = value if isinstance(value, Camera) else None
        if camera is None or not camera.is_open:
            self._set_status("Could not open the camera.", "#8f1d1d")
            return
        self.camera = camera
        self._set_status(f"Camera {camera.index} — live", "#0b6b2f")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._tick()

    def stop(self) -> None:
        self._stop_timer()
        if self.camera is not None:
            try:
                self.camera.stop_stream()
            except Exception:
                pass
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._show_blank()
        self._set_status("Camera is off", "#555")

    def _tick(self) -> None:
        if self.camera is None:
            return
        frame = self.camera.stream_frame()
        if frame is None:
            # Keep trying. Some cameras need a few empty reads before they
            # produce a stable preview.
            self._set_status("Waiting for camera...", "#555")
            self._show_blank()
        else:
            self._set_status(
                f"Camera {self.camera.index} — {frame.shape[1]}x{frame.shape[0]}",
                "#0b6b2f",
            )
            try:
                photo = _to_photo(frame)
            except Exception:
                photo = _blank_photo()
            self.preview.configure(image=photo)
            self.preview.image = photo
        self._timer_id = self.after(self.REFRESH_MS, self._tick)

    def _stop_timer(self) -> None:
        if self._timer_id is not None:
            try:
                self.after_cancel(self._timer_id)
            except Exception:
                pass
            self._timer_id = None

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.configure(text=text, foreground=color)

    def _show_blank(self) -> None:
        photo = _blank_photo()
        self.preview.configure(image=photo)
        self.preview.image = photo

    def destroy(self) -> None:
        self._stop_timer()
        self.stop()
        super().destroy()


# --------------------------------------------------------------------------- #
# Static capture panel (one card or one face, camera snapshot or file)
# --------------------------------------------------------------------------- #
class CapturePanel(ttk.LabelFrame):
    """One input slot: take a picture from the camera or load a file."""

    def __init__(self, master, title: str, app: "App") -> None:
        super().__init__(master, text=title, padding=8)
        self.app = app
        self.image: Any = None  # np.ndarray (camera) or str path (file)
        self.source_label = ttk.Label(self, text="No image selected", anchor="center")
        self.source_label.pack(fill="x")

        self.preview = tk.Label(
            self, width=PREVIEW_W, height=PREVIEW_H, relief="sunken", bg="#101418"
        )
        self.preview.pack(pady=6)
        self._show_blank()

        buttons = ttk.Frame(self)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Camera", command=self.from_camera).pack(
            side="left", expand=True, fill="x"
        )
        ttk.Button(
            buttons, text="File", command=self.from_file
        ).pack(side="left", expand=True, fill="x", padx=(6, 0))

    # -- inputs ----------------------------------------------------------
    def from_camera(self) -> None:
        index = self.app.camera_index.get()
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = 0
        self.app.run_async(
            lambda: Camera(index).read(),
            self._set_capture,
            busy_msg=f"Reading camera {index}...",
        )

    def _set_capture(self, frame: Any, error: Optional[BaseException]) -> None:
        if error:
            self.app.prestyle_messagebox("Camera", str(error))
            return
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            self.app.prestyle_messagebox(
                "Camera", "The camera did not return an image."
            )
            return
        self.set_image(frame, f"Camera frame ({frame.shape[1]}x{frame.shape[0]})")

    def from_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image", filetypes=IMAGE_TYPES
        )
        if not path:
            return
        try:
            image = load_image(path)
        except (OSError, ValueError) as exc:
            self.app.prestyle_messagebox("Image", str(exc))
            return
        self.set_image(path, _safe_filename(path))

    def set_image(self, image: Any, label: str) -> None:
        self.image = image
        self.source_label.configure(text=label)
        try:
            photo = _to_photo(load_image(image))
        except (OSError, ValueError):
            return
        self.preview.configure(image=photo)
        self.preview.image = photo  # keep a reference

    def clear(self) -> None:
        self.image = None
        self.source_label.configure(text="No image selected")
        self._show_blank()

    def _show_blank(self) -> None:
        photo = _blank_photo()
        self.preview.configure(image=photo)
        self.preview.image = photo

    @property
    def ready(self) -> bool:
        return self.image is not None and (
            isinstance(self.image, np.ndarray) and self.image.size > 0
        )


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
class VerifyTab(ttk.Frame):
    def __init__(self, master, app: "App") -> None:
        super().__init__(master, padding=10)
        self.app = app

        intro = ttk.Label(
            self,
            text=(
                "1. Add the ID card (camera snapshot or file).\n"
                "2. Add a live capture of the person (camera snapshot or file).\n"
                "3. Press VERIFY IDENTITY.\n\n"
                "The app compares the face in the live capture against the portrait\n"
                "printed on the card, then against the record stored locally for\n"
                "that ID number."
            ),
            justify="left",
            foreground="#333",
            font=("", 10),
            padding=(0, 0, 0, 10),
        )
        intro.pack(fill="x")

        inputs = ttk.Frame(self)
        inputs.pack(side="left", fill="y", padx=(0, 12))
        self.card = CapturePanel(inputs, "1 - ID card", app)
        self.card.pack(fill="x")
        self.face = CapturePanel(inputs, "2 - Live capture", app)
        self.face.pack(fill="x", pady=(10, 0))

        action = ttk.Frame(inputs)
        action.pack(fill="x", pady=(12, 0))
        ttk.Button(action, text="VERIFY IDENTITY", command=self.start).pack(
            fill="x", ipady=7
        )
        ttk.Button(action, text="Clear", command=self.clear).pack(
            fill="x", padx=(6, 0), ipady=7
        )

        self.result = ttk.Frame(self, padding=(6, 0))
        self.result.pack(side="right", fill="both", expand=True)
        self._build_empty()

    def _build_empty(self) -> None:
        for child in self.result.winfo_children():
            child.destroy()
        ttk.Label(
            self.result,
            text="Result will appear here.",
            foreground="#555",
            font=("", 11),
        ).pack(anchor="w", pady=(0, 8))
        ttk.Label(
            self.result,
            text=(
                "Add the two images on the left and press VERIFY IDENTITY.\n"
                "VERIFIED means the live face matched the card and, when available,\n"
                "the local record. REJECTED means the check failed.\n"
                "NEEDS REVIEW means the scores are in the uncertain band."
            ),
            foreground="#777",
            justify="left",
        ).pack(anchor="w")

    def start(self) -> None:
        if not self.card.ready or not self.face.ready:
            self.app.prestyle_messagebox(
                "Verify",
                "Add both the ID card image and the live capture first.",
            )
            return
        card, face = self.card.image, self.face.image
        self.app.run_async(
            lambda: self.app.system().verify(card, face),
            self.show,
            busy_msg="Reading card and matching faces...",
        )

    def show(self, result: Any, error: Optional[BaseException]) -> None:
        if error:
            self.app.prestyle_messagebox("Verify", str(error))
            return
        for child in self.result.winfo_children():
            child.destroy()

        fg, bg = VERDICT_COLORS.get(result.verdict, ("#333", "#eee"))
        banner = tk.Frame(self.result, bg=bg, padx=16, pady=12)
        banner.pack(fill="x")
        tk.Label(
            banner,
            text=VERDICT_LABELS.get(result.verdict, result.verdict),
            bg=bg,
            fg=fg,
            font=("", 22, "bold"),
        ).pack(anchor="w")
        tk.Label(
            banner,
            text=result.basis,
            bg=bg,
            fg=fg,
            font=("", 10),
        ).pack(anchor="w")

        fields = ttk.Frame(self.result, padding=(0, 14))
        fields.pack(fill="x")
        rows = [
            ("ID number", result.card_fields.label("id_number")),
            ("Name on card", result.card_fields.label("full_name")),
            ("Date of birth", result.card_fields.label("date_of_birth")),
            (
                "Date of expiry",
                result.card_fields.label("date_of_expiry")
                + ("  (EXPIRED)" if result.card_expired else ""),
            ),
            ("Card portrait check", _score(result.live_vs_card)),
            ("Local record check", _score(result.live_vs_enrolled)),
        ]
        if result.person is not None:
            rows.append(("Record name", result.person.display_name))
        for index, (label, value) in enumerate(rows):
            ttk.Label(fields, text=label, foreground="#666").grid(
                row=index, column=0, sticky="w", pady=1
            )
            ttk.Label(
                fields, text=value, font=("", 11, "bold")
            ).grid(row=index, column=1, sticky="w", padx=(18, 0), pady=1)
        fields.columnconfigure(1, weight=1)

        notes = tk.Text(
            self.result, height=7, wrap="word", relief="flat", bg="#f7f7f7"
        )
        notes.pack(fill="both", expand=True)
        for reason in result.reasons:
            notes.insert("end", f"• {reason}\n")
        for warning in result.warnings:
            notes.insert("end", f"! {warning}\n", "warn")
        notes.tag_configure("warn", foreground="#8a5a00")
        notes.configure(state="disabled")

        if result.card_fields.raw_lines:
            detail = "OCR: " + " | ".join(result.card_fields.raw_lines[:8])
            ttk.Label(
                self.result,
                text=detail,
                foreground="#999",
                wraplength=520,
            ).pack(anchor="w", pady=(6, 0))

    def clear(self) -> None:
        self.card.clear()
        self.face.clear()
        self._build_empty()


class EnrollTab(ttk.Frame):
    def __init__(self, master, app: "App") -> None:
        super().__init__(master, padding=10)
        self.app = app

        intro = ttk.Label(
            self,
            text=(
                "1. Add the ID card (camera snapshot or file).\n"
                "2. Add a live capture of the person (camera snapshot or file).\n"
                "3. Press ENROLL IDENTITY.\n\n"
                "Enrolling stores the ID number, the printed portrait and a face\n"
                "descriptor in the local database. The live face must match the\n"
                "printed portrait before anything is stored."
            ),
            justify="left",
            foreground="#333",
            font=("", 10),
            padding=(0, 0, 0, 10),
        )
        intro.pack(fill="x")

        inputs = ttk.Frame(self)
        inputs.pack(side="left", fill="y", padx=(0, 12))
        self.card = CapturePanel(inputs, "1 - ID card", app)
        self.card.pack(fill="x")
        self.face = CapturePanel(
            inputs, "2 - Live capture of the holder", app
        )
        self.face.pack(fill="x", pady=(10, 0))

        action = ttk.Frame(inputs)
        action.pack(fill="x", pady=(12, 0))
        ttk.Button(action, text="ENROLL IDENTITY", command=self.start).pack(
            fill="x", ipady=7
        )
        ttk.Button(action, text="Clear", command=self.clear).pack(
            fill="x", padx=(6, 0), ipady=7
        )

        self.message = tk.Text(
            self, wrap="word", relief="flat", bg="#f7f7f7", state="disabled"
        )
        self.message.pack(side="right", fill="both", expand=True, padx=(6, 0))
        self._set(
            "Add the card and the live capture on the left, then press\n"
            "ENROLL IDENTITY."
        )

    def _set(self, text: str) -> None:
        self.message.configure(state="normal")
        self.message.delete("1.0", "end")
        self.message.insert("1.0", text)
        self.message.configure(state="disabled")

    def start(self) -> None:
        if not self.card.ready or not self.face.ready:
            self.app.prestyle_messagebox(
                "Enroll",
                "Add both the ID card image and the live capture first.",
            )
            return
        card, face = self.card.image, self.face.image
        self.app.run_async(
            lambda: self.app.system().enroll(card, face),
            self.show,
            busy_msg="Enrolling...",
        )

    def show(self, result: Any, error: Optional[BaseException]) -> None:
        if isinstance(error, EnrollError):
            self._set("REFUSED\n\n" + str(error))
            return
        if error:
            self.app.prestyle_messagebox("Enroll", str(error))
            return
        verb = "Updated" if result.updated else "Enrolled"
        self._set(
            f"{verb}: {result.id_number}\n"
            f"Name: {result.full_name or '-'}\n"
            f"Live face vs printed portrait: {result.live_vs_card:.3f}\n"
            f"Portrait saved to: {result.portrait_path}\n"
            + (
                "\nWarnings:\n" + "\n".join(f"• {w}" for w in result.warnings)
                if result.warnings
                else ""
            )
        )
        self.app.refresh_records()


class RecordsTab(ttk.Frame):
    COLUMNS = ("id_number", "name", "dob", "expiry", "updated")

    def __init__(self, master, app: "App") -> None:
        super().__init__(master, padding=10)
        self.app = app
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="left")
        ttk.Button(
            bar, text="Forget selected", command=self.forget
        ).pack(side="left", padx=(8, 0))

        self.tree = ttk.Treeview(
            self, columns=self.COLUMNS, show="headings", height=18
        )
        headings = {
            "id_number": ("ID number", 150),
            "name": ("Name", 200),
            "dob": ("Date of birth", 120),
            "expiry": ("Expiry", 120),
            "updated": ("Updated (UTC)", 190),
        }
        for key, (title, width) in headings.items():
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True, pady=(10, 0))

    def refresh(self) -> None:
        self.app.run_async(
            lambda: self.app.system().people(),
            self._fill,
            busy_msg="Loading records...",
        )

    def _fill(self, people: Any, error: Optional[BaseException]) -> None:
        if error:
            self.app.prestyle_messagebox("Records", str(error))
            return
        self.tree.delete(*self.tree.get_children())
        for person in people:
            self.tree.insert(
                "",
                "end",
                values=(
                    person.id_number,
                    person.display_name,
                    person.date_of_birth or "-",
                    person.date_of_expiry or "-",
                    person.updated_at,
                ),
            )

    def forget(self) -> None:
        selection = self.tree.selection()
        if not selection:
            self.app.prestyle_messagebox("Records", "Select a record first.")
            return
        id_number = self.tree.item(selection[0], "values")[0]
        if not messagebox.askyesno(
            "Forget",
            f"Remove {id_number} from the local database?",
            parent=self.app,
        ):
            return
        self.app.run_async(
            lambda: self.app.system().forget(id_number),
            lambda _ok, err: (
                self.refresh()
                if err is None
                else self.app.prestyle_messagebox("Records", str(err))
            ),
            busy_msg="Removing...",
        )


class HistoryTab(ttk.Frame):
    COLUMNS = ("when", "result", "id_number", "name", "card", "record")

    def __init__(self, master, app: "App") -> None:
        super().__init__(master, padding=10)
        self.app = app
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="left")
        ttk.Label(
            bar, text=f"Local log: {config.home_dir() / 'idverify.db'}"
        ).pack(side="left", padx=(10, 0))

        self.tree = ttk.Treeview(
            self, columns=self.COLUMNS, show="headings", height=18
        )
        headings = {
            "when": ("When (UTC)", 190),
            "result": ("Result", 110),
            "id_number": ("ID number", 150),
            "name": ("Name", 170),
            "card": ("Card", 70),
            "record": ("Record", 80),
        }
        for key, (title, width) in headings.items():
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor="w")
        self.tree.tag_configure(REJECTED, foreground="#8f1d1d")
        self.tree.tag_configure(VERIFIED, foreground="#0b6b2f")
        self.tree.tag_configure(REVIEW, foreground="#8a5a00")
        self.tree.pack(fill="both", expand=True, pady=(10, 0))

    def refresh(self) -> None:
        self.app.run_async(
            lambda: self.app.system().history(200),
            self._fill,
            busy_msg="Loading history...",
        )

    def _fill(self, entries: Any, error: Optional[BaseException]) -> None:
        if error:
            self.app.prestyle_messagebox("History", str(error))
            return
        self.tree.delete(*self.tree.get_children())
        for entry in entries:
            self.tree.insert(
                "",
                "end",
                tags=(entry.result,),
                values=(
                    entry.created_at,
                    entry.result,
                    entry.id_number or "-",
                    entry.full_name or "-",
                    _score(entry.live_vs_card),
                    _score(entry.live_vs_enrolled),
                ),
            )


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class App(tk.Tk):
    def __init__(self, home: Optional[str] = None) -> None:
        super().__init__()
        self.title(f"IDVerify {__version__} — offline identity verification")
        self.geometry("1120x720")
        self.minsize(940, 620)
        self._home = home
        self._system: Optional[IdentitySystem] = None
        self._busy = False
        self._on_close_requested = False

        self._build_header()
        self._build_body()
        self._build_statusbar()
        self._check_models_on_startup()

    # -- header ----------------------------------------------------------
    def _build_header(self) -> None:
        header = tk.Frame(self, bg="#14263c", padx=12, pady=10)
        header.pack(fill="x")

        logo = tk.Text(
            header,
            height=3,
            width=12,
            bg="#14263c",
            fg="#9fc4e8",
            borderwidth=0,
            highlightthickness=0,
            font=("", 8),
            padx=4,
            pady=2,
        )
        logo.pack(side="left")
        logo.insert("1.0", LOGO)
        logo.configure(state="disabled", wrap="none")

        tk.Label(
            header,
            text="IDVerify",
            bg="#14263c",
            fg="white",
            font=("", 16, "bold"),
        ).pack(side="left", padx=(14, 0))

        tk.Label(
            header,
            text="  no network — all data stays on this machine",
            bg="#14263c",
            fg="#9fc4e8",
            font=("", 10),
        ).pack(side="left")

        camera_box = ttk.Frame(header)
        camera_box.pack(side="right")
        ttk.Label(camera_box, text="Camera index").pack(side="left")
        self.camera_index = ttk.Spinbox(camera_box, from_=0, to=9, width=3)
        self.camera_index.pack(side="left", padx=(6, 0))
        ttk.Label(
            camera_box,
            text=f"Data: {config.home_dir()}",
            foreground="#9fc4e8",
            font=("", 9),
        ).pack(side="right")

    # -- body ------------------------------------------------------------
    def _build_body(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=6, pady=(4, 0))
        self.verify_tab = VerifyTab(notebook, self)
        self.enroll_tab = EnrollTab(notebook, self)
        self.records_tab = RecordsTab(notebook, self)
        self.history_tab = HistoryTab(notebook, self)
        notebook.add(self.verify_tab, text="  Verify  ")
        notebook.add(self.enroll_tab, text="  Enroll  ")
        notebook.add(self.records_tab, text="  Records  ")
        notebook.add(self.history_tab, text="  History  ")
        self.notebook = notebook
        notebook.bind("<<NotebookTabChanged>>", self._on_tab)

        # The live camera preview lives on the Verify tab so it is easy to
        # reach while aiming the card and the face.
        self.verify_tab.live_camera = LiveCameraPanel(
            self.verify_tab, self
        )
        self.verify_tab.live_camera.pack(fill="x", pady=(0, 10))

        self.after(150, self.history_tab.refresh)

    # -- status bar ------------------------------------------------------
    def _build_statusbar(self) -> None:
        statusbar = tk.Frame(self, bg="#eeeeee", padx=10, pady=5)
        statusbar.pack(side="bottom", fill="x")
        self.status = tk.StringVar(value="Ready")
        ttk.Label(statusbar, textvariable=self.status).pack(side="left")
        ttk.Label(
            statusbar,
            text="IDVerify — offline",
            foreground="#999",
            font=("", 9),
        ).pack(side="right")

    # -- startup model check ---------------------------------------------
    def _check_models_on_startup(self) -> None:
        try:
            from .face import FaceEngine

            FaceEngine()
        except ModelMissingError as exc:
            self.after(
                0,
                lambda: messagebox.showwarning(
                    "Face models needed",
                    "IDVerify could not find the face models it needs to run.\n\n"
                    f"{exc}\n\n"
                    "If you downloaded the models, point the IDVERIFY_MODELS\n"
                    "environment variable at the folder that holds:\n"
                    f"  • {config.DETECTOR_FILE}\n"
                    f"  • {config.RECOGNIZER_FILE}",
                    parent=self,
                ),
            )

    # -- shared system ---------------------------------------------------
    def system(self) -> IdentitySystem:
        if self._system is None:
            self._system = IdentitySystem(home=self._home)
        return self._system

    def _on_tab(self, _event: Any) -> None:
        current = self.notebook.select()
        name = current.winfo_name() if hasattr(current, "winfo_name") else ""
        if name.endswith("records_tab"):
            self.records_tab.refresh()
        elif name.endswith("history_tab"):
            self.history_tab.refresh()

    # -- worker helper ---------------------------------------------------
    def run_async(
        self,
        work: Callable[[], Any],
        done: Callable[[Any, Optional[BaseException]], None],
        busy_msg: str = "Working...",
    ) -> None:
        if self._busy:
            return
        self._busy = True
        self.status.set(busy_msg)

        def runner() -> None:
            try:
                value: Any = work()
                failure: Optional[BaseException] = None
            except BaseException as exc:  # surfaced to the UI, never swallowed
                value, failure = None, exc
            self.after(0, lambda: self._finish(value, failure, done))

        threading.Thread(target=runner, daemon=True).start()

    def _finish(
        self,
        value: Any,
        failure: Optional[BaseException],
        done: Callable[[Any, Optional[BaseException]], None],
    ) -> None:
        self._busy = False
        self.status.set("Ready")
        if isinstance(failure, ModelMissingError):
            self.prestyle_messagebox("Models", str(failure))
            return
        done(value, failure)

    def prestyle_messagebox(self, title: str, message: str) -> None:
        """Messagebox styled to the app's look-and-feel."""
        messagebox.showerror(title, message, parent=self)

    def refresh_records(self) -> None:
        self.records_tab.refresh()

    def on_closing(self) -> None:
        if self._on_close_requested:
            self.destroy()
            return
        self._on_close_requested = True
        self.status.set("Closing...")
        self._close_live_camera()
        self.after(200, self._really_close)

    def _close_live_camera(self) -> None:
        try:
            panel = getattr(self.verify_tab, "live_camera", None)
            if panel is not None:
                panel.stop()
        except Exception:
            pass

    def _really_close(self) -> None:
        try:
            if self._system is not None:
                self._system.store.close()
        except Exception:
            pass
        self._on_close_requested = False
        self.destroy()

    def mainloop(self, *args: Any, **kwargs: Any) -> None:
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        super().mainloop(*args, **kwargs)


def _score(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def run(home: Optional[str] = None) -> None:
    """Start the desktop application (blocking)."""
    app = App(home=home)
    app.mainloop()
