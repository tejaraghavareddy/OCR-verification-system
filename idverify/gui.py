"""Desktop interface (Tkinter) for IDVerify.

Runs the same :class:`~idverify.engine.IdentitySystem` as the CLI, on the local
machine: camera frames, card images and the SQLite audit log all stay on disk.
Heavy work (OCR + face models) happens on a worker thread so the window never
freezes; results are posted back with ``root.after``.
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
    "REJECTED": ("#8f1d1d", "#fbdede"),
    REVIEW: ("#8a5a00", "#fdf0d5"),
}

PREVIEW_W, PREVIEW_H = 232, 156
IMAGE_TYPES = [
    ("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
    ("All files", "*.*"),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_photo(bgr: np.ndarray, size: tuple[int, int] = (PREVIEW_W, PREVIEW_H)) -> ImageTk.PhotoImage:
    rgb = bgr[:, :, ::-1] if bgr.ndim == 3 else np.stack([bgr] * 3, axis=-1)
    image = Image.fromarray(np.ascontiguousarray(rgb))
    image.thumbnail(size, Image.LANCZOS)
    return ImageTk.PhotoImage(image)


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

        buttons = ttk.Frame(self)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Camera", command=self.from_camera).pack(
            side="left", expand=True, fill="x"
        )
        ttk.Button(buttons, text="File", command=self.from_file).pack(
            side="left", expand=True, fill="x", padx=(6, 0)
        )

    # -- inputs ----------------------------------------------------------
    def from_camera(self) -> None:
        index = self.app.camera_index.get()
        self.app.run_async(
            lambda: Camera(index).read(),
            self._set_capture,
            busy_msg=f"Reading camera {index}...",
        )

    def _set_capture(self, frame, error) -> None:
        if error:
            messagebox.showerror("Camera", str(error), parent=self.app)
            return
        self.set_image(frame, f"Camera frame ({frame.shape[1]}x{frame.shape[0]})")

    def from_file(self) -> None:
        path = filedialog.askopenfilename(title="Choose an image", filetypes=IMAGE_TYPES)
        if not path:
            return
        try:
            image = load_image(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Image", str(exc), parent=self.app)
            return
        self.set_image(path, path.rsplit("/", 1)[-1])

    def set_image(self, image, label: str) -> None:
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
        self.preview.configure(image="")
        self.preview.image = None

    @property
    def ready(self) -> bool:
        return self.image is not None


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
class VerifyTab(ttk.Frame):
    def __init__(self, master, app: "App") -> None:
        super().__init__(master, padding=10)
        self.app = app

        inputs = ttk.Frame(self)
        inputs.pack(side="left", fill="y")
        self.card = CapturePanel(inputs, "1 - ID card", app)
        self.card.pack(fill="x")
        self.face = CapturePanel(inputs, "2 - Live capture", app)
        self.face.pack(fill="x", pady=(10, 0))

        ttk.Button(inputs, text="VERIFY IDENTITY", command=self.start).pack(
            fill="x", pady=(12, 0), ipady=6
        )

        self.result = ttk.Frame(self, padding=(16, 0))
        self.result.pack(side="left", fill="both", expand=True)
        self._build_empty()

    def _build_empty(self) -> None:
        for child in self.result.winfo_children():
            child.destroy()
        ttk.Label(
            self.result,
            text="Load the ID card and a live capture,\nthen press VERIFY IDENTITY.",
            foreground="#555",
            font=("", 11),
        ).pack(anchor="w", pady=(0, 8))
        ttk.Label(
            self.result,
            text=(
                "The system compares the face in the live capture against\n"
                "the portrait printed on the card, then against the record\n"
                "stored locally for that ID number."
            ),
            foreground="#777",
            justify="left",
        ).pack(anchor="w")

    def start(self) -> None:
        if not self.card.ready or not self.face.ready:
            messagebox.showinfo(
                "Verify", "Select both the ID card image and the live capture.",
                parent=self.app,
            )
            return
        card, face = self.card.image, self.face.image
        self.app.run_async(
            lambda: self.app.system().verify(card, face),
            self.show,
            busy_msg="Reading card and matching faces...",
        )

    def show(self, result, error) -> None:
        if error:
            messagebox.showerror("Verify", str(error), parent=self.app)
            return
        for child in self.result.winfo_children():
            child.destroy()

        fg, bg = VERDICT_COLORS.get(result.verdict, ("#333", "#eee"))
        banner = tk.Frame(self.result, bg=bg, padx=14, pady=10)
        banner.pack(fill="x")
        tk.Label(
            banner, text=VERDICT_LABELS.get(result.verdict, result.verdict),
            bg=bg, fg=fg, font=("", 20, "bold"),
        ).pack(anchor="w")
        tk.Label(
            banner, text=result.basis, bg=bg, fg=fg, font=("", 10)
        ).pack(anchor="w")

        fields = ttk.Frame(self.result, padding=(0, 12))
        fields.pack(fill="x")
        rows = [
            ("ID number", result.card_fields.label("id_number")),
            ("Name on card", result.card_fields.label("full_name")),
            ("Date of birth", result.card_fields.label("date_of_birth")),
            ("Date of expiry", result.card_fields.label("date_of_expiry"))
            + ("  (EXPIRED)" if result.card_expired else ""),
            ("Card portrait check", _score(result.live_vs_card)),
            ("Local record check", _score(result.live_vs_enrolled)),
        ]
        if result.person is not None:
            rows.append(("Record name", result.person.display_name))
        for index, (label, value) in enumerate(rows):
            ttk.Label(fields, text=label, foreground="#666").grid(
                row=index, column=0, sticky="w", pady=1
            )
            ttk.Label(fields, text=value, font=("", 10, "bold")).grid(
                row=index, column=1, sticky="w", padx=(18, 0), pady=1
            )
        fields.columnconfigure(1, weight=1)

        notes = tk.Text(self.result, height=7, wrap="word", relief="flat", bg="#f7f7f7")
        notes.pack(fill="both", expand=True)
        for reason in result.reasons:
            notes.insert("end", f"• {reason}\n")
        for warning in result.warnings:
            notes.insert("end", f"! {warning}\n", "warn")
        notes.tag_configure("warn", foreground="#8a5a00")
        notes.configure(state="disabled")

        if result.card_fields.raw_lines:
            detail = "OCR: " + " | ".join(result.card_fields.raw_lines[:8])
            ttk.Label(self.result, text=detail, foreground="#999", wraplength=520).pack(
                anchor="w", pady=(6, 0)
            )


class EnrollTab(ttk.Frame):
    def __init__(self, master, app: "App") -> None:
        super().__init__(master, padding=10)
        self.app = app

        inputs = ttk.Frame(self)
        inputs.pack(side="left", fill="y")
        self.card = CapturePanel(inputs, "1 - ID card", app)
        self.card.pack(fill="x")
        self.face = CapturePanel(inputs, "2 - Live capture of the holder", app)
        self.face.pack(fill="x", pady=(10, 0))
        ttk.Button(inputs, text="ENROLL IDENTITY", command=self.start).pack(
            fill="x", pady=(12, 0), ipady=6
        )

        self.message = tk.Text(self, wrap="word", relief="flat", bg="#f7f7f7")
        self.message.pack(side="left", fill="both", expand=True, padx=(16, 0))
        self._set(
            "Enrolment stores the ID number, the printed portrait and a face "
            "descriptor in the local database.\n\nThe live face must match the "
            "printed portrait before anything is stored."
        )

    def _set(self, text: str) -> None:
        self.message.configure(state="normal")
        self.message.delete("1.0", "end")
        self.message.insert("1.0", text)
        self.message.configure(state="disabled")

    def start(self) -> None:
        if not self.card.ready or not self.face.ready:
            messagebox.showinfo(
                "Enroll", "Select both the ID card image and the live capture.",
                parent=self.app,
            )
            return
        card, face = self.card.image, self.face.image
        self.app.run_async(
            lambda: self.app.system().enroll(card, face),
            self.show,
            busy_msg="Enrolling...",
        )

    def show(self, result, error) -> None:
        if isinstance(error, EnrollError):
            self._set("REFUSED\n\n" + str(error))
            return
        if error:
            messagebox.showerror("Enroll", str(error), parent=self.app)
            return
        verb = "Updated" if result.updated else "Enrolled"
        self._set(
            f"{verb}: {result.id_number}\n"
            f"Name: {result.full_name or '-'}\n"
            f"Live face vs printed portrait: {result.live_vs_card:.3f}\n"
            f"Portrait saved to: {result.portrait_path}\n"
            + ("\nWarnings:\n" + "\n".join(f"• {w}" for w in result.warnings)
               if result.warnings else "")
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
        ttk.Button(bar, text="Forget selected", command=self.forget).pack(
            side="left", padx=(8, 0)
        )

        self.tree = ttk.Treeview(self, columns=self.COLUMNS, show="headings", height=18)
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
            lambda: self.app.system().people(), self._fill, busy_msg="Loading records..."
        )

    def _fill(self, people, error) -> None:
        if error:
            messagebox.showerror("Records", str(error), parent=self.app)
            return
        self.tree.delete(*self.tree.get_children())
        for person in people:
            self.tree.insert(
                "", "end",
                values=(
                    person.id_number, person.display_name,
                    person.date_of_birth or "-", person.date_of_expiry or "-",
                    person.updated_at,
                ),
            )

    def forget(self) -> None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Records", "Select a record first.", parent=self.app)
            return
        id_number = self.tree.item(selection[0], "values")[0]
        if not messagebox.askyesno(
            "Forget", f"Remove {id_number} from the local database?", parent=self.app
        ):
            return
        self.app.run_async(
            lambda: self.app.system().forget(id_number),
            lambda _ok, err: self.refresh() if not err else messagebox.showerror(
                "Records", str(err), parent=self.app
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
        ttk.Label(bar, text=f"Local log: {config.home_dir() / 'idverify.db'}").pack(
            side="left", padx=(10, 0)
        )

        self.tree = ttk.Treeview(self, columns=self.COLUMNS, show="headings", height=18)
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
        self.tree.tag_configure("REJECTED", foreground="#8f1d1d")
        self.tree.tag_configure(VERIFIED, foreground="#0b6b2f")
        self.tree.tag_configure(REVIEW, foreground="#8a5a00")
        self.tree.pack(fill="both", expand=True, pady=(10, 0))

    def refresh(self) -> None:
        self.app.run_async(
            lambda: self.app.system().history(200),
            self._fill,
            busy_msg="Loading history...",
        )

    def _fill(self, entries, error) -> None:
        if error:
            messagebox.showerror("History", str(error), parent=self.app)
            return
        self.tree.delete(*self.tree.get_children())
        for entry in entries:
            self.tree.insert(
                "", "end", tags=(entry.result,),
                values=(
                    entry.created_at, entry.result, entry.id_number or "-",
                    entry.full_name or "-", _score(entry.live_vs_card),
                    _score(entry.live_vs_enrolled),
                ),
            )


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class App(tk.Tk):
    def __init__(self, home: Optional[str] = None) -> None:
        super().__init__()
        self.title(f"IDVerify {__version__} - offline identity verification")
        self.geometry("1080x700")
        self.minsize(900, 600)
        self._home = home
        self._system: Optional[IdentitySystem] = None
        self._busy = False

        header = tk.Frame(self, bg="#14263c", padx=12, pady=8)
        header.pack(fill="x")
        tk.Label(
            header, text="IDVERIFY", bg="#14263c", fg="white", font=("", 16, "bold")
        ).pack(side="left")
        tk.Label(
            header, text="  no network - all data stays on this machine",
            bg="#14263c", fg="#9fc4e8", font=("", 10),
        ).pack(side="left")

        camera_box = ttk.Frame(header)
        camera_box.pack(side="right")
        ttk.Label(camera_box, text="Camera").pack(side="left")
        self.camera_index = tk.Spinbox(camera_box, from_=0, to=5, width=3)
        self.camera_index.pack(side="left", padx=(6, 0))

        self.status = tk.StringVar(value="Ready")
        statusbar = tk.Frame(self, bg="#eeeeee", padx=10, pady=4)
        statusbar.pack(side="bottom", fill="x")
        ttk.Label(statusbar, textvariable=self.status).pack(side="left")
        ttk.Label(statusbar, text=f"Data: {config.home_dir()}").pack(side="right")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=6, pady=6)
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

        self.after(200, self.history_tab.refresh)

    # -- shared system ---------------------------------------------------
    def system(self) -> IdentitySystem:
        if self._system is None:
            self._system = IdentitySystem(home=self._home)
        return self._system

    def _on_tab(self, _event) -> None:
        current = self.notebook.select()
        if current.endswith("records_tab"):
            self.records_tab.refresh()
        elif current.endswith("history_tab"):
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
            self.after(0, lambda: finish(value, failure))

        def finish(value: Any, failure: Optional[BaseException]) -> None:
            self._busy = False
            self.status.set("Ready")
            if isinstance(failure, (ModelMissingError,)):
                messagebox.showerror("Models", str(failure), parent=self)
                return
            done(value, failure)

        threading.Thread(target=runner, daemon=True).start()

    def refresh_records(self) -> None:
        self.records_tab.refresh()


def _score(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def run(home: Optional[str] = None) -> None:
    """Start the desktop application (blocking)."""
    app = App(home=home)
    app.mainloop()