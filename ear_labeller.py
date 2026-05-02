"""
ear_labeller.py
===============
Interactive GUI for labelling ear images as Left or Right orientation.
Works with WebP, PNG, JPG images (default: .webp from precompute step).

USAGE
-----
    python ear_labeller.py \
        --input_dir  "C:/Users/sansk/Downloads/UERC26_Modified/data/public" \
        --output_dir labelled_ears \
        --n_samples  400

CONTROLS
--------
    L  or  Left Arrow   : Label as LEFT ear
    R  or  Right Arrow  : Label as RIGHT ear
    S                   : Skip this image (uncertain)
    U  or  Backspace    : Undo last label
    Q                   : Quit and save progress

OUTPUT
------
    labelled_ears/
        left/      ← images labelled as left  (saved as WebP lossless)
        right/     ← images labelled as right (saved as WebP lossless)
        skipped/   ← images you were unsure about
    labelled_ears/session_log.csv  ← full audit log (used for resume)

NOTE: All output images are saved as WebP lossless regardless of source format,
      keeping the pipeline consistent with the precompute step.
"""

import argparse
import csv
import os
import random
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox

from PIL import Image, ImageTk

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Ear orientation labelling GUI")
    p.add_argument("--input_dir",  required=True,
                   help="Root folder of your resized WebP ear dataset "
                        "(output of standalone_precompute.py)")
    p.add_argument("--output_dir", default="labelled_ears",
                   help="Where to save labelled images (as WebP lossless)")
    p.add_argument("--n_samples",  type=int, default=400,
                   help="How many images to randomly sample for labelling")
    p.add_argument("--extensions", nargs="+",
                   default=[".webp", ".png", ".jpg", ".jpeg"],
                   help="Image extensions to scan for (default includes .webp first)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--img_display_size", type=int, default=480,
                   help="Square canvas size in pixels for displaying images")
    return p.parse_args()

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_images(root, extensions):
    exts = {e.lower() for e in extensions}
    return sorted([
        str(p) for p in Path(root).rglob("*")
        if p.suffix.lower() in exts
    ])

def ensure_dirs(base):
    for sub in ("left", "right", "skipped"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

def save_as_webp(src_path, output_dir, label):
    """
    Open any image format and save it as WebP lossless into output_dir/label/.
    Handles filename collisions (same stem from different subject folders).
    Returns the destination path.
    """
    dest_dir  = os.path.join(output_dir, label)
    stem      = Path(src_path).stem
    dest_path = os.path.join(dest_dir, f"{stem}.webp")

    if os.path.exists(dest_path):
        dest_path = os.path.join(dest_dir, f"{stem}_{random.randint(1000, 9999)}.webp")

    img = Image.open(src_path).convert("RGB")
    img.save(dest_path, format="WEBP", lossless=True)
    return dest_path

# ── GUI ───────────────────────────────────────────────────────────────────────

class EarLabeller:
    # All colours are standard 6-digit hex — safe on Windows Tkinter
    BG        = "#1e1e2e"
    PANEL_BG  = "#2a2a3e"
    TEXT      = "#cdd6f4"
    MUTED     = "#6c7086"
    LEFT_CLR  = "#f38ba8"   # red-pink  → left
    RIGHT_CLR = "#89dceb"   # cyan      → right
    SKIP_CLR  = "#f9e2af"   # yellow    → skip
    UNDO_CLR  = "#b4befe"   # lavender  → undo
    DONE_CLR  = "#a6e3a1"   # green     → progress bar

    def __init__(self, root, image_paths, output_dir, display_size):
        self.root       = root
        self.images     = image_paths
        self.output_dir = output_dir
        self.disp       = display_size
        self.idx        = 0
        self.history    = []   # list of (src_path, label, dest_path)
        self.log        = []   # pending rows for session_log.csv

        root.title("Ear Orientation Labeller  —  UERC26")
        root.configure(bg=self.BG)
        root.resizable(False, False)

        self._build_ui()
        self._bind_keys()
        self._load_image()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        PANEL = 210

        # Image canvas
        self.canvas = tk.Canvas(
            self.root, width=self.disp, height=self.disp,
            bg=self.PANEL_BG, highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, padx=(20, 10), pady=(20, 10))

        # Right sidebar
        panel = tk.Frame(self.root, bg=self.PANEL_BG, width=PANEL,
                         padx=14, pady=14)
        panel.grid(row=0, column=1, sticky="ns", padx=(0, 20), pady=(20, 10))
        panel.grid_propagate(False)

        title_font = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        label_font = tkfont.Font(family="Segoe UI", size=11)
        count_font = tkfont.Font(family="Courier New", size=22, weight="bold")
        small_font = tkfont.Font(family="Segoe UI", size=9)

        tk.Label(panel, text="EAR LABELLER", font=title_font,
                 bg=self.PANEL_BG, fg=self.TEXT).pack(pady=(0, 16))

        # Progress counter
        self.count_var = tk.StringVar(value="0 / 0")
        tk.Label(panel, textvariable=self.count_var, font=count_font,
                 bg=self.PANEL_BG, fg=self.TEXT).pack()

        # Progress bar
        self.prog_bar = tk.Canvas(panel, width=170, height=8,
                                  bg="#45475a", highlightthickness=0)
        self.prog_bar.pack(pady=(4, 16))
        self.prog_fill = self.prog_bar.create_rectangle(
            0, 0, 0, 8, fill=self.DONE_CLR, outline=""
        )

        # Tally
        self._section(panel, small_font, "TALLY")
        self.left_var  = tk.IntVar(value=0)
        self.right_var = tk.IntVar(value=0)
        self.skip_var  = tk.IntVar(value=0)

        for text, var, clr in [
            ("◀  Left",  self.left_var,  self.LEFT_CLR),
            ("▶  Right", self.right_var, self.RIGHT_CLR),
            ("—  Skip",  self.skip_var,  self.SKIP_CLR),
        ]:
            row = tk.Frame(panel, bg=self.PANEL_BG)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=text, font=label_font,
                     bg=self.PANEL_BG, fg=clr,
                     width=10, anchor="w").pack(side="left")
            tk.Label(row, textvariable=var, font=label_font,
                     bg=self.PANEL_BG, fg=self.TEXT).pack(side="right")

        # File info
        self._section(panel, small_font, "FILE", top=20)
        self.fname_var = tk.StringVar()
        tk.Label(panel, textvariable=self.fname_var, font=small_font,
                 bg=self.PANEL_BG, fg=self.TEXT,
                 wraplength=PANEL - 28, justify="left").pack(anchor="w")

        self._section(panel, small_font, "SUBJECT", top=10)
        self.subj_var = tk.StringVar()
        tk.Label(panel, textvariable=self.subj_var, font=small_font,
                 bg=self.PANEL_BG, fg=self.MUTED,
                 wraplength=PANEL - 28, justify="left").pack(anchor="w")

        # Controls legend
        self._section(panel, small_font, "CONTROLS", top=20)
        for key, desc, clr in [
            ("L / ←",  "Left ear",    self.LEFT_CLR),
            ("R / →",  "Right ear",   self.RIGHT_CLR),
            ("S",      "Skip",        self.SKIP_CLR),
            ("U / ⌫",  "Undo",        self.UNDO_CLR),
            ("Q",      "Quit & save", self.MUTED),
        ]:
            row = tk.Frame(panel, bg=self.PANEL_BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=key,  font=small_font, width=8,
                     bg=self.PANEL_BG, fg=clr, anchor="w").pack(side="left")
            tk.Label(row, text=desc, font=small_font,
                     bg=self.PANEL_BG, fg=self.TEXT).pack(side="left")

        # Bottom button row
        btn_frame = tk.Frame(self.root, bg=self.BG)
        btn_frame.grid(row=1, column=0, columnspan=2,
                       padx=20, pady=(0, 20), sticky="ew")
        btn_frame.columnconfigure((0, 1, 2, 3), weight=1)

        btn = dict(font=tkfont.Font(family="Segoe UI", size=12, weight="bold"),
                   relief="flat", cursor="hand2", padx=10, pady=8, bd=0)

        tk.Button(btn_frame, text="◀  LEFT  (L)",
                  bg=self.LEFT_CLR, fg=self.BG,
                  command=lambda: self._label("left"), **btn
                  ).grid(row=0, column=0, padx=4, sticky="ew")

        tk.Button(btn_frame, text="RIGHT  (R)  ▶",
                  bg=self.RIGHT_CLR, fg=self.BG,
                  command=lambda: self._label("right"), **btn
                  ).grid(row=0, column=1, padx=4, sticky="ew")

        tk.Button(btn_frame, text="SKIP  (S)",
                  bg=self.SKIP_CLR, fg=self.BG,
                  command=lambda: self._label("skipped"), **btn
                  ).grid(row=0, column=2, padx=4, sticky="ew")

        tk.Button(btn_frame, text="UNDO  (U)",
                  bg=self.UNDO_CLR, fg=self.BG,
                  command=self._undo, **btn
                  ).grid(row=0, column=3, padx=4, sticky="ew")

        # Status bar
        self.status_var = tk.StringVar(value="Ready — press L, R, or S to begin")
        tk.Label(self.root, textvariable=self.status_var,
                 bg=self.BG, fg=self.MUTED, font=small_font, anchor="w"
                 ).grid(row=2, column=0, columnspan=2,
                        padx=20, pady=(0, 8), sticky="ew")

    def _section(self, parent, font, text, top=0):
        tk.Label(parent, text=text, font=font,
                 bg=self.PANEL_BG, fg=self.MUTED).pack(
            anchor="w", pady=(top, 0))
        tk.Frame(parent, bg=self.MUTED, height=1).pack(fill="x", pady=(2, 6))

    def _bind_keys(self):
        for seq in ("<KeyPress-l>", "<KeyPress-L>", "<Left>"):
            self.root.bind(seq, lambda e: self._label("left"))
        for seq in ("<KeyPress-r>", "<KeyPress-R>", "<Right>"):
            self.root.bind(seq, lambda e: self._label("right"))
        for seq in ("<KeyPress-s>", "<KeyPress-S>"):
            self.root.bind(seq, lambda e: self._label("skipped"))
        for seq in ("<KeyPress-u>", "<KeyPress-U>", "<BackSpace>"):
            self.root.bind(seq, lambda e: self._undo())
        for seq in ("<KeyPress-q>", "<KeyPress-Q>"):
            self.root.bind(seq, lambda e: self._quit())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_image(self):
        if self.idx >= len(self.images):
            self._finish()
            return

        path = self.images[self.idx]
        self.canvas.delete("all")

        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((self.disp, self.disp), Image.LANCZOS)

            bg_colour = tuple(
                int(self.PANEL_BG.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)
            )
            bg = Image.new("RGB", (self.disp, self.disp), bg_colour)
            bg.paste(img, ((self.disp - img.width) // 2,
                           (self.disp - img.height) // 2))

            self._tk_img = ImageTk.PhotoImage(bg)
            self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

            # Index counter overlay (plain 6-digit hex only — Windows safe)
            self.canvas.create_text(
                8, 8, anchor="nw",
                text=f"#{self.idx + 1}",
                fill="#888888",
                font=tkfont.Font(family="Courier New", size=10)
            )

        except Exception as e:
            self.canvas.create_text(
                self.disp // 2, self.disp // 2,
                text=f"Cannot load image\n{Path(path).name}\n\n{e}",
                fill=self.LEFT_CLR, font=("Segoe UI", 11), justify="center"
            )

        total = len(self.images)
        self.count_var.set(f"{self.idx} / {total}")
        self.fname_var.set(Path(path).name)
        self.subj_var.set(Path(path).parent.name)
        pct = self.idx / total if total else 0
        self.prog_bar.coords(self.prog_fill, 0, 0, int(170 * pct), 8)
        self.status_var.set(
            f"Image {self.idx + 1} of {total}  —  subject: {Path(path).parent.name}"
        )

    # ── Labelling ─────────────────────────────────────────────────────────────

    def _label(self, label):
        if self.idx >= len(self.images):
            return
        src  = self.images[self.idx]
        dest = save_as_webp(src, self.output_dir, label)
        self.history.append((src, label, dest))
        self.log.append({"src": src, "label": label, "dest": dest})

        if label == "left":
            self.left_var.set(self.left_var.get() + 1)
            self._flash(self.LEFT_CLR)
        elif label == "right":
            self.right_var.set(self.right_var.get() + 1)
            self._flash(self.RIGHT_CLR)
        else:
            self.skip_var.set(self.skip_var.get() + 1)
            self._flash(self.SKIP_CLR)

        self.idx += 1
        self._load_image()

    def _undo(self):
        if not self.history:
            self.status_var.set("Nothing to undo.")
            return
        src, label, dest = self.history.pop()
        self.log.append({"src": src, "label": f"UNDO_{label}", "dest": dest})
        try:
            if os.path.exists(dest):
                os.remove(dest)
        except Exception as e:
            self.status_var.set(f"Undo warning: {e}")

        if label == "left":
            self.left_var.set(max(0, self.left_var.get() - 1))
        elif label == "right":
            self.right_var.set(max(0, self.right_var.get() - 1))
        else:
            self.skip_var.set(max(0, self.skip_var.get() - 1))

        self.idx -= 1
        self._load_image()
        self.status_var.set(f"Undid '{label}' for {Path(src).name}")

    def _flash(self, colour):
        self.canvas.configure(highlightthickness=4, highlightbackground=colour)
        self.root.after(130, lambda: self.canvas.configure(highlightthickness=0))

    # ── Session persistence ───────────────────────────────────────────────────

    def _save_log(self):
        log_path = os.path.join(self.output_dir, "session_log.csv")
        mode = "a" if os.path.exists(log_path) else "w"
        with open(log_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["src", "label", "dest"])
            if mode == "w":
                writer.writeheader()
            writer.writerows(self.log)
        self.log.clear()
        return log_path

    def _finish(self):
        log_path = self._save_log()
        l, r, s = self.left_var.get(), self.right_var.get(), self.skip_var.get()
        messagebox.showinfo("Labelling Complete", (
            f"All {len(self.images)} images labelled!\n\n"
            f"  Left    : {l}\n"
            f"  Right   : {r}\n"
            f"  Skipped : {s}\n\n"
            f"Log saved to:\n{log_path}\n\n"
            f"Next step:\n  python train_ear_orientation.py \\\n"
            f"    --data_dir labelled_ears \\\n"
            f"    --save_path ear_orientation_model.pt"
        ))
        self.root.destroy()

    def _quit(self):
        log_path  = self._save_log()
        l, r, s   = self.left_var.get(), self.right_var.get(), self.skip_var.get()
        remaining = len(self.images) - self.idx
        messagebox.showinfo("Session Saved", (
            f"Progress saved.\n\n"
            f"  Labelled  : {l + r + s}\n"
            f"  Left      : {l}\n"
            f"  Right     : {r}\n"
            f"  Skipped   : {s}\n"
            f"  Remaining : {remaining}\n\n"
            f"Re-run the script to resume from where you left off.\n"
            f"Log saved to:\n{log_path}"
        ))
        self.root.destroy()

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Scanning {args.input_dir} ...")
    all_images = find_images(args.input_dir, args.extensions)
    print(f"Found {len(all_images)} images")

    if not all_images:
        print("No images found. Check --input_dir and --extensions.")
        return

    random.seed(args.seed)
    sample = random.sample(all_images, min(args.n_samples, len(all_images)))
    print(f"Sampled {len(sample)} images for labelling")

    ensure_dirs(args.output_dir)

    # Resume: compute which images are already definitively labelled
    log_path = os.path.join(args.output_dir, "session_log.csv")
    labelled, undone = set(), set()
    if os.path.exists(log_path):
        with open(log_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["label"].startswith("UNDO_"):
                    undone.add(row["src"])
                else:
                    labelled.add(row["src"])
        already_done = labelled - undone
        remaining    = [p for p in sample if p not in already_done]
        print(f"Resuming: {len(already_done)} already labelled, "
              f"{len(remaining)} remaining")
    else:
        remaining = sample

    if not remaining:
        print("All sampled images are already labelled. Done!")
        return

    root = tk.Tk()
    EarLabeller(root, remaining, args.output_dir, args.img_display_size)
    root.mainloop()

if __name__ == "__main__":
    main()