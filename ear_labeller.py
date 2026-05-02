"""
ear_labeller.py
===============
Interactive GUI for labelling ear images as Left or Right orientation.

USAGE
-----
    python ear_labeller.py \
        --input_dir  "C:/Users/sansk/Downloads/UERC26_data(1)/data/public" \
        --output_dir labelled_ears \
        --n_samples  400 \
        --seed       42

CONTROLS
--------
    L  or  ←  →  Left Arrow   : Label as LEFT ear
    R  or  →  Right Arrow      : Label as RIGHT ear
    S                          : Skip this image (uncertain)
    U  or  Backspace           : Undo last label
    Q                          : Quit and save progress

OUTPUT
------
    labelled_ears/
        left/    ← images labelled as left
        right/   ← images labelled as right
        skipped/ ← images you were unsure about
    labelled_ears/session_log.csv  ← full audit log
"""

import argparse
import csv
import os
import random
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox

from PIL import Image, ImageTk, ImageOps

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Ear orientation labelling GUI")
    p.add_argument("--input_dir",  required=True,
                   help="Root folder of your ear dataset")
    p.add_argument("--output_dir", default="labelled_ears",
                   help="Where to copy labelled images")
    p.add_argument("--n_samples",  type=int, default=400,
                   help="How many images to sample for labelling")
    p.add_argument("--extensions", nargs="+",
                   default=[".png", ".jpg", ".jpeg"])
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--img_display_size", type=int, default=480,
                   help="Display size (square) for images in the GUI")
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

def copy_image(src, output_dir, label):
    dest_dir = os.path.join(output_dir, label)
    dest     = os.path.join(dest_dir, Path(src).name)
    # Avoid overwrite collisions
    if os.path.exists(dest):
        stem = Path(src).stem
        ext  = Path(src).suffix
        dest = os.path.join(dest_dir, f"{stem}_{random.randint(1000,9999)}{ext}")
    shutil.copy2(src, dest)
    return dest

# ── GUI ───────────────────────────────────────────────────────────────────────

class EarLabeller:
    # Colour palette
    BG          = "#1e1e2e"
    PANEL_BG    = "#2a2a3e"
    TEXT        = "#cdd6f4"
    MUTED       = "#6c7086"
    LEFT_CLR    = "#f38ba8"   # red-ish
    RIGHT_CLR   = "#89dceb"   # cyan-ish
    SKIP_CLR    = "#f9e2af"   # yellow
    UNDO_CLR    = "#b4befe"   # lavender
    DONE_CLR    = "#a6e3a1"   # green

    def __init__(self, root, image_paths, output_dir, display_size):
        self.root        = root
        self.images      = image_paths
        self.output_dir  = output_dir
        self.disp        = display_size
        self.idx         = 0
        self.history     = []   # list of (src_path, label, dest_path)
        self.log         = []   # full CSV log

        root.title("Ear Orientation Labeller")
        root.configure(bg=self.BG)
        root.resizable(False, False)

        self._build_ui()
        self._bind_keys()
        self._load_image()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        W = self.disp + 40          # total window width
        PANEL = 200                 # right panel width

        # ── Image canvas ──
        self.canvas = tk.Canvas(
            self.root, width=self.disp, height=self.disp,
            bg=self.PANEL_BG, highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, padx=(20, 10), pady=(20, 10))

        # ── Right info panel ──
        panel = tk.Frame(self.root, bg=self.PANEL_BG, width=PANEL,
                         padx=14, pady=14)
        panel.grid(row=0, column=1, sticky="ns", padx=(0, 20), pady=(20, 10))
        panel.grid_propagate(False)

        title_font  = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        label_font  = tkfont.Font(family="Segoe UI", size=11)
        count_font  = tkfont.Font(family="Courier New", size=22, weight="bold")
        small_font  = tkfont.Font(family="Segoe UI", size=9)

        tk.Label(panel, text="EAR LABELLER", font=title_font,
                 bg=self.PANEL_BG, fg=self.TEXT).pack(pady=(0, 16))

        # Progress counter
        self.count_var = tk.StringVar(value="0 / 0")
        tk.Label(panel, textvariable=self.count_var, font=count_font,
                 bg=self.PANEL_BG, fg=self.TEXT).pack()

        self.prog_var = tk.DoubleVar(value=0)
        self.prog_bar = tk.Canvas(panel, width=160, height=8,
                                  bg="#45475a", highlightthickness=0)
        self.prog_bar.pack(pady=(4, 16))
        self.prog_fill = self.prog_bar.create_rectangle(
            0, 0, 0, 8, fill=self.DONE_CLR, outline=""
        )

        # Tally
        tk.Label(panel, text="TALLY", font=small_font,
                 bg=self.PANEL_BG, fg=self.MUTED).pack(anchor="w")
        sep = tk.Frame(panel, bg=self.MUTED, height=1)
        sep.pack(fill="x", pady=(2, 8))

        self.left_var  = tk.IntVar(value=0)
        self.right_var = tk.IntVar(value=0)
        self.skip_var  = tk.IntVar(value=0)

        for label, var, clr in [
            ("◀  Left",  self.left_var,  self.LEFT_CLR),
            ("▶  Right", self.right_var, self.RIGHT_CLR),
            ("—  Skip",  self.skip_var,  self.SKIP_CLR),
        ]:
            row = tk.Frame(panel, bg=self.PANEL_BG)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, font=label_font,
                     bg=self.PANEL_BG, fg=clr, width=10, anchor="w").pack(side="left")
            tk.Label(row, textvariable=var, font=label_font,
                     bg=self.PANEL_BG, fg=self.TEXT).pack(side="right")

        # Filename
        tk.Label(panel, text="FILE", font=small_font,
                 bg=self.PANEL_BG, fg=self.MUTED).pack(anchor="w", pady=(20, 0))
        sep2 = tk.Frame(panel, bg=self.MUTED, height=1)
        sep2.pack(fill="x", pady=(2, 6))
        self.fname_var = tk.StringVar()
        tk.Label(panel, textvariable=self.fname_var, font=small_font,
                 bg=self.PANEL_BG, fg=self.TEXT,
                 wraplength=PANEL - 28, justify="left").pack(anchor="w")

        # Controls legend
        tk.Label(panel, text="CONTROLS", font=small_font,
                 bg=self.PANEL_BG, fg=self.MUTED).pack(anchor="w", pady=(20, 0))
        sep3 = tk.Frame(panel, bg=self.MUTED, height=1)
        sep3.pack(fill="x", pady=(2, 6))

        controls = [
            ("L / ←",  "Left ear",   self.LEFT_CLR),
            ("R / →",  "Right ear",  self.RIGHT_CLR),
            ("S",       "Skip",       self.SKIP_CLR),
            ("U / ⌫",  "Undo",       self.UNDO_CLR),
            ("Q",       "Quit & save",self.MUTED),
        ]
        for key, desc, clr in controls:
            row = tk.Frame(panel, bg=self.PANEL_BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=key, font=small_font, width=8,
                     bg=self.PANEL_BG, fg=clr, anchor="w").pack(side="left")
            tk.Label(row, text=desc, font=small_font,
                     bg=self.PANEL_BG, fg=self.TEXT).pack(side="left")

        # ── Bottom button row ──
        btn_frame = tk.Frame(self.root, bg=self.BG)
        btn_frame.grid(row=1, column=0, columnspan=2,
                       padx=20, pady=(0, 20), sticky="ew")
        btn_frame.columnconfigure((0, 1, 2, 3), weight=1)

        btn_cfg = dict(font=tkfont.Font(family="Segoe UI", size=12, weight="bold"),
                       relief="flat", cursor="hand2",
                       padx=10, pady=8, bd=0)

        tk.Button(btn_frame, text="◀  LEFT (L)",
                  bg=self.LEFT_CLR, fg=self.BG,
                  command=lambda: self._label("left"),
                  **btn_cfg).grid(row=0, column=0, padx=4, sticky="ew")

        tk.Button(btn_frame, text="RIGHT (R)  ▶",
                  bg=self.RIGHT_CLR, fg=self.BG,
                  command=lambda: self._label("right"),
                  **btn_cfg).grid(row=0, column=1, padx=4, sticky="ew")

        tk.Button(btn_frame, text="SKIP (S)",
                  bg=self.SKIP_CLR, fg=self.BG,
                  command=lambda: self._label("skipped"),
                  **btn_cfg).grid(row=0, column=2, padx=4, sticky="ew")

        tk.Button(btn_frame, text="UNDO (U)",
                  bg=self.UNDO_CLR, fg=self.BG,
                  command=self._undo,
                  **btn_cfg).grid(row=0, column=3, padx=4, sticky="ew")

        # Status bar
        self.status_var = tk.StringVar(value="Ready — press L, R, or S to begin")
        tk.Label(self.root, textvariable=self.status_var,
                 bg=self.BG, fg=self.MUTED,
                 font=tkfont.Font(family="Segoe UI", size=9),
                 anchor="w").grid(row=2, column=0, columnspan=2,
                                  padx=20, pady=(0, 8), sticky="ew")

    def _bind_keys(self):
        self.root.bind("<KeyPress-l>",       lambda e: self._label("left"))
        self.root.bind("<KeyPress-L>",       lambda e: self._label("left"))
        self.root.bind("<Left>",             lambda e: self._label("left"))
        self.root.bind("<KeyPress-r>",       lambda e: self._label("right"))
        self.root.bind("<KeyPress-R>",       lambda e: self._label("right"))
        self.root.bind("<Right>",            lambda e: self._label("right"))
        self.root.bind("<KeyPress-s>",       lambda e: self._label("skipped"))
        self.root.bind("<KeyPress-S>",       lambda e: self._label("skipped"))
        self.root.bind("<KeyPress-u>",       lambda e: self._undo())
        self.root.bind("<KeyPress-U>",       lambda e: self._undo())
        self.root.bind("<BackSpace>",        lambda e: self._undo())
        self.root.bind("<KeyPress-q>",       lambda e: self._quit())
        self.root.bind("<KeyPress-Q>",       lambda e: self._quit())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_image(self):
        if self.idx >= len(self.images):
            self._finish()
            return

        path = self.images[self.idx]
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((self.disp, self.disp), Image.LANCZOS)

            # Centre on canvas with padding
            bg = Image.new("RGB", (self.disp, self.disp), tuple(
                int(self.PANEL_BG.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)
            ))
            x = (self.disp - img.width)  // 2
            y = (self.disp - img.height) // 2
            bg.paste(img, (x, y))

            self._tk_img = ImageTk.PhotoImage(bg)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

            # Show a faint index overlay
            self.canvas.create_text(
                8, 8, anchor="nw",
                text=f"#{self.idx + 1}",
                fill="#888888",
                font=tkfont.Font(family="Courier New", size=10)
            )
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.disp // 2, self.disp // 2,
                text=f"Cannot load image\n{Path(path).name}\n\n{e}",
                fill=self.LEFT_CLR, font=("Segoe UI", 11), justify="center"
            )

        # Update sidebar info
        total = len(self.images)
        done  = self.idx
        self.count_var.set(f"{done} / {total}")
        self.fname_var.set(Path(path).name)

        pct = done / total if total else 0
        self.prog_bar.coords(self.prog_fill, 0, 0, int(160 * pct), 8)
        self.status_var.set(f"Image {self.idx + 1} of {total}  —  "
                            f"{Path(path).parent.name}/{Path(path).name}")

    # ── Labelling actions ─────────────────────────────────────────────────────

    def _label(self, label):
        if self.idx >= len(self.images):
            return

        src  = self.images[self.idx]
        dest = copy_image(src, self.output_dir, label)
        self.history.append((src, label, dest))
        self.log.append({"src": src, "label": label, "dest": dest})

        # Update tally
        if label == "left":
            self.left_var.set(self.left_var.get() + 1)
            self._flash_canvas(self.LEFT_CLR)
        elif label == "right":
            self.right_var.set(self.right_var.get() + 1)
            self._flash_canvas(self.RIGHT_CLR)
        else:
            self.skip_var.set(self.skip_var.get() + 1)
            self._flash_canvas(self.SKIP_CLR)

        self.idx += 1
        self._load_image()

    def _undo(self):
        if not self.history:
            self.status_var.set("Nothing to undo.")
            return

        src, label, dest = self.history.pop()
        self.log.append({"src": src, "label": f"UNDO_{label}", "dest": dest})

        # Remove copied file
        try:
            if os.path.exists(dest):
                os.remove(dest)
        except Exception as e:
            self.status_var.set(f"Undo warning: {e}")

        # Roll back tally
        if label == "left":
            self.left_var.set(max(0, self.left_var.get() - 1))
        elif label == "right":
            self.right_var.set(max(0, self.right_var.get() - 1))
        else:
            self.skip_var.set(max(0, self.skip_var.get() - 1))

        self.idx -= 1
        self._load_image()
        self.status_var.set(f"Undid label '{label}' for {Path(src).name}")

    def _flash_canvas(self, colour):
        """Brief border flash to give visual feedback on key press."""
        self.canvas.configure(highlightthickness=4, highlightbackground=colour)
        self.root.after(120, lambda: self.canvas.configure(highlightthickness=0))

    # ── Session end ───────────────────────────────────────────────────────────

    def _save_log(self):
        log_path = os.path.join(self.output_dir, "session_log.csv")
        # Append if exists
        mode = "a" if os.path.exists(log_path) else "w"
        with open(log_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["src", "label", "dest"])
            if mode == "w":
                writer.writeheader()
            writer.writerows(self.log)
        return log_path

    def _finish(self):
        log_path = self._save_log()
        left  = self.left_var.get()
        right = self.right_var.get()
        skip  = self.skip_var.get()
        total = left + right + skip
        msg = (
            f"All {len(self.images)} images labelled!\n\n"
            f"  Left   : {left}\n"
            f"  Right  : {right}\n"
            f"  Skipped: {skip}\n\n"
            f"Log saved to:\n{log_path}\n\n"
            f"You can now run train_ear_orientation.py!"
        )
        messagebox.showinfo("Labelling Complete", msg)
        self.root.destroy()

    def _quit(self):
        log_path = self._save_log()
        left  = self.left_var.get()
        right = self.right_var.get()
        skip  = self.skip_var.get()
        remaining = len(self.images) - self.idx
        msg = (
            f"Session saved.\n\n"
            f"  Labelled : {left + right + skip}\n"
            f"  Left     : {left}\n"
            f"  Right    : {right}\n"
            f"  Skipped  : {skip}\n"
            f"  Remaining: {remaining}\n\n"
            f"Re-run the script to continue from where you left off.\n"
            f"Log saved to:\n{log_path}"
        )
        messagebox.showinfo("Session Saved", msg)
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

    # ── Check for existing session log and skip already-labelled images ──
    log_path = os.path.join(args.output_dir, "session_log.csv")
    already_done = set()
    if os.path.exists(log_path):
        with open(log_path, newline="") as f:
            for row in csv.DictReader(f):
                if not row["label"].startswith("UNDO_"):
                    already_done.add(row["src"])
        # Remove undone entries
        undone = set()
        with open(log_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["label"].startswith("UNDO_"):
                    undone.add(row["src"])
        already_done -= undone

    remaining = [p for p in sample if p not in already_done]
    if already_done:
        print(f"Resuming: {len(already_done)} already labelled, "
              f"{len(remaining)} remaining")
    
    if not remaining:
        print("All sampled images have been labelled. Done!")
        return

    root = tk.Tk()
    app  = EarLabeller(root, remaining, args.output_dir, args.img_display_size)
    root.mainloop()

if __name__ == "__main__":
    main()