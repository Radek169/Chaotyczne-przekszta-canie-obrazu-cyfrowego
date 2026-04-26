import hashlib
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

# ----------------------------- Narzędzia ogólne -----------------------------

def key_to_seed(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def as_uint8_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.uint8)


def array_to_image(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def make_preview(img: Image.Image, max_size=(330, 330)) -> ImageTk.PhotoImage:
    copy = img.copy()
    copy.thumbnail(max_size)
    return ImageTk.PhotoImage(copy)


def difference_percent(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.shape != b.shape:
        return float("nan")
    return 100.0 * np.mean(a != b)


def neighbor_correlation(arr: np.ndarray, direction: str = "horizontal") -> float:
    gray = (
        0.299 * arr[:, :, 0].astype(np.float64)
        + 0.587 * arr[:, :, 1].astype(np.float64)
        + 0.114 * arr[:, :, 2].astype(np.float64)
    )

    if direction == "horizontal":
        x = gray[:, :-1].ravel()
        y = gray[:, 1:].ravel()
    else:
        x = gray[:-1, :].ravel()
        y = gray[1:, :].ravel()

    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

# ----------------------------- Etap 1 ---------------------------------------
# Naiwny scrambling: cykliczne przesunięcia wierszy i kolumn.
# Odwracalność: przesunięcie o +s odwracamy przez -s.

def stage1_scramble(arr: np.ndarray, key: str) -> np.ndarray:
    seed = key_to_seed(key)
    h, w, _ = arr.shape
    row_shift_base = seed % max(w, 1)
    col_shift_base = (seed >> 16) % max(h, 1)

    out = arr.copy()

    # Każdy wiersz przesuwany deterministycznie.
    for r in range(h):
        shift = (row_shift_base + r) % w
        out[r, :, :] = np.roll(out[r, :, :], shift=shift, axis=0)

    # Każda kolumna przesuwana deterministycznie.
    for c in range(w):
        shift = (col_shift_base + c) % h
        out[:, c, :] = np.roll(out[:, c, :], shift=shift, axis=0)

    return out


def stage1_unscramble(arr: np.ndarray, key: str) -> np.ndarray:
    seed = key_to_seed(key)
    h, w, _ = arr.shape
    row_shift_base = seed % max(w, 1)
    col_shift_base = (seed >> 16) % max(h, 1)

    out = arr.copy()

    # Odwracamy w odwrotnej kolejności: najpierw kolumny, potem wiersze.
    for c in range(w):
        shift = (col_shift_base + c) % h
        out[:, c, :] = np.roll(out[:, c, :], shift=-shift, axis=0)

    for r in range(h):
        shift = (row_shift_base + r) % w
        out[r, :, :] = np.roll(out[r, :, :], shift=-shift, axis=0)

    return out

# ----------------------------- Etap 2 ---------------------------------------
# Czysta permutacja pikseli. Wartości pikseli NIE są zmieniane.
# P: {0...N-1} -> {0...N-1}
# Jeżeli scrambled[P[i]] = original[i], to inverse[P[i]] = i.

def permutation_for_shape(shape, key: str) -> np.ndarray:
    h, w, _ = shape
    n = h * w
    rng = np.random.default_rng(key_to_seed(key))
    return rng.permutation(n)


def inverse_permutation(p: np.ndarray) -> np.ndarray:
    inv = np.empty_like(p)
    inv[p] = np.arange(len(p))
    return inv


def stage2_scramble(arr: np.ndarray, key: str) -> np.ndarray:
    h, w, c = arr.shape
    p = permutation_for_shape(arr.shape, key)
    flat = arr.reshape(-1, c)
    out = np.empty_like(flat)
    out[p] = flat
    return out.reshape(h, w, c)


def stage2_unscramble(arr: np.ndarray, key: str) -> np.ndarray:
    h, w, c = arr.shape
    p = permutation_for_shape(arr.shape, key)
    inv = inverse_permutation(p)
    flat = arr.reshape(-1, c)
    out = flat[p]  # równoważne: original = scrambled[inverse_indices]
    # Alternatywa jawna:
    # out = np.empty_like(flat); out[inv] = flat
    return out.reshape(h, w, c)

# ----------------------------- Etap 3 ---------------------------------------
# Hybryda: najpierw permutacja z Etapu 2, potem substytucja XOR maską PRNG.
# f(p,k) = p XOR m(k)
# f^-1(p,k) = p XOR m(k), bo XOR jest samoodwrotny.
# To nadal NIE jest bezpieczny szyfr, ale spełnia wymóg mechanizmu wzmacniającego.

def xor_mask_for_shape(shape, key: str) -> np.ndarray:
    seed = key_to_seed("substitution:" + key)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def stage3_scramble(arr: np.ndarray, key: str) -> np.ndarray:
    permuted = stage2_scramble(arr, "perm:" + key)
    mask = xor_mask_for_shape(permuted.shape, key)
    return np.bitwise_xor(permuted, mask)


def stage3_unscramble(arr: np.ndarray, key: str) -> np.ndarray:
    mask = xor_mask_for_shape(arr.shape, key)
    unmasked = np.bitwise_xor(arr, mask)
    return stage2_unscramble(unmasked, "perm:" + key)

# ----------------------------- Dispatcher -----------------------------------

def scramble(arr: np.ndarray, stage: int, key: str) -> np.ndarray:
    if stage == 1:
        return stage1_scramble(arr, key)
    if stage == 2:
        return stage2_scramble(arr, key)
    if stage == 3:
        return stage3_scramble(arr, key)
    raise ValueError("Nieznany etap")

def unscramble(arr: np.ndarray, stage: int, key: str) -> np.ndarray:
    if stage == 1:
        return stage1_unscramble(arr, key)
    if stage == 2:
        return stage2_unscramble(arr, key)
    if stage == 3:
        return stage3_unscramble(arr, key)
    raise ValueError("Nieznany etap")

# ----------------------------- GUI ------------------------------------------

class ScramblingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Projekt M-II — Chaotyczne przekształcanie obrazu cyfrowego")
        self.geometry("1180x760")

        self.original_arr = None
        self.scrambled_arr = None
        self.recovered_arr = None

        self.original_img = None
        self.scrambled_img = None
        self.recovered_img = None

        self.original_preview = None
        self.scrambled_preview = None
        self.recovered_preview = None

        self._build_ui()

    def _build_ui(self):
        controls = ttk.Frame(self, padding=10)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(controls, text="Wczytaj obraz", command=self.load_image).grid(row=0, column=0, padx=5)

        ttk.Label(controls, text="Etap:").grid(row=0, column=1, padx=5)
        self.stage_var = tk.IntVar(value=1)
        ttk.Combobox(
            controls,
            textvariable=self.stage_var,
            values=[1, 2, 3],
            width=5,
            state="readonly",
        ).grid(row=0, column=2, padx=5)

        ttk.Label(controls, text="Poprawny klucz:").grid(row=0, column=3, padx=5)
        self.key_var = tk.StringVar(value="tajny_klucz_123")
        ttk.Entry(controls, textvariable=self.key_var, width=24).grid(row=0, column=4, padx=5)

        ttk.Label(controls, text="Błędny klucz:").grid(row=0, column=5, padx=5)
        self.bad_key_var = tk.StringVar(value="tajny_klucz_124")
        ttk.Entry(controls, textvariable=self.bad_key_var, width=24).grid(row=0, column=6, padx=5)

        self.use_bad_key = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="Unscramble błędnym kluczem",
            variable=self.use_bad_key
        ).grid(row=0, column=7, padx=5)

        ttk.Button(controls, text="Scramble", command=self.do_scramble).grid(row=1, column=0, padx=5, pady=8)
        ttk.Button(controls, text="Unscramble", command=self.do_unscramble).grid(row=1, column=1, padx=5, pady=8)
        ttk.Button(controls, text="Pełny test", command=self.full_test).grid(row=1, column=2, padx=5, pady=8)
        ttk.Button(controls, text="Zapisz wyniki", command=self.save_results).grid(row=1, column=3, padx=5, pady=8)

        image_frame = ttk.Frame(self, padding=10)
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.original_label = self._image_panel(image_frame, "Obraz oryginalny", 0)
        self.scrambled_label = self._image_panel(image_frame, "Obraz przekształcony", 1)
        self.recovered_label = self._image_panel(image_frame, "Obraz odtworzony", 2)

        metrics_frame = ttk.LabelFrame(self, text="Metryki i wnioski eksperymentalne", padding=10)
        metrics_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=10, pady=10)

        self.metrics_text = tk.Text(metrics_frame, height=11, wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        self._write_metrics("Wczytaj obraz, wybierz etap i klucz, następnie kliknij Scramble oraz Unscramble.")

    def _image_panel(self, parent, title, col):
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.grid(row=0, column=col, sticky="nsew", padx=8)
        parent.columnconfigure(col, weight=1)
        label = ttk.Label(frame)
        label.pack(expand=True)
        return label

    def _write_metrics(self, text: str):
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, text)

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[
                ("Obrazy", "*.png *.jpg *.jpeg *.bmp"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"),
                ("Wszystkie pliki", "*.*"),
            ],
        )
        if not path:
            return

        img = Image.open(path).convert("RGB")
        self.original_img = img
        self.original_arr = as_uint8_rgb(img)
        self.scrambled_arr = None
        self.recovered_arr = None
        self.scrambled_img = None
        self.recovered_img = None

        self._refresh_images()
        self._update_metrics()

    def _refresh_images(self):
        if self.original_arr is not None:
            self.original_img = array_to_image(self.original_arr)
            self.original_preview = make_preview(self.original_img)
            self.original_label.configure(image=self.original_preview)

        if self.scrambled_arr is not None:
            self.scrambled_img = array_to_image(self.scrambled_arr)
            self.scrambled_preview = make_preview(self.scrambled_img)
            self.scrambled_label.configure(image=self.scrambled_preview)
        else:
            self.scrambled_label.configure(image="")

        if self.recovered_arr is not None:
            self.recovered_img = array_to_image(self.recovered_arr)
            self.recovered_preview = make_preview(self.recovered_img)
            self.recovered_label.configure(image=self.recovered_preview)
        else:
            self.recovered_label.configure(image="")

    def do_scramble(self):
        if self.original_arr is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return

        stage = self.stage_var.get()
        key = self.key_var.get()
        self.scrambled_arr = scramble(self.original_arr, stage, key)
        self.recovered_arr = None
        self._refresh_images()
        self._update_metrics()

    def do_unscramble(self):
        if self.scrambled_arr is None:
            messagebox.showwarning("Brak wyniku", "Najpierw wykonaj Scramble.")
            return

        stage = self.stage_var.get()
        key = self.bad_key_var.get() if self.use_bad_key.get() else self.key_var.get()
        self.recovered_arr = unscramble(self.scrambled_arr, stage, key)
        self._refresh_images()
        self._update_metrics()

    def full_test(self):
        if self.original_arr is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return

        stage = self.stage_var.get()
        key = self.key_var.get()
        bad_key = self.bad_key_var.get()

        self.scrambled_arr = scramble(self.original_arr, stage, key)
        good_recovered = unscramble(self.scrambled_arr, stage, key)
        bad_recovered = unscramble(self.scrambled_arr, stage, bad_key)

        self.recovered_arr = good_recovered
        self._refresh_images()

        good_diff = difference_percent(self.original_arr, good_recovered)
        bad_diff = difference_percent(self.original_arr, bad_recovered)

        base = self._metrics_string()
        base += "\n\nTEST POPRAWNEGO I BŁĘDNEGO KLUCZA"
        base += f"\nRóżnica po unscramblingu poprawnym kluczem: {good_diff:.6f}%"
        base += f"\nRóżnica po unscramblingu błędnym kluczem: {bad_diff:.6f}%"
        base += "\n\nInterpretacja:"
        if good_diff == 0:
            base += "\n- Algorytm jest odwracalny dla poprawnego klucza."
        else:
            base += "\n- UWAGA: wynik nie został idealnie odtworzony."
        base += "\n- Błędny klucz pokazuje wrażliwość na parametr/seed."
        base += "\n- Etap 1 zwykle pozostawia struktury obrazu, bo przesunięcia zachowują lokalne relacje w wierszach/kolumnach."
        base += "\n- Etap 2 usuwa układ przestrzenny, ale zachowuje histogram i wartości pikseli."
        base += "\n- Etap 3 zmienia także wartości pikseli przez XOR, ale nadal nie jest bezpiecznym szyfrem."
        self._write_metrics(base)

    def _metrics_string(self) -> str:
        lines = []
        stage = self.stage_var.get()
        lines.append(f"Etap: {stage}")
        lines.append(f"Rozmiar obrazu: {self.original_arr.shape[1]} x {self.original_arr.shape[0]} px")

        orig_h = neighbor_correlation(self.original_arr, "horizontal")
        orig_v = neighbor_correlation(self.original_arr, "vertical")
        lines.append(f"Korelacja sąsiednich pikseli — oryginał poziomo: {orig_h:.6f}")
        lines.append(f"Korelacja sąsiednich pikseli — oryginał pionowo: {orig_v:.6f}")

        if self.scrambled_arr is not None:
            scr_h = neighbor_correlation(self.scrambled_arr, "horizontal")
            scr_v = neighbor_correlation(self.scrambled_arr, "vertical")
            lines.append(f"Korelacja sąsiednich pikseli — po scramblingu poziomo: {scr_h:.6f}")
            lines.append(f"Korelacja sąsiednich pikseli — po scramblingu pionowo: {scr_v:.6f}")

        if self.recovered_arr is not None:
            diff = difference_percent(self.original_arr, self.recovered_arr)
            used_key = "błędny" if self.use_bad_key.get() else "poprawny"
            lines.append(f"Różnica oryginał vs odtworzony ({used_key} klucz): {diff:.6f}%")

        return "\n".join(lines)

    def _update_metrics(self):
        if self.original_arr is None:
            return
        self._write_metrics(self._metrics_string())

    def save_results(self):
        if self.original_arr is None:
            messagebox.showwarning("Brak obrazu", "Nie ma czego zapisać.")
            return

        directory = filedialog.askdirectory(title="Wybierz folder zapisu")
        if not directory:
            return

        if self.original_arr is not None:
            array_to_image(self.original_arr).save(os.path.join(directory, "oryginal.png"))
        if self.scrambled_arr is not None:
            array_to_image(self.scrambled_arr).save(os.path.join(directory, "scrambled.png"))
        if self.recovered_arr is not None:
            array_to_image(self.recovered_arr).save(os.path.join(directory, "recovered.png"))

        with open(os.path.join(directory, "metryki.txt"), "w", encoding="utf-8") as f:
            f.write(self.metrics_text.get("1.0", tk.END))

        messagebox.showinfo("Zapisano", f"Wyniki zapisano w folderze:\n{directory}")

if __name__ == "__main__":
    app = ScramblingApp()
    app.mainloop()