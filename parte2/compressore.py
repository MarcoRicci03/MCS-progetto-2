import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dct_utils import compress_image, create_mask

class ImageCompressorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Compressore DCT - Progetto 2")
        self.root.geometry("900x600")
        self.root.resizable(False, False)

        self.img_array = None
        self.img_path = None

        self.create_interface()

    def create_interface(self):
        frame_file = ttk.LabelFrame(self.root, text="1. Carica immagine BMP")
        frame_file.pack(fill="x", padx=10, pady=10)

        self.label_file = ttk.Label(self.root, text="1. Nessun file selezionato")
        self.label_file.pack(side="left", padx=10, pady=10)

        btn_sfoglia = ttk.Button(frame_file, text="Sfoglia...", command=self.load_image)
        btn_sfoglia.pack(side="right", padx=10, pady=10)

        frame_param = ttk.LabelFrame(self.root, text="2. Parametri")
        frame_param.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame_param, text="F (dimensione blocco):").grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.var_F = tk.IntVar(value=8)
        self.spin_F = ttk.Spinbox(frame_param, from_=2, to=64, textvariable=self.var_F, width=10)
        self.spin_F.grid(row=0, column=1, padx=10, pady=8)

        ttk.Label(frame_param, text="d (soglia frequenze):").grid(row=1, column=0, padx=10, pady=8, sticky="w")
        self.var_d = tk.IntVar(value=5)
        self.spin_d = ttk.Spinbox(frame_param, from_=0, to=14, textvariable=self.var_d, width=10)
        self.spin_d.grid(row=1, column=1, padx=10, pady=8)

        self.label_range = ttk.Label(frame_param, text="Con F=8, d deve stare tra 0 e 14")
        self.label_range.grid(row=1, column=2, padx=10, pady=8, sticky="w")

        self.var_F.trace_add("write", self.update_range_d)

        btn_execute = ttk.Button(self.root, text="Comprimi e visualizza", command=self.execute)
        btn_execute.pack(pady=20)

    def update_range_d(self, *_):
        try:
            F = self.var_F.get()
            max_d = 2 * F - 2
            self.spin_d.config(to=max_d)
            self.label_range.config(text=f"Con F={F}, d deve stare tra 0 e {max_d}")
        except:
            pass

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Seleziona immagine BMP",
            filetypes=[("Bitmap", "*.bmp")]
        )

        if not path:
            return

        try:
            img = Image.open(path).convert("L")
            self.img_array = np.array(img, dtype=np.uint8)
            self.img_path = path
            self.label_file.config(text=path.split("/")[-1])
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile aprire il file:\n{e}")

    def execute(self):
        if self.img_array is None:
            messagebox.showwarning("Attenzione", "Carica prima un'immagine BMP.")
            return

        F = self.var_F.get()
        d = self.var_d.get()

        if d < 0 or d > 2 * F - 2:
            messagebox.showerror("Errore", f"d deve essere compreso tra 0 e {2*F - 2}.")
            return

        try:
            img_orig, img_comp = compress_image(self.img_array, F, d)
            self.show_figures(img_orig, img_comp, F, d)
        except Exception as e:
            messagebox.showerror("Errore", str(e))

    def show_figures(self, originale, compressa, F, d):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(originale, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Immagine originale")
        axes[0].axis("off")

        axes[1].imshow(compressa, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(f"Immagine compressa (F={F}, d={d})")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCompressorApp(root)
    root.mainloop()