import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from dct_utils import compress_image, create_mask


class ImageCompressorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Compressore DCT - Progetto 2")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        self.img_array = None
        self.photo_image = None

        self.create_interface()

    def create_interface(self):
        frame_file = ttk.LabelFrame(self.root, text="1. Carica immagine BMP")
        frame_file.pack(fill="x", padx=10, pady=10)

        self.label_file = ttk.Label(frame_file, text="Nessun file selezionato")
        self.label_file.pack(side="left", padx=10, pady=10)

        btn_sfoglia = ttk.Button(frame_file, text="Sfoglia...", command=self.load_image)
        btn_sfoglia.pack(side="right", padx=10, pady=10)

        frame_preview = ttk.LabelFrame(self.root, text="Anteprima immagine caricata")
        frame_preview.pack(fill="both", expand=True, padx=10, pady=10)

        self.label_preview = ttk.Label(frame_preview, text="Nessuna immagine caricata")
        self.label_preview.pack(expand=True)

        frame_param = ttk.LabelFrame(self.root, text="2. Parametri")
        frame_param.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame_param, text="F (dimensione blocco):").grid(
            row=0, column=0, padx=10, pady=8, sticky="w"
        )
        self.var_F = tk.IntVar(value=8)
        self.spin_F = ttk.Spinbox(
            frame_param, from_=2, to=64, textvariable=self.var_F, width=10
        )
        self.spin_F.grid(row=0, column=1, padx=10, pady=8)

        ttk.Label(frame_param, text="d (soglia frequenze):").grid(
            row=1, column=0, padx=10, pady=8, sticky="w"
        )
        self.var_d = tk.IntVar(value=5)
        self.spin_d = ttk.Spinbox(
            frame_param, from_=0, to=14, textvariable=self.var_d, width=10
        )
        self.spin_d.grid(row=1, column=1, padx=10, pady=8)

        self.label_range = ttk.Label(
            frame_param, text="Con F=8, d deve stare tra 0 e 14"
        )
        self.label_range.grid(row=1, column=2, padx=10, pady=8, sticky="w")

        self.var_F.trace_add("write", self.update_range_d)

        btn_execute = ttk.Button(
            self.root, text="Comprimi e visualizza", command=self.execute
        )
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
            title="Seleziona immagine BMP", filetypes=[("Bitmap", "*.bmp")]
        )

        if not path:
            return

        try:
            img = Image.open(path)
            self.img_array = np.array(img, dtype=np.uint8)
            self.label_file.config(text=path.split("/")[-1])
            self.update_preview()
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile aprire il file:\n{e}")

    def update_preview(self):
        if self.img_array is None:
            self.label_preview.config(image="", text="Nessuna immagine caricata")
            return

        img_pil = Image.fromarray(self.img_array)
        max_w, max_h = 400, 300
        w, h = img_pil.size
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(img_pil)
        self.label_preview.config(image=self.photo_image, text="")

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
            img_comp = compress_image(self.img_array, F, d)
            self.show_figures(img_comp, F, d)
        except Exception as e:
            messagebox.showerror("Errore", str(e))

    def show_figures(self, compressa, F, d):
        orig = self.img_array
        orig_h, orig_w = orig.shape
        comp_h, comp_w = compressa.shape

        max_display_w = 800
        max_display_h = 500
        scale = min(max_display_w / orig_w, max_display_h / orig_h, 1.0)
        display_w = int(orig_w * scale)
        display_h = int(orig_h * scale)

        fig, axes = plt.subplots(1, 2, figsize=(display_w / 80, display_h / 80))

        mask = create_mask(F, d)
        total_coeffs = F * F
        kept_coeffs = np.sum(mask)
        compression_pct = (1 - kept_coeffs / total_coeffs) * 100
        kept_pct = 100 - compression_pct

        axes[0].imshow(orig, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title(f"Originale ({orig_w}x{orig_h})\nF={F}, d={d}", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(compressa, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(
            f"Compressa\n" f"F={F}, d={d} | " f"Coef. mantenuti: {kept_pct:.1f}%",
            fontsize=10,
        )
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCompressorApp(root)
    root.mainloop()
