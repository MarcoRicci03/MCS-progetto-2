import numpy as np
from scipy.fft import dctn, idctn


# crea la maschera che contiene solo le frequenze con k+l < d
def create_mask(F, d):
    k = np.arange(F)[:, np.newaxis]
    l = np.arange(F)
    return (k + l) < d


# divide l'immagine in blocchi F x F, scarta gli avanzi e ricompone
def compress_image(img_array, F, d):
    H, W = img_array.shape

    H_new = H - (H % F)
    W_new = W - (W % F)
    img_cropped = img_array[:H_new, :W_new].astype(np.float64)

    n_vert = H_new // F
    n_horiz = W_new // F
    mask = create_mask(F, d)
    compressed = np.zeros_like(img_cropped, dtype=np.uint8)

    for i in range(n_vert):
        for j in range(n_horiz):
            r0, r1 = i * F, (i + 1) * F
            c0, c1 = j * F, (j + 1) * F
            block = img_cropped[r0:r1, c0:c1]
            c = dctn(block, norm="ortho")
            c[~mask] = 0.0
            reconstructed = idctn(c, norm="ortho")
            compressed[r0:r1, c0:c1] = np.clip(np.round(reconstructed), 0, 255).astype(
                np.uint8
            )

    stats = {
        "mask": mask,
        "kept_coeffs": int(np.sum(mask)),
        "total_coeffs": F * F,
    }
    return compressed, stats
