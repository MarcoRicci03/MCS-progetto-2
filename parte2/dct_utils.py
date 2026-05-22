import numpy as np
from scipy.fft import dctn, idctn


# crea la maschera che contiene solo le frequenze con k+l < d
def create_mask(F, d):
    mask = np.zeros((F, F), dtype=bool)
    for k in range(F):
        for l in range(F):
            if k + l < d:
                mask[k, l] = True
    return mask


# comprime un singolo blocco F x F
def compress_block(block, mask):
    # DCT2 della libreria
    c = dctn(block.astype(float), norm='ortho')

    # taglio delle frequenze alte
    c[~mask] = 0.0

    # trasformata inversa
    reconstructed_block = idctn(c, norm='ortho')

    # arrotondamento e clipping tra 0 e 255
    reconstructed_block = np.round(reconstructed_block)
    reconstructed_block = np.clip(reconstructed_block, 0, 255)

    return reconstructed_block.astype(np.uint8)


# divide l'immagine in blocchi F x F, scarta gli avanzi e ricompone
def compress_image(img_array, F, d):
    H, W = img_array.shape

    n_vertical_blocks = H // F
    n_horizontal_blocks = W // F

    H_crop = n_vertical_blocks * F
    W_crop = n_horizontal_blocks * F

    img_crop = img_array[:H_crop, :W_crop].copy()
    compressed_img = np.zeros_like(img_crop)

    mask = create_mask(F, d)

    for i in range(n_vertical_blocks):
        for j in range(n_horizontal_blocks):
            r0, r1 = i * F, (i + 1) * F
            c0, c1 = j * F, (j + 1) * F
            block = img_crop[r0:r1, c0:c1]
            compressed_img[r0:r1, c0:c1] = compress_block(block, mask)

    return img_crop, compressed_img