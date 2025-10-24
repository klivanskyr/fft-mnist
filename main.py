import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from PIL import Image

# -------- load MNIST --------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
images = np.concatenate([x_train, x_test], axis=0).astype(np.float32)
labels = np.concatenate([y_train, y_test], axis=0)

# -------- pick 10 samples per digit --------
digit_to_idxs = {d: [] for d in range(10)}
for i, lab in enumerate(labels):
    if len(digit_to_idxs[lab]) < 10:
        digit_to_idxs[lab].append(i)
    if all(len(v) == 10 for v in digit_to_idxs.values()):
        break

# -------- helpers --------
def to_uint8(img):
    m, M = img.min(), img.max()
    if M - m < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - m) / (M - m) * 255.0).astype(np.uint8)

def make_pair_tile_200x200(img_28):
    """
    Returns a 200x200 RGB tile with original digit (left 100x200)
    and FFT magnitude (right 100x200). Digit and FFT are each
    100x100 scaled and centered vertically inside 100x200.
    """
    # original to uint8 28x28
    orig = to_uint8(img_28)

    # FFT magnitude (log for visibility)
    fft2 = np.fft.fft2(img_28)
    fft2_shift = np.fft.fftshift(fft2)
    mag = np.abs(fft2_shift)
    mag_log = np.log1p(mag)  # 28x28 float
    fft_u8 = to_uint8(mag_log)

    # resize both to 100x100
    orig_100 = Image.fromarray(orig).resize((100, 100), Image.BILINEAR)
    fft_100  = Image.fromarray(fft_u8).resize((100, 100), Image.BILINEAR)

    # create 200x200 tile and paste each into a 100x200 half, centered vertically
    tile = Image.new("L", (200, 200), color=0)  # grayscale
    # vertical centering: paste at y=50 so 100px image sits in 200px height
    tile.paste(orig_100, (0, 50))
    tile.paste(fft_100,  (100, 50))

    # convert to RGB for consistent display/saving
    tile = tile.convert("RGB")
    return tile

# -------- build 10x10 grid of 200x200 pair tiles --------
tiles = []
for d in range(10):
    for j in range(10):
        idx = digit_to_idxs[d][j]
        tile = make_pair_tile_200x200(images[idx] / 255.0)  # normalize for FFT stability
        tiles.append(tile)

# stitch into a single 2000x2000 image
grid_w = 10
grid_h = 10
tile_size = 200
big = Image.new("RGB", (grid_w * tile_size, grid_h * tile_size), color=(0, 0, 0))

k = 0
for r in range(grid_h):
    for c in range(grid_w):
        big.paste(tiles[k], (c * tile_size, r * tile_size))
        k += 1

# show and save
plt.figure(figsize=(10, 10))
plt.imshow(big)
plt.axis("off")
plt.title("Original (left) + FFT (right), 10 samples per digit")
plt.show()

big.save("mnist_pairs_200x200_grid.png")
print("Saved to mnist_pairs_200x200_grid.png")
