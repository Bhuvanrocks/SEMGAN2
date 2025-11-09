# prepare_domains.py
import os
from PIL import Image, ImageFilter, ImageEnhance
import shutil

SRC_DIR = "Electron_Microscopy_Images/SEM"  # change if different
OUT_DIR = "data"
A_DIR = os.path.join(OUT_DIR, "A")
B_DIR = os.path.join(OUT_DIR, "B")
os.makedirs(A_DIR, exist_ok=True)
os.makedirs(B_DIR, exist_ok=True)

# allowed image extensions
EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def transform_image_for_B(img: Image.Image) -> Image.Image:
    # A deterministic-ish transform pipeline to create a second domain.
    # You can modify this to simulate other imaging conditions.
    # Steps: slight blur + adaptive contrast + sharpen + add gaussian noise (approx)
    img = img.convert("L")  # grayscale
    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    return img

print("Scanning", SRC_DIR)
count = 0
for fname in os.listdir(SRC_DIR):
    if not fname.lower().endswith(EXT):
        continue
    src_path = os.path.join(SRC_DIR, fname)
    try:
        img = Image.open(src_path)
    except Exception as e:
        print("Skipping", src_path, ":", e)
        continue

    base_name = f"{count:05d}.png"
    a_out = os.path.join(A_DIR, base_name)
    b_out = os.path.join(B_DIR, base_name)

    # save A (original, converted to grayscale PNG)
    img.convert("L").save(a_out)

    # create and save B (transformed)
    b_img = transform_image_for_B(img)
    b_img.save(b_out)

    count += 1

print(f"Prepared {count} images in {A_DIR} and {B_DIR}")
