"""
This script applies traditional image processing techniques to noisy images to simulate the effects of traditional document processing.
1. Generate clean images
2. Add noise to the clean images
3. Apply traditional image processing to noisy images <- We are here
"""
import cv2
import os
import numpy as np

src_dir = "noisy"
out_dir = "trad"
os.makedirs(out_dir, exist_ok=True)


n_png = 0
for fname in os.listdir(src_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff")):
        continue

    in_path = os.path.join(src_dir, fname)
    img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"skip {fname}, cannot read")
        continue
    background = cv2.GaussianBlur(img, (51, 51), 0)
    norm = cv2.divide(img.astype(np.float32), background.astype(np.float32) + 1)
    dst = np.empty_like(norm)
    norm = cv2.normalize(norm, dst, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_eq = clahe.apply(norm)

    """
    Adding a binary thresholding step to enhance text visibility.
    But in the new version, I skip this step to retain more of the original image characteristics.
    """
    binary = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
    img_eq = cv2.bitwise_and(img_eq, binary)


    out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".png")
    cv2.imwrite(out_path, img_eq)
    n_png += 1
    if n_png % 100 == 0:
        print(f"Processed {n_png} images")
