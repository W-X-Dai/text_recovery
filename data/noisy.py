"""
This script adds noise and distortions to clean images to simulate real-world conditions.
1. Generate clean images
2. Add noise to the clean images <- We are here
3. Apply traditional image processing to noisy images
"""
import cv2
import numpy as np
import os

def add_shadow(img):
    """Add random shadow to the image."""
    h, w = img.shape[:2]
    cx, cy = np.random.randint(0, w), np.random.randint(0, h)
    radius = np.random.randint(min(h, w)//3, min(h, w)//2)

    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    mask = np.exp(-dist**2 / (2 * (radius**2)))
    mask = cv2.normalize(mask, None, 0.2, 0.8, cv2.NORM_MINMAX)
    shaded = (img.astype(np.float32) * mask).astype(np.uint8)
    return shaded

def add_noise_blur(img):
    """Add slight noise and blur to the image."""
    noise = np.random.normal(0, 7, img.shape).astype(np.int16)
    noisy = cv2.add(img, noise, dtype=cv2.CV_8U)

    blurred = cv2.GaussianBlur(noisy, (5, 5), 2)

    # Random motion blur
    kernel_size = 6
    kernel = np.zeros((kernel_size, kernel_size))
    if np.random.rand() < 0.5:  
        kernel[kernel_size//2, :] = 1.0
    else:
        kernel[:, kernel_size//2] = 1.0
    kernel /= kernel_size
    blurred = cv2.filter2D(blurred, -1, kernel)

    # Random brightness adjustment
    bias = np.random.randint(-15, 15)
    blurred = cv2.add(blurred, bias)

    return blurred

def add_jpeg_compression(img, quality=30):
    """Simulate JPEG compression artifacts to degrade image quality."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

# File paths
input_dir = "clean"
output_dir = "noisy"
os.makedirs(output_dir, exist_ok=True)

n_img = 0
for fname in os.listdir(input_dir):
    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        out = add_shadow(img)
        out = add_noise_blur(out)
        out = add_jpeg_compression(out, quality=30)  # Simulate mobile photo/low-compression scan effect

        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, out)
        n_img += 1
        if n_img % 100 == 0:
            print(f"Processed {n_img} images")
