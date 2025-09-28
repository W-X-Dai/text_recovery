import cv2
import numpy as np

def add_shadow(img):
    h, w = img.shape[:2]
    # 建立一個漸層遮罩
    mask = np.zeros((h, w), dtype=np.float32)

    # 隨機選光源中心
    cx, cy = np.random.randint(0, w), np.random.randint(0, h)
    radius = np.random.randint(min(h,w)//2, max(h,w))

    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x-cx)**2 + (y-cy)**2)
            mask[y, x] = np.exp(-dist**2 / (2*(radius**2)))

    mask = cv2.normalize(mask, None, 0.5, 1.0, cv2.NORM_MINMAX)  # [0.5,1]
    shaded = (img.astype(np.float32) * mask).astype(np.uint8)
    return shaded

def add_noise_blur(img):
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    noisy = cv2.add(img, noise, dtype=cv2.CV_8U)
    blurred = cv2.GaussianBlur(noisy, (5,5), 1.5)
    return blurred


img = cv2.imread("clean.png", cv2.IMREAD_GRAYSCALE)
shadow_img = add_shadow(img)
shadow_img = add_noise_blur(shadow_img)
cv2.imwrite("shadow.png", shadow_img)
