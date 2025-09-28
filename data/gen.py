import os
import random
import string
from PIL import Image, ImageDraw, ImageFont

def random_text(min_len=10, max_len=50):
    """randomly generate a line of text"""
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_letters + " ", k=length))

def draw_bold_text(draw, position, text, font, fill=0, bold=False):
    """在指定位置繪製文字，支援隨機粗體"""
    x, y = position
    if bold:
        # 模擬粗體，四個方向偏移
        offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for ox, oy in offsets:
            draw.text((x + ox, y + oy), text, font=font, fill=fill)
    else:
        draw.text((x, y), text, font=font, fill=fill)

def generate_images(
    save_dir="clean",
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    img_size=(384, 512),
    n_images=50000
):
    os.makedirs(save_dir, exist_ok=True)

    for i in range(n_images):
        if i % 100 == 0:
            print(f"Generating image {i}/{n_images}")
        img = Image.new("L", img_size, 255)  # 白底
        draw = ImageDraw.Draw(img)
        font_size = random.randint(20, 40)
        font = ImageFont.truetype(font_path, font_size)

        # 多行隨機文字
        y = 20
        for _ in range(random.randint(5, 12)):
            line = random_text()
            draw_bold_text(draw, (10, y), line, font, fill=0, bold=True)
            y += 30

        img.save(os.path.join(save_dir, f"{i:04d}.png"))

if __name__ == "__main__":
    generate_images()
