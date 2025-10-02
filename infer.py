"""This file is the inference script for the UNet model."""
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import yaml

from model import UNet

def build_transform():
    return T.Compose([
        T.ToTensor()
    ])

def to_pil(t):
    t = t.clamp(0, 1).cpu()
    return T.ToPILImage()(t)

def out_name(fname: str, keep_trad_suffix: bool) -> str:
    if keep_trad_suffix:
        return fname
    return re.sub(r"_trad(?=\.)", "", fname)

@torch.no_grad()
def main():
    with open("conf.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    infer_cfg = cfg["infer"]

    input_dir = Path(infer_cfg["input_dir"])
    output_dir = Path(infer_cfg["output_dir"])
    ckpt_path = infer_cfg["ckpt"]
    keep_trad_suffix = infer_cfg.get("keep_trad_suffix", False)

    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    model.eval()

    transform = build_transform()

    # Capture all images in the input directory
    for input_image in sorted(input_dir.iterdir()):
        if not input_image.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        img = Image.open(input_image).convert("RGB")

        # Convert to tensor
        x = transform(img).unsqueeze(0).to(device)
        h, w = x.shape[2], x.shape[3]

        # Padding
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Inference
        pred = model(x)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            pred = pred[:, :, :h, :w]

        pred_img = to_pil(pred.squeeze(0))

        out_filename = out_name(input_image.name, keep_trad_suffix)
        pred_img.save(output_dir / out_filename)
        print(f"Saved result to {output_dir/out_filename}")

if __name__ == "__main__":
    main()
