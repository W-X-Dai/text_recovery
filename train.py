"""This file is the main training script for the UNet model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_msssim
import yaml
import random
from dataset import PairedImageDataset
from model import UNet
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm


# Loss functions
criterion_l1 = nn.L1Loss()
criterion_ssim = pytorch_msssim.SSIM(data_range=1.0, size_average=True, channel=3)

# Sobel filter for edge loss
sobel_x = torch.tensor([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]], dtype=torch.float32).view(1,1,3,3)
sobel_y = sobel_x.transpose(2,3)

def sobel_filter(img):
    # img: (B,C,H,W)
    sobel_x_ = sobel_x.to(img.device).repeat(img.size(1),1,1,1)
    sobel_y_ = sobel_y.to(img.device).repeat(img.size(1),1,1,1)
    gx = F.conv2d(img, sobel_x_, padding=1, groups=img.size(1))
    gy = F.conv2d(img, sobel_y_, padding=1, groups=img.size(1))
    return torch.sqrt(gx**2 + gy**2 + 1e-6)

def edge_loss(pred, target):
    return F.l1_loss(sobel_filter(pred), sobel_filter(target))

def combined_loss(pred, target):
    """Combine L1, SSIM, and edge loss."""
    l1 = criterion_l1(pred, target)
    ssim = 1 - criterion_ssim(pred, target)  # maximize SSIM
    edge = edge_loss(pred, target)
    return l1 + 0.5*ssim + 0.5*edge


def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PairedImageDataset(
        config["data"]["clean_dir"],
        config["data"]["trad_dir"],
        config["data"]["image_size"]
    )
    loader = DataLoader(dataset,
                        batch_size=config["train"]["batch_size"],
                        shuffle=True)

    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config["train"]["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode="min",
                                                     factor=0.5,
                                                     patience=5)

    save_path = Path(config["train"]["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["train"]["epochs"]):
        model.train()
        epoch_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}", leave=False)
        for trad, clean in pbar:
            trad, clean = trad.to(device), clean.to(device)
            pred = model(trad)
            loss = combined_loss(pred, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{config['train']['epochs']} Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)

        if (epoch + 1) % config["train"]["save_every"] == 0:
            ckpt_path = save_path / f"checkpoint_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)

            # Save 10 sample images
            model.eval()
            with torch.no_grad():
                idxs = random.sample(range(len(dataset)), 10)
                samples = [dataset[i] for i in idxs]
                trad_batch = torch.stack([s[0] for s in samples]).to(device)
                clean_batch = torch.stack([s[1] for s in samples]).to(device)
                pred_batch = model(trad_batch)

            img_dir = save_path / f"results_epoch{epoch+1}"
            img_dir.mkdir(parents=True, exist_ok=True)

            for i in range(5):
                comparison = torch.cat([
                    trad_batch[i:i+1].clamp(0,1),
                    pred_batch[i:i+1].clamp(0,1),
                    clean_batch[i:i+1].clamp(0,1)
                ], dim=0)
                save_image(comparison, img_dir / f"compare_{i}.png", nrow=3)

            print(f"Saved 5 comparison images to {img_dir}")


if __name__ == "__main__":
    with open("conf.yaml", "r") as f:
        config = yaml.safe_load(f)
    train(config)
