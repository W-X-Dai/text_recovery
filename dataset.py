"""This file defines the dataset class for paired images."""
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class PairedImageDataset(Dataset):
    def __init__(self, clean_dir, trad_dir, image_size=256):
        self.clean_dir = clean_dir
        self.trad_dir = trad_dir
        self.clean_files = sorted(os.listdir(clean_dir))

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_name = self.clean_files[idx]
        # trad_name = clean_name.replace(".png", "_trad.png")
        trad_name = clean_name.replace(".png", ".png")

        clean_path = os.path.join(self.clean_dir, clean_name)
        trad_path = os.path.join(self.trad_dir, trad_name)

        clean = Image.open(clean_path).convert("RGB")
        trad = Image.open(trad_path).convert("RGB")

        clean = self.transform(clean)
        trad = self.transform(trad)

        return trad, clean  # noisy, clean
