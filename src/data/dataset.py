import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class DeepGlobeDataset(Dataset):
    """
    Dataset loader for the DeepGlobe dataset from Kaggle.
    Assumes data is placed under:
        data/train/ and data/val/
    """

    def __init__(self, root_dir='data', split='train', img_size=1024):
        """
        Args:
            root_dir (str): path to folder containing train/ and val/
            split (str): 'train' or 'val'
            img_size (int): size to resize images (square)
        """
        assert split in ('train', 'val'), "split must be 'train' or 'val'"
        self.split = split
        self.root_dir = root_dir
        self.img_size = img_size

        # folder path
        folder = os.path.join(root_dir, split)

        # list of image files
        self.image_files = [
            os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {folder}")

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # map to [-1,1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img
