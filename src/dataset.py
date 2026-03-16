"""
dataset.py — PyTorch Dataset for Cloud Removal GAN
Maps processed (cloudy) images → clean (target) satellite images.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CloudRemovalDataset(Dataset):
    """
    Loads pairs of (cloudy_image, clean_image) for supervised GAN training.

    Args:
        processed_dir (str): Path to folder with cloud-augmented images.
        target_dir (str): Path to folder with clean original images.
        transform: torchvision transforms to apply to both images.
    """

    def __init__(self, processed_dir: str, target_dir: str, transform=None):
        self.processed_dir = processed_dir
        self.target_dir = target_dir
        self.transform = transform

        # Index files by serial number for robust matching
        self.target_map = self._index_by_serial(target_dir)
        self.processed_map = self._index_by_serial(processed_dir)

        # Keep only pairs that exist in both directories
        self.matched = sorted(
            k for k in self.processed_map if k in self.target_map
        )

        if not self.matched:
            raise RuntimeError(
                f"No matching image pairs found between:\n"
                f"  {processed_dir}\n  {target_dir}"
            )

        print(f"[Dataset] Found {len(self.matched)} matched image pairs.")

    def __len__(self):
        return len(self.matched)

    def __getitem__(self, idx):
        key = self.matched[idx]
        processed_img = Image.open(self.processed_map[key]).convert('RGB')
        target_img = Image.open(self.target_map[key]).convert('RGB')

        if self.transform:
            processed_img = self.transform(processed_img)
            target_img = self.transform(target_img)

        return processed_img, target_img

    @staticmethod
    def _index_by_serial(directory: str) -> dict:
        """Build a {serial_number: filepath} dict from directory .jpg files."""
        index = {}
        for fname in os.listdir(directory):
            if not fname.lower().endswith('.jpg'):
                continue
            serial = fname.split('.')[0]  # e.g. "1234" from "1234.jpg"
            index[serial] = os.path.join(directory, fname)
        return index


def get_default_transform(img_size: int = 64):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = CloudRemovalDataset(
        processed_dir='data/processed',
        target_dir='data/raw',
        transform=get_default_transform(64),
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    inputs, targets = next(iter(loader))
    print("Input batch shape:", inputs.shape)
    print("Target batch shape:", targets.shape)
