"""
data/dataset.py
Medical image dataset loader.
Expects: root_dir/class_name/image_files
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def get_transforms(split="train", image_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


class MedicalImageDataset(Dataset):
    """
    Loads images from a folder where each subfolder = one class.
    root/
        class_a/img1.jpg ...
        class_b/img2.png ...
    """
    VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, root_dir, transform=None, max_per_class=None):
        self.root_dir  = root_dir
        self.transform = transform
        self.samples   = []
        self.classes   = []
        self.class_to_idx = {}
        self._load(max_per_class)

    def _load(self, max_per_class):
        classes = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])
        if not classes:
            raise ValueError(f"No class subfolders found in: {self.root_dir}")

        self.classes      = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(self.root_dir, cls)
            files = [
                f for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in self.VALID_EXT
            ]
            if max_per_class:
                files = files[:max_per_class]
            for f in files:
                self.samples.append((os.path.join(cls_dir, f), self.class_to_idx[cls]))

        logger.info(f"Loaded {len(self.samples)} samples | classes: {self.class_to_idx}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label


class _SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_client_dataloaders(
    client_id,
    data_path,
    image_size=224,
    batch_size=16,
    train_ratio=0.8,
    val_ratio=0.1,
    num_workers=0,
    seed=42,
    max_per_class=None,
):
    """Return dict with train/val/test DataLoaders + metadata."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    full_ds = MedicalImageDataset(data_path, transform=None, max_per_class=max_per_class)
    total   = len(full_ds)
    n_train = int(total * train_ratio)
    n_val   = int(total * val_ratio)
    n_test  = total - n_train - n_val

    gen = torch.Generator().manual_seed(seed)
    train_sub, val_sub, test_sub = random_split(full_ds, [n_train, n_val, n_test], generator=gen)

    train_ds = _SubsetWithTransform(train_sub, get_transforms("train", image_size))
    val_ds   = _SubsetWithTransform(val_sub,   get_transforms("val",   image_size))
    test_ds  = _SubsetWithTransform(test_sub,  get_transforms("test",  image_size))

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=False)

    return {
        "train":        loader(train_ds, shuffle=True),
        "val":          loader(val_ds,   shuffle=False),
        "test":         loader(test_ds,  shuffle=False),
        "num_classes":  len(full_ds.classes),
        "classes":      full_ds.classes,
        "dataset_size": n_train,
    }
