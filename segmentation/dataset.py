import os
from pathlib import Path
from typing import Tuple, Optional, List
import random

import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset

import torchvision.transforms.functional as TF

def random_resize(img: Image.Image, label: Image.Image, scale_range=(0.5, 2.0)):
    # Randomly resize image
    scale_factor = random.uniform(scale_range[0], scale_range[1])
    new_size = (int(scale_factor*img.width), int(scale_factor*img.height))
    # resize img and lable
    img_resized = img.resize(new_size, Image.BILINEAR)
    label_resized = label.resize(new_size, Image.NEAREST)
    img = img_resized
    label = label_resized

    return img, label

def random_crop(img: Image.Image, label: Image.Image, size: Tuple[int, int]):
    # Random crop to the target size; pad if smaller than desired size
    # The input size is a tuple (width, height)
    target_width, target_height = size
    img_width, img_height = img.size
    # if image size is smaller than target size, we should make up some part
    if img_width < target_width or img_height < target_height:
        pad_width = max(0, target_width - img_width)
        pad_height = max(0, target_height - img_height)

        img = Image.new('RGB', (img_width + pad_width, img_height + pad_height), 0)
        img.paste(img, (0, 0))
        label = Image.new('L', (img_width + pad_width, img_height + pad_height), 255)
        label.paste(label, (0, 0))
    # randomly choose crop part
    x = random.randint(0, img.size[0] - target_width)
    y = random.randint(0, img.size[1] - target_height)
    # crop image and label
    img_cropped = img.crop((x, y, x + target_width, y + target_height))
    label_cropped = label.crop((x, y, x + target_width, y + target_height))
    img = img_cropped
    label = label_cropped
    return img, label

def center_crop(img: Image.Image, label: Image.Image, size: Tuple[int, int]):
    # For validation: resize keeping aspect ratio so that the crop fits, then center-crop
    if img.size != size:
        img = ImageOps.fit(img, size, method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))
        label = ImageOps.fit(label, size, method=Image.Resampling.NEAREST, centering=(0.5, 0.5))
    return img, label

def to_tensor_and_norm(img: Image.Image):
    # Convert PIL image to tensor and normalize with ImageNet stats
    return TF.normalize(TF.to_tensor(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class CityscapesSegDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: Tuple[int, int] = (512, 1024),
        augment: bool = True,
        use_train_id: bool = False,
    ):
        """
        root: Cityscapes root containing leftImg8bit/ and gtFine/
        split: 'train' or 'val'
        img_size: output size (W, H)
        augment: whether to apply training augmentations
        use_train_id: if True, convert labelIds to trainId (19 classes setup)
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.use_train_id = use_train_id

        self.left_dir = self.root / "leftImg8bit" / split
        self.gt_dir = self.root / "gtFine" / split

        if not self.left_dir.exists() or not self.gt_dir.exists():
            raise FileNotFoundError(f"Expect {self.left_dir} and {self.gt_dir} to exist")

        self.samples = self._gather_pairs()

        # Optional: trainId mapping (from official labels.py)
        self.id_to_trainid = None
        if self.use_train_id:
            # Reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
            id_to_trainid = {
                0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0,
                8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255,
                16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
                24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16,
                32: 17, 33: 18,
            }
            self.id_to_trainid = np.full(256, 255, dtype=np.uint8)
            for k, v in id_to_trainid.items():
                self.id_to_trainid[k] = v

    def _gather_pairs(self) -> List[Tuple[Path, Path]]:
        # Collect (left image, gt labelIds) pairs based on filename stem
        pairs = []
        for city_dir in sorted(self.left_dir.glob("*")):
            if not city_dir.is_dir():
                continue
            for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                stem = "_".join(img_path.stem.split("_")[:-1])
                gt_name = f"{stem}_gtFine_labelIds.png"
                gt_path = self.gt_dir / city_dir.name / gt_name
                pairs.append((img_path, gt_path))
        if len(pairs) == 0:
            raise RuntimeError(f"No pairs found under {self.left_dir}")
        return pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        label = Image.open(gt_path)  # single-channel labelIds

        if self.augment:
            # Random resize, random crop, random horizontal flip
            img, label = random_resize(img, label, scale_range=(0.5, 2.0))
            img, label = random_crop(img, label, self.img_size)
            if random.random() < 0.5:
                img = TF.hflip(img)
                label = TF.hflip(label)
        else:
            img, label = center_crop(img, label, self.img_size)

        img_t = to_tensor_and_norm(img)
        label_np = np.array(label, dtype=np.uint8)

        if self.use_train_id and self.id_to_trainid is not None:
            label_np = self.id_to_trainid[label_np]

        label_t = torch.from_numpy(label_np.astype(np.int64))  # [H, W], ignore_index=255

        return {
            "image": img_t,          # [3, H, W], float
            "label": label_t,          # [H, W], long
        }