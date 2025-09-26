import os
import argparse
from typing import Tuple, List, Dict, Optional

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from network import UNet
from dataset import CityscapesSegDataset


# trainId -> color (Cityscapes standard 19 classes)
def get_trainid_palette() -> np.ndarray:
    # Index = trainId (0..18), 255 for ignore
    palette = {
        0: (128, 64, 128),  # road
        1: (244, 35, 232),  # sidewalk
        2: (70, 70, 70),  # building
        3: (102, 102, 156),  # wall
        4: (190, 153, 153),  # fence
        5: (153, 153, 153),  # pole
        6: (250, 170, 30),  # traffic light
        7: (220, 220, 0),  # traffic sign
        8: (107, 142, 35),  # vegetation
        9: (152, 251, 152),  # terrain
        10: (70, 130, 180),  # sky
        11: (220, 20, 60),  # person
        12: (255, 0, 0),  # rider
        13: (0, 0, 142),  # car
        14: (0, 0, 70),  # truck
        15: (0, 60, 100),  # bus
        16: (0, 80, 100),  # train
        17: (0, 0, 230),  # motorcycle
        18: (119, 11, 32),  # bicycle
        255: (0, 0, 0),  # ignore (rendered as black)
    }
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:] = 0
    for k, rgb in palette.items():
        lut[k] = rgb
    return lut


# labelId -> trainId mapping
# Reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
def get_id_to_trainid_lut() -> np.ndarray:
    id_to_trainid = {
        0: 255,
        1: 255,
        2: 255,
        3: 255,
        4: 255,
        5: 255,
        6: 255,
        7: 0,
        8: 1,
        9: 255,
        10: 255,
        11: 2,
        12: 3,
        13: 4,
        14: 255,
        15: 255,
        16: 255,
        17: 5,
        18: 255,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        29: 255,
        30: 255,
        31: 16,
        32: 17,
        33: 18,
    }
    lut = np.full(256, 255, dtype=np.uint8)
    for k, v in id_to_trainid.items():
        lut[k] = v
    return lut


# -----------------------------
# Visualization utilities
# -----------------------------
def colorize_label(label: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    label: [H, W] uint8, values in 0..18 or 255 (trainId space)
    palette: [256, 3] uint8
    return: [H, W, 3] uint8 colored image
    """
    return palette[label]


def overlay_segmentation(
    image_rgb: np.ndarray, color_label: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """
    image_rgb: [H, W, 3] uint8 original image
    color_label: [H, W, 3] uint8 colored segmentation
    alpha: float in [0,1]; alpha for color label
    """
    image_float = image_rgb.astype(np.float32)
    label_float = color_label.astype(np.float32)
    blended = (1 - alpha) * image_float + alpha * label_float
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


def visualize(
    data_root: str,
    split: str,
    batch_size: int = 2,
    num_workers: int = 2,
    img_height: int = 512,
    img_width: int = 1024,
    use_train_id: Optional[bool] = None,
    num_classes: Optional[int] = None,
    show_count: int = 4,
    overlay_alpha: float = 0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (img_width, img_height)

    # Dataset and loader
    ds = CityscapesSegDataset(root=data_root, split=split, img_size=img_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = UNet(out_channels=num_classes)
    model.load_state_dict(torch.load("checkpoints/model.pth"))
    model.to(device)
    model.eval()

    # Prepare LUTs and palette
    id2trainid_lut = get_id_to_trainid_lut()
    palette = get_trainid_palette()

    shown = 0
    
    os.makedirs("results", exist_ok=True)
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)  # [B, 3, H, W]
            mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(
                1, 3, 1, 1
            )
            std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(
                1, 3, 1, 1
            )

            vis = imgs.float() * std + mean  # unnormalize to [0,1]
            vis = vis.clamp(0.0, 1.0)
            vis = (vis * 255.0).round().to(torch.uint8)  # [B, 3, H, W], uint8

            logits = model(imgs)  # [B,C,h,w]
            H, W = imgs.shape[-2:]
            if logits.shape[-2:] != (H, W):
                logits = F.interpolate(
                    logits, size=(H, W), mode="bilinear", align_corners=False
                )

            preds = (
                logits.argmax(1).cpu().to(torch.uint8)
            )  # [B,H,W] in model class space

            for b in range(preds.size(0)):
                if shown >= show_count:
                    break

                pred = preds[b].numpy()
                if use_train_id:
                    pred_vis = pred
                else:
                    pred_vis = id2trainid_lut[pred]  # map to trainId for coloring

                color_label = colorize_label(pred_vis, palette)  # [H,W,3] uint8

                vis_img = (
                    vis[b].permute(1, 2, 0).cpu().numpy()
                )  # [H,W,3] uint8 (resized)

                overlay = overlay_segmentation(vis_img, color_label, alpha=overlay_alpha)

                # Show 3-panel: resized image / colored label / overlay
                fig = plt.figure(figsize=(12, 5))
                ax1 = plt.subplot(1, 3, 1)
                ax1.imshow(vis_img)
                ax1.set_title("Input Image")
                ax1.axis("off")
                ax2 = plt.subplot(1, 3, 2)
                ax2.imshow(color_label)
                ax2.set_title("Segemtation")
                ax2.axis("off")
                ax3 = plt.subplot(1, 3, 3)
                ax3.imshow(overlay)
                ax3.set_title("Overlay")
                ax3.axis("off")
                plt.tight_layout()
                plt.savefig('results/figure%d.png'%shown, dpi=200)
                plt.show(block=shown==show_count-1)

                shown += 1

            if shown >= show_count:
                break


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cityscapes inference and visualization"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Cityscapes root directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to run on",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument(
        "--show_count", type=int, default=3, help="How many images to display"
    )
    parser.add_argument(
        "--overlay_alpha", type=float, default=0.5, help="Alpha for overlay blending"
    )
    args = parser.parse_args()

    num_classes = 19

    visualize(
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_height=args.img_height,
        img_width=args.img_width,
        use_train_id=True,
        num_classes=num_classes,
        show_count=args.show_count,
        overlay_alpha=args.overlay_alpha,
    )


if __name__ == "__main__":
    main()
