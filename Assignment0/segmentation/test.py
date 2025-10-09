import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from network import UNet
from dataset import CityscapesSegDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, required=True, help="Cityscapes root directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size"
    )
    parser.add_argument("--img_height", type=int, default=256, help="Input height")
    parser.add_argument("--img_width", type=int, default=256, help="Input width")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (args.img_width, args.img_height)

    num_classes = 19
    ignore_index = 255

    test_set = CityscapesSegDataset(
        root=args.data_root,
        split="test",
        img_size=img_size,
        augment=False,
        use_train_id=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = UNet(out_channels=num_classes)
    model.load_state_dict(torch.load("checkpoints/model.pth"))
    model.to(device)
    _, test_miou = utils.evaluate(
        model,
        test_loader,
        device,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    print(f"[Test] mIoU {test_miou:.4f}")


if __name__ == "__main__":
    main()
