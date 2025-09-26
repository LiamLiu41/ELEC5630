import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from network import UNet
from dataset import CityscapesSegDataset


def init_weights(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Task2.1: Loss function
def cross_entropy_loss(
    prediction: torch.Tensor, labels: torch.Tensor, ignore_index=255
):
    # prediction: [B, C, H, W], float32
    # labels: [B, H, W], int64
    # return the mean cross entropy loss
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, required=True, help="Cityscapes root directory"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--img_height", type=int, default=256, help="Input height")
    parser.add_argument("--img_width", type=int, default=256, help="Input width")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for Adam"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (args.img_width, args.img_height)

    num_classes = 19
    ignore_index = 255

    train_set = CityscapesSegDataset(
        root=args.data_root,
        split="train",
        img_size=img_size,
        augment=True,
        use_train_id=True,
    )
    val_set = CityscapesSegDataset(
        root=args.data_root,
        split="val",
        img_size=img_size,
        augment=False,
        use_train_id=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = UNet(out_channels=num_classes)
    model.apply(init_weights)
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        tic = time.time()
        for i, batch in enumerate(train_loader):
            imgs = batch["image"].to(device)  # [B, C, H, W], float32
            labels = batch["label"].to(device)  # [B, H, W], int64
            
            optimizer.zero_grad()

            # Task2.2: Feed forward
            ############## your code ############
            loss : torch.Tensor # Please compute the scalar loss

            #####################################

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 50 == 0:
                avg = running_loss / 50
                speed = (i + 1) / (time.time() - tic + 1e-9)
                print(
                    f"Epoch {epoch} | iter {i+1}/{len(train_loader)} | loss {avg:.4f} | iters/s {speed:.2f}"
                )
                running_loss = 0.0

        val_loss, val_miou = utils.evaluate(
            model,
            val_loader,
            device,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        print(f"[Val] Epoch {epoch} | loss {val_loss:.4f} | mIoU {val_miou:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model.pth")


if __name__ == "__main__":
    main()
