import torch
import torch.nn as nn

@torch.no_grad()
def compute_mIoU(conf_mat: torch.Tensor) -> float:
    # Compute mean IoU from a confusion matrix
    diag = torch.diag(conf_mat)
    sum_rows = conf_mat.sum(dim=1)
    sum_cols = conf_mat.sum(dim=0)
    denom = (sum_rows + sum_cols - diag).clamp_min(1)
    iou = diag / denom
    return float(iou.mean().item())


@torch.no_grad()
def update_confusion_matrix(conf_mat, preds, gts, num_classes, ignore_index=255):
    # Update confusion matrix given predictions and ground-truth labels
    valid = gts != ignore_index
    preds = preds[valid]
    gts = gts[valid]
    k = (gts >= 0) & (gts < num_classes)
    preds = preds[k]
    gts = gts[k]
    idx = gts * num_classes + preds
    conf = torch.bincount(idx, minlength=num_classes * num_classes)
    conf = conf.reshape(num_classes, num_classes).to(conf_mat.device)
    conf_mat += conf
    return conf_mat

@torch.no_grad()
def evaluate(model, val_dataloader, device, num_classes, ignore_index=255):
    # Evaluate on validation set: average CE loss and mIoU
    model.eval()
    conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    total_loss = 0.0
    for batch in val_dataloader:
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item()
        preds = out.argmax(1)
        conf_mat = update_confusion_matrix(
            conf_mat, preds.cpu(), labels.cpu(), num_classes, ignore_index
        )
    miou = compute_mIoU(conf_mat.float())
    avg_loss = total_loss / max(1, len(val_dataloader))
    return avg_loss, miou