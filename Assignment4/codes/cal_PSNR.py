import os
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def calculate_psnr(gt_img, pred_img):
    """计算单张图像的 PSNR"""
    mse = np.mean((gt_img - pred_img) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

def get_epoch_number(filename):
    """从文件名中提取 epoch 数字"""
    match = re.search(r'_(\d+).png', filename)
    return int(match.group(1)) if match else -1

def compute_psnr_for_folder(folder_path):
    """读取目录下 gt/pred 图片并计算 PSNR"""
    gt_files = [f for f in os.listdir(folder_path) if f.startswith('gt_')]
    pred_files = [f for f in os.listdir(folder_path) if f.startswith('pred_')]

    # 建立 epoch->文件映射
    gt_dict = {get_epoch_number(f): f for f in gt_files}
    pred_dict = {get_epoch_number(f): f for f in pred_files}

    common_epochs = sorted(set(gt_dict.keys()) & set(pred_dict.keys()))
    psnr_list = []

    for epoch in common_epochs:
        gt_path = os.path.join(folder_path, gt_dict[epoch])
        pred_path = os.path.join(folder_path, pred_dict[epoch])

        gt_img = np.array(Image.open(gt_path).convert('RGB')).astype(np.float32)
        pred_img = np.array(Image.open(pred_path).convert('RGB')).astype(np.float32)

        psnr = calculate_psnr(gt_img, pred_img)
        psnr_list.append((epoch, psnr))
        print(f"Epoch {epoch}: PSNR = {psnr:.2f}")

    return psnr_list

def plot_psnr(psnr_list, title):
    """绘制 PSNR 曲线并保存图像"""
    epochs, psnrs = zip(*psnr_list)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, psnrs, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title(title)
    plt.grid(True)

    # 确保保存路径存在
    save_path = f"{title.replace(' ', '_')}.png"
    directory = os.path.dirname(save_path)

    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # 保存图像
    plt.savefig(save_path)
    plt.close()  # 关闭图形，释放内存

if __name__ == "__main__":
    folders = ["output/3DGS_random", "output/NeRF", "output/3DGS"]
    for folder in folders:
        if os.path.exists(folder):
            print(f"\nProcessing folder: {folder}")
            psnr_list = compute_psnr_for_folder(folder)
            plot_psnr(psnr_list, title=f"PSNR over Epochs ({folder})")
        else:
            print(f"Folder not found: {folder}")