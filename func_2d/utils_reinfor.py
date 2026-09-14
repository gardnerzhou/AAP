import torch
import numpy as np
import torch.nn.functional as F
import os


def check_prompt_positions(pred_mask, coords, labels):
    """
    检查prompt点是否在正确的位置
    Args:
        pred_mask: 预测的分割掩码 [1, 1, H, W]
        coords: prompt点坐标 [N, 2]
        labels: prompt点标签 [N]
    Returns:
        bool: 所有点是否都在正确的位置
    """
    coords = coords.long()
    for i in range(len(coords)):
        x, y = coords[i]
        if x >= pred_mask.shape[3] or y >= pred_mask.shape[2]:
            return False
        pred_value = pred_mask[0, 0, y, x]
        if labels[i] == 1 and pred_value < 0.5:
            return False
        if labels[i] == 0 and pred_value > 0.5:
            return False
    return True


def generate_gaussian_map(coords, img_h, img_w, sigma=10):
    """
    生成高斯引导图
    coords: tensor, shape [N, 2]，每行为[x, y]
    img_h, img_w: 图像高宽
    sigma: 高斯核标准差
    返回: [1, img_h, img_w] 的高斯图
    """
    device = coords.device if isinstance(coords, torch.Tensor) else 'cpu'
    gaussian_map = torch.zeros((1, img_h, img_w), device=device)
    if coords.shape[0] == 0:
        return gaussian_map
    y = torch.arange(0, img_h, device=device).view(-1, 1).repeat(1, img_w)
    x = torch.arange(0, img_w, device=device).repeat(img_h, 1)
    for i in range(coords.shape[0]):
        cx, cy = coords[i]
        gaussian = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        gaussian_map += gaussian
    gaussian_map = gaussian_map / (gaussian_map.max() + 1e-8)
    return gaussian_map
