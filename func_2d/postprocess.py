import cv2
import numpy as np
import torch
from scipy import ndimage


def postprocess_segmentation_mask(mask, min_area_ratio=0.01, morph_kernel_size=5, use_largest_component=True):
    """
    对分割结果进行后处理，解决多区域、边缘噪点、孤立噪声等问题
    """
    if mask.ndim == 4:
        B = mask.shape[0]
        results = []
        for i in range(B):
            mask_i = mask[i]
            if mask_i.ndim == 3 and mask_i.shape[0] == 1:
                mask_i = mask_i[0]
            result_i = _process_single_mask(mask_i, min_area_ratio, morph_kernel_size, use_largest_component)
            results.append(result_i[np.newaxis, :, :])
        return np.concatenate(results, axis=0)

    return _process_single_mask(mask, min_area_ratio, morph_kernel_size, use_largest_component)


def _process_single_mask(mask, min_area_ratio, morph_kernel_size, use_largest_component):
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]

    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    H, W = mask.shape
    total_area = H * W
    min_area = int(total_area * min_area_ratio)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_close, connectivity=8)

    if use_largest_component:
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            processed_mask = (labels == largest_label).astype(np.uint8) * 255
        else:
            processed_mask = mask_close
    else:
        processed_mask = np.zeros_like(mask_close)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                processed_mask[labels == label] = 255

    blurred = cv2.GaussianBlur(processed_mask, (5, 5), 0)
    _, smooth_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return smooth_mask


def postprocess_batch_masks(pred_tensor, image_size=1024, min_area_ratio=0.01,
                            morph_kernel_size=5, use_largest_component=True):
    pred_np = pred_tensor.detach().cpu().numpy()
    B = pred_np.shape[0]
    processed_list = []

    for i in range(B):
        mask_i = pred_np[i]
        processed_mask = postprocess_segmentation_mask(
            mask_i,
            min_area_ratio=min_area_ratio,
            morph_kernel_size=morph_kernel_size,
            use_largest_component=use_largest_component
        )
        processed_mask = (processed_mask / 255.0).astype(np.float32)
        if processed_mask.ndim == 2:
            processed_mask = processed_mask[np.newaxis, :, :]
        processed_list.append(processed_mask)

    processed_tensor = torch.tensor(np.stack(processed_list), device=pred_tensor.device)
    return processed_tensor
