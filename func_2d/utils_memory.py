import torch
import torch.nn.functional as F
from typing import List


def update_memory_2(
                        memory_bank_list: List[torch.Tensor],
                        memory_bank_size: int) -> List[torch.Tensor]:
    """
    并行化优化后的贪心DPP选择算法（修正版）
    :param all_features: 待选样本特征列表
    :param all_data: 对应的原始数据
    :return: 选择后的子集数据列表
    """
    to_cat_feature_matrix = []
    for element in memory_bank_list:
        to_cat_feature_matrix.append((element[0].reshape(-1)).cuda(non_blocking=True))
    feature_matrix = torch.stack(to_cat_feature_matrix)
    feature_matrix = F.normalize(feature_matrix, p=2, dim=1)
    K = torch.mm(feature_matrix, feature_matrix.T)

    selected_indices = []
    remaining_indices = list(range(len(memory_bank_list)))

    K_inv = torch.empty((0, 0), device=K.device)
    det_K = 1.0

    for _ in range(memory_bank_size):
        if not remaining_indices:
            break

        k_i = K[remaining_indices][:, selected_indices]
        k_ii = K[remaining_indices, remaining_indices].unsqueeze(1)

        if len(selected_indices) == 0:
            det_new = k_ii
        else:
            v = torch.mm(k_i, K_inv)
            det_new = det_K * (k_ii - torch.sum(v * k_i, dim=1, keepdim=True) + 1e-6)

        best_idx_in_remaining = torch.argmax(det_new).item()
        best_idx = remaining_indices[best_idx_in_remaining]
        max_det = det_new[best_idx_in_remaining].item()

        selected_indices.append(best_idx)
        remaining_indices.pop(best_idx_in_remaining)

        if len(selected_indices) == 1:
            K_inv = 1.0 / K[best_idx, best_idx].unsqueeze(0).unsqueeze(0)
        else:
            k_i = K[best_idx, selected_indices[:-1]].unsqueeze(0)
            k_ii = K[best_idx, best_idx].unsqueeze(0).unsqueeze(0)

            v = torch.mm(K_inv, k_i.T)
            new_K_inv_top = torch.cat([K_inv + torch.mm(v, v.T) / (k_ii - torch.mm(k_i, v) + 1e-6),
                                     -v / (k_ii - torch.mm(k_i, v) + 1e-6)], dim=1)
            new_K_inv_bottom = torch.cat([-v.T / (k_ii - torch.mm(k_i, v) + 1e-6),
                                        1.0 / (k_ii - torch.mm(k_i, v) + 1e-6)], dim=1)
            K_inv = torch.cat([new_K_inv_top, new_K_inv_bottom], dim=0)

        det_K = max_det

    return [memory_bank_list[i] for i in selected_indices]
