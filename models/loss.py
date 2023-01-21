import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss


def balanced_bce_loss(output, target, class_weight=None):
    loss = BCELoss(weight=class_weight[target.long()])
    return loss(output, target)


# this contains sigmoid itself
def balanced_bcewithlogits_loss(output, target, class_weight=None):
    loss = BCEWithLogitsLoss(pos_weight=class_weight[1])
    return loss


# ref: https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
def binary_focal_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
    reduction: str = "sum",
    eps: float = 1e-8,
) -> torch.Tensor:
    p_t = output
    loss_tmp = -alpha * torch.pow(1 - p_t, gamma) * target * torch.log(p_t + eps) \
    - (1 - alpha) * torch.pow(p_t, gamma) * (1 - target) * torch.log(1 - p_t + eps)

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)

    return loss


def recon_loss(output, data, size_average=True):
    x_num, x_cat, x_num_mask, x_cat_mask = data
    NUM_LEN, CAT_LEN = x_num.size(-1), x_cat.size(-1)
    output_num, output_cat = output[..., :NUM_LEN], output[..., -CAT_LEN:]

    # not consider inverse zero padding part and missing part
    # print(x_num_mask.size())
    # print(x_cat_mask.size())
    output_num, x_num = output_num[x_num_mask > 0], x_num[x_num_mask > 0]
    output_cat, x_cat = output_cat[x_cat_mask > 0], x_cat[x_cat_mask > 0]
    output_cat = torch.sigmoid(output_cat)
    # print(output_num.size(0), x_num.size(0))
    # print(output_cat.size(0), x_cat.size(0))
    # print(output_cat.dtype, x_cat.dtype)

    # print(output_cat[:100], x_cat[:100])
    # import os
    # os._exit(0)
    mse_loss = MSELoss()
    ce_loss = BCELoss()
    if size_average:
        N_ce = x_cat_mask.data.sum()
    else:
        N_ce = 1.
    loss = mse_loss(output_num, x_num) + ce_loss(output_cat, x_cat)/N_ce

    return loss


def mask_loss(output, data, size_average=True):
    x_num, x_cat, x_num_mask, x_cat_mask = data
    x_mask = torch.cat([x_num_mask, x_cat_mask], 2)
    feat_dim = x_mask.size()[-1]

    # not consider inverse zero padding part
    non_pad = torch.sum(x_mask, 2) > 0
    non_pad = torch.unsqueeze(non_pad, -1)
    non_pad = non_pad.expand(-1, -1, feat_dim)
    output, x_mask = output[non_pad], x_mask[non_pad]
    output = torch.sigmoid(output)
    # print(output.size())
    # print(x_mask.size())

    ce_loss = BCELoss()
    if size_average:
        N_ce = torch.numel(x_mask)
    else:
        N_ce = 1.
    loss = ce_loss(output, x_mask) / N_ce

    return loss


# Learning to Rank
# paper: https://dl.acm.org/doi/pdf/10.1145/1273496.1273513
# code: https://github.com/allegro/allRank/blob/master/allrank/models/losses/listNet.py
eps = 1E-10
def prob_model(pi: torch.Tensor, type_="topk", k=5):
    """
    type_:
        topk
        target_1
    """
    P = torch.ones(1, device="cuda:0", requires_grad=True)
    if type_ == "topk":
        N = k
    elif type_ == "target_1":
        N = len(pi)
    for i in range(N):
        P = P * F.softmax(pi[i:], dim=0)[0]
    return P


def rank_loss(event_time, target, output, k=10, alpha=5.0, focal=False, gamma=2.0):
    """
    Compute loss: L(y, z)
    compute "y" by event_time
    output corresponds to "z" in paper
    Top k probability by choosing target == 1 (positive label) in a mini-batch
    """
    # After min-max scaling, y in [0, 1]
    min_t = torch.min(event_time)
    max_t = torch.max(event_time)
    y = (event_time - min_t) / (max_t - min_t)
    # 1 - y in order to correspond to hazard ratio
    d_t = event_time.size(0)
    y = torch.ones(d_t, device="cuda:0", requires_grad=True) - y
    z = torch.squeeze(output)

    type_ = "target_1"
    # type_ = "topk"
    if type_ == "topk":
        topk, topk_idx = torch.topk(y, k)
    elif type_ == "target_1":
        topk_idx = torch.squeeze(target == 1)

    y = y[topk_idx]
    z = z[topk_idx]

    Py = prob_model(y, type_, k)
    Pz = prob_model(z, type_, k)

    Pz = Pz + eps
    logPz = torch.log(Pz)

    if focal:
        alpha = alpha * (1 - Pz)**gamma
    return - alpha * torch.sum(Py * logPz)
