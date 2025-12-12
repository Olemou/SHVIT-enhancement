import torch
import torch.optim as optim
import numpy as np
from model.shvit_enhancement import SHViTEnhanced

def cosine_schedule(epoch, optimizer, base_lrs, warmup_epochs, max_epochs, min_lr=1e-6):
    """
    Quadratic warmup + cosine LR decay.

    Args:
        epoch (int): current epoch
        optimizer (Optimizer): torch optimizer
        base_lrs (list): base learning rates for each param group
        warmup_epochs (int)
        max_epochs (int)
        min_lr (float)

    Returns:
        list: current learning rates
    """
    lrs = []
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        if epoch < warmup_epochs:
            lr = min_lr + (base_lr - min_lr) * (epoch / warmup_epochs) ** 2
        else:
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
        pg['lr'] = lr.item() if isinstance(lr, np.float64) else lr # Convert numpy float to Python float
        lrs.append(pg['lr'])
    return lrs

# === Separate head and backbone parameters ===
head_params = list(SHViTEnhanced.head.parameters())
backbone_params = [p for n, p in SHViTEnhanced.named_parameters() if "head" not in n]

# === Optimizer with different LRs ===
optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': 7e-5},   # smaller LR for pretrained backbone
    {'params': head_params, 'lr': 1e-4}        # larger LR for new head
], weight_decay=0.05)

# Base LRs for cosine scheduler
base_lrs = [pg['lr'] for pg in optimizer.param_groups]