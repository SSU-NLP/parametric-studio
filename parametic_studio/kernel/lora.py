# Vendored from training/utils/module/lora.py (Copyright (c) Microsoft Corporation, Apache-2.0).
# deepspeed/dropout/fuse machinery removed — the studio needs a reversible, single-device LoRA only.
import math

import torch
from torch import nn
from torch.nn import functional as F


class LinearLayer_LoRA(nn.Module):
    """y = Wx + b + (x @ right @ left)·scaling — W/b frozen, only the low-rank pair trains.
    left starts at zero → the delta is 0 at init (forward identical until trained)."""

    def __init__(self, weight, lora_dim=8, lora_scaling=1, bias=None):
        super().__init__()
        self.weight, self.bias = weight, bias
        rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(columns, lora_dim, dtype=weight.dtype, device=weight.device))
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows, dtype=weight.dtype, device=weight.device))
        self.lora_scaling = lora_scaling / lora_dim
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        self.weight.requires_grad = False

    def forward(self, input):
        return F.linear(input, self.weight, self.bias) + (
            input @ self.lora_right_weight @ self.lora_left_weight) * self.lora_scaling


def _swap(model, name, new_module):
    parent, attr = name.rsplit(".", 1)
    setattr(model.get_submodule(parent), attr, new_module)


def convert_to_lora(model, lora_dim=8):
    """Swap every nn.Linear under model.layers for a LoRA layer (reversible via remove_lora)."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and ".layers." in name:
            _swap(model, name, LinearLayer_LoRA(module.weight, lora_dim, bias=module.bias))


def remove_lora(model):
    """Put the frozen original weight/bias back into plain nn.Linear — bit-exact undo."""
    for name, module in list(model.named_modules()):
        if isinstance(module, LinearLayer_LoRA):
            rows, cols = module.weight.shape
            lin = nn.Linear(cols, rows, bias=module.bias is not None)
            lin.weight = module.weight
            if module.bias is not None:
                lin.bias = module.bias
            module.weight.requires_grad_(True)
            _swap(model, name, lin)
