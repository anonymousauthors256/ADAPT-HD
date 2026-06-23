"""Building-block layers: quantizable linear layer and learnable threshold."""
import torch
import torch.nn as nn

from .quant import quant_global_STE


class LearnableSTEAbsoluteThreshold(nn.Module):
    """Learnable absolute threshold with a hard forward and soft (sigmoid)
    surrogate gradient via the straight-through estimator."""

    def __init__(self, init_ratio=0.0, sharpness=1.0, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.logit_ratio = nn.Parameter(torch.tensor(init_ratio, dtype=self.dtype))
        self.sharpness = sharpness

    def forward(self, x):
        x = x.to(self.dtype)
        ratio = self.logit_ratio

        # hard output
        hard = (x > ratio).to(self.dtype)
        # soft surrogate (for gradients)
        soft = torch.sigmoid(self.sharpness * (x - ratio))
        # STE trick
        return hard.detach() - soft.detach() + soft


class SimpleLinear(nn.Module):
    """Dense linear layer with optional weight/input quantization (STE)."""

    def __init__(self, in_features, out_features, num_bits_weight=None,
                 num_bits_inp=None, bias=True, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits_weight = num_bits_weight
        self.num_bits_inp = num_bits_inp
        self.data_type = dtype

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=self.data_type)
        )
        self.bias = (
            nn.Parameter(torch.randn(out_features, dtype=self.data_type))
            if bias else None
        )

    def forward(self, x):
        """x: (batch_size, in_features) -> (batch_size, out_features)."""
        if self.num_bits_inp:
            x_q = quant_global_STE(x, num_bits=self.num_bits_inp)
        else:
            x_q = x

        if self.num_bits_weight:
            w = quant_global_STE(self.weight, num_bits=self.num_bits_weight)
        else:
            w = self.weight

        y = x_q @ w.T
        if self.bias is not None:
            if self.num_bits_weight:
                y = y + quant_global_STE(self.bias, num_bits=self.num_bits_weight)
            else:
                y = y + self.bias
        return y
