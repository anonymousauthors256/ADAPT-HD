"""Hyperdimensional computing models: AdaptHD, BipolarHDC, and inference variant."""
import torch
import torch.nn as nn

from .quant import quant_global_STE
from .layers import SimpleLinear, LearnableSTEAbsoluteThreshold


class BipolarHDC(nn.Module):
    """Pure bipolar HDC baseline (no adaptive masking)."""

    def __init__(self, f, m, D, bit_width=8, weight_bit_width=8,
                 inp_bit_width=None, bias_linear=None, verbose=False,
                 dtype=torch.float32):
        super().__init__()
        self.f = f
        self.m = m
        self.bias_linear = bias_linear
        self.verbose = verbose
        self.bit_width = bit_width
        self.weight_bit_width = weight_bit_width
        self.inp_bit_width = inp_bit_width
        self.dtype = dtype

        self.M = nn.Parameter(torch.randn(m, D, dtype=self.dtype))
        self.lin_enc = SimpleLinear(
            f, D, bias=self.bias_linear,
            num_bits_inp=self.inp_bit_width,
            num_bits_weight=self.weight_bit_width,
            dtype=self.dtype,
        )

    def forward(self, x):
        z_enc = self.lin_enc(x)
        M_q = quant_global_STE(self.M, num_bits=self.bit_width)
        z_enc_q = quant_global_STE(z_enc, num_bits=self.bit_width)
        logits = z_enc_q @ M_q.T
        return logits


class AdaptHD(nn.Module):
    """AdaptHD: HDC encoder with a learnable adaptive sparsity mask.

    A pooled, thresholded bitmask (g_bin) gates blocks of the encoded
    hypervector, enabling per-sample dimension reduction.
    """

    def __init__(self, f, m, k, D, threshold=0.1, bit_width=8,
                 weight_bit_width=8, inp_bit_width=None, pool_bitwidth=None,
                 bias_linear=None, dropout_p=0, verbose=False,
                 dtype=torch.float32):
        super().__init__()
        self.f = f
        self.m = m
        self.k = k
        self.bias_linear = bias_linear
        self.verbose = verbose
        self.threshold = threshold
        self.bit_width = bit_width
        self.weight_bit_width = weight_bit_width
        self.inp_bit_width = inp_bit_width
        self.pool_bitwidth = pool_bitwidth
        self.interpolation = int(D) // int(k)
        self.dtype = dtype
        self.dropout_p = dropout_p

        if self.dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)

        self.M = nn.Parameter(torch.randn(m, D, dtype=self.dtype))
        self.adaptivepool = nn.AdaptiveMaxPool1d(k)
        self.lin_enc = SimpleLinear(
            f, D, bias=self.bias_linear,
            num_bits_inp=self.inp_bit_width,
            num_bits_weight=self.weight_bit_width,
            dtype=self.dtype,
        )
        self.bin_threshold = LearnableSTEAbsoluteThreshold(
            init_ratio=threshold, sharpness=1, dtype=self.dtype
        )

    def forward(self, x):
        # bitmask generation
        bin_logits = self.adaptivepool(x)
        g_bin = self.bin_threshold(bin_logits)
        z_enc = self.lin_enc(x)

        M_q = quant_global_STE(self.M, num_bits=self.bit_width)
        g_bin_D = g_bin.repeat_interleave(self.interpolation, dim=1)
        z_enc_q = quant_global_STE(z_enc, num_bits=self.bit_width)

        if self.dropout_p > 0:
            z_enc_q = self.dropout(z_enc_q)

        logits = (z_enc_q * g_bin_D) @ M_q.T
        return logits, g_bin_D


class AdaptHDInference(nn.Module):
    """Inference-time AdaptHD using full-precision class memory and a
    batched masked matmul. Matches the trained AdaptHD numerics for eval."""

    def __init__(self, f, m, k, D, threshold=0.0, bit_width=8,
                 weight_bit_width=8, inp_bit_width=None, pool_bitwidth=None,
                 bias_linear=None, verbose=False, dtype=torch.float32):
        super().__init__()
        self.f = f
        self.m = m
        self.k = k
        self.bias_linear = bias_linear
        self.verbose = verbose
        self.threshold = threshold
        self.bit_width = bit_width
        self.weight_bit_width = weight_bit_width
        self.inp_bit_width = inp_bit_width
        self.pool_bitwidth = pool_bitwidth
        self.interpolation = int(D) // int(k)
        self.dtype = dtype

        self.M = nn.Parameter(torch.randn(m, D, dtype=self.dtype))
        self.adaptivepool = nn.AdaptiveMaxPool1d(k)
        self.lin_enc = SimpleLinear(
            f, D, bias=self.bias_linear,
            num_bits_inp=None, num_bits_weight=None, dtype=self.dtype,
        )
        self.bin_threshold = LearnableSTEAbsoluteThreshold(
            init_ratio=0.1, sharpness=1, dtype=self.dtype
        )

    def forward(self, x):
        bin_logits = self.adaptivepool(x)
        g_bin = self.bin_threshold(bin_logits)
        z_enc = self.lin_enc(x)

        g_bin_D = g_bin.repeat_interleave(self.interpolation, dim=1)
        z_enc_q = quant_global_STE(z_enc, num_bits=self.bit_width)

        M_scaled = self.M.unsqueeze(0) * g_bin_D.unsqueeze(1)
        logits = torch.bmm(M_scaled, z_enc_q.unsqueeze(2)).squeeze(2)
        return logits, g_bin_D
