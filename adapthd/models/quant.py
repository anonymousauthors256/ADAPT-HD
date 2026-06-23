"""Quantization functions with Straight-Through Estimator (STE)."""
import torch


def quant_global_STE(W, num_bits=8):
    """Global symmetric quantization with STE gradient passthrough.

    Supports 1-bit (sign/bipolar), 2-bit (ternary-ish 4-level), and
    n-bit (>=3) uniform symmetric quantization.
    """
    qmax = 2 ** (num_bits - 1) - 1
    normalizer = 2 ** (num_bits - 1)
    W_det = W.detach()
    max_abs = torch.clamp(W_det.abs().max(), min=1e-12)

    if num_bits == 1:
        W_q = torch.sign(W_det)
        W_q = W_q + (W - W_det)  # STE
        return W_q

    if num_bits == 2:
        W_norm = W_det / max_abs
        levels = torch.tensor(
            [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0],
            device=W.device,
            dtype=W.dtype,
        )
        W_flat = W_norm.view(-1, 1)
        distances = torch.abs(W_flat - levels.view(1, -1))
        indices = torch.argmin(distances, dim=1)
        W_q = levels[indices].view_as(W_norm)
        W_q = W_q * max_abs
        W_q = W_q + (W - W_det)  # STE
        return W_q

    W_q = torch.round(W_det / max_abs * qmax).clamp(-normalizer, qmax) / normalizer
    W_q = W_q + (W - W_det)  # STE
    return W_q


def quant_rowwise_STE(W, num_bits=8):
    """Row-wise symmetric quantization with STE gradient passthrough."""
    qmax = 2 ** (num_bits - 1) - 1
    normalizer = 2 ** (num_bits - 1)

    W_det = W.detach()
    row_max_abs = W_det.abs().max(dim=1, keepdim=True)[0]
    row_max_abs = torch.clamp(row_max_abs, min=1e-12)

    W_normalized = W_det / row_max_abs * qmax
    W_quantized_int = torch.round(W_normalized).clamp(-normalizer, qmax)
    W_q = W_quantized_int / normalizer
    W_q = W_q + (W - W_det)  # STE
    return W_q
