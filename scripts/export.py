"""Export a trained AdaptHD checkpoint to bit-packed .bpt and aligned .bptx.

Example:
    python -m scripts.export --ckpt runs/isolet_2000/isolet_D2000.pt \
        --out runs/isolet_2000/model
"""
import argparse

import torch

from adapthd import AdaptHD, save_bipolar_model, convert_bpt_to_bptx
from adapthd.models import quant_global_STE


def parse_args():
    p = argparse.ArgumentParser(description="Export AdaptHD to .bpt/.bptx")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True, help="output path stem (no extension)")
    p.add_argument(
        "--packed-params", nargs="*", default=["M"],
        help="state_dict param names to bipolarize + bit-pack (default: M)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]

    # Reconstruct model shape from the saved config + weight shapes.
    M_shape = ckpt["state_dict"]["M"].shape
    enc_shape = ckpt["state_dict"]["lin_enc.weight"].shape
    m, D = int(M_shape[0]), int(M_shape[1])
    f = int(enc_shape[1])

    model = AdaptHD(
        f=f, m=m, k=cfg["k"], D=D,
        threshold=cfg["mask_threshold"],
        bit_width=cfg["bit_width"],
        weight_bit_width=cfg["weight_bit_width"],
        inp_bit_width=cfg["inp_bit_width"],
        bias_linear=cfg["bias_linear"],
        dropout_p=cfg["dropout_p"],
    )
    model.load_state_dict(ckpt["state_dict"])

    bpt_path = args.out + ".bpt"
    bptx_path = args.out + ".bptx"

    save_bipolar_model(model, set(args.packed_params), bpt_path, quant_global_STE)
    print(f"Wrote {bpt_path}")
    convert_bpt_to_bptx(bpt_path, bptx_path)


if __name__ == "__main__":
    main()
