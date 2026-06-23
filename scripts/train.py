"""Train an AdaptHD model from a dataset config preset.

Example:
    python -m scripts.train --dataset isolet --D 2000 \
        --data-dir ./data --device cuda --out runs/isolet_2000

Expects a data directory containing NumPy arrays:
    X_train.npy y_train.npy X_val.npy y_val.npy X_test.npy y_test.npy
(adapt :func:`load_arrays` to your own data pipeline as needed).
"""
import os
import argparse

import numpy as np
import torch

from adapthd import (
    get_config,
    list_datasets,
    AdaptHD,
    train_model_switching_loss,
    test_model_clean,
    standardize,
    get_class_weights,
)


def load_arrays(data_dir):
    def _l(name):
        return np.load(os.path.join(data_dir, name))
    return (
        _l("X_train.npy"), _l("y_train.npy"),
        _l("X_val.npy"), _l("y_val.npy"),
        _l("X_test.npy"), _l("y_test.npy"),
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train AdaptHD")
    p.add_argument("--dataset", required=True,
                   help=f"one of: {', '.join(list_datasets())}")
    p.add_argument("--D", type=int, default=None,
                   help="hypervector dim preset (e.g. 2000 or 10000)")
    p.add_argument("--index", type=int, default=None,
                   help="preset index (0=compact, 1=large); overrides --D")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="output dir for checkpoint")
    p.add_argument("--standardize", action="store_true",
                   help="apply manual mean/std standardization (e.g. PAMAP2)")
    p.add_argument("--class-weights", action="store_true",
                   help="use balanced class weights (imbalanced datasets)")
    p.add_argument("--config", default=None, help="path to a custom configs.yaml")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.dataset, D=args.D, index=args.index, path=args.config)
    print("Resolved config:", cfg.to_dict())

    X_train, y_train, X_val, y_val, X_test, y_test = load_arrays(args.data_dir)

    if args.standardize:
        X_train, X_val, X_test = standardize(X_train, X_test, X_val)

    f = X_train.shape[1]
    m = int(np.max(y_train)) + 1
    print(f"Features f={f}, classes m={m}")

    model = AdaptHD(
        f=f, m=m, k=cfg.k, D=cfg.D,
        threshold=cfg.mask_threshold,
        bit_width=cfg.bit_width,
        weight_bit_width=cfg.weight_bit_width,
        inp_bit_width=cfg.inp_bit_width,
        bias_linear=cfg.bias_linear,
        dropout_p=cfg.dropout_p,
        dtype=cfg.torch_dtype,
    )

    class_weights = None
    if args.class_weights:
        class_weights = get_class_weights(y_train, args.device)

    val_acc_array, sparsity_train = train_model_switching_loss(
        model, X_train, y_train, X_val, y_val,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        acc_th=cfg.acc_th,
        lambda_norm=cfg.lambda_norm,
        device=args.device,
        dtype=cfg.torch_dtype,
        verbose=not args.quiet,
        class_weights=class_weights,
    )

    acc, avg_zeros = test_model_clean(
        model, X_test, y_test, dtype=cfg.torch_dtype, device=args.device
    )
    print(f"\nTest accuracy: {acc:.4f}")
    print(f"Avg. zeroed dimensions / sample: {avg_zeros:.1f} of {cfg.D} "
          f"({100*avg_zeros/cfg.D:.1f}% reduction)")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        ckpt = os.path.join(args.out, f"{args.dataset}_D{cfg.D}.pt")
        torch.save({"state_dict": model.state_dict(),
                    "config": cfg.to_dict(),
                    "test_acc": acc}, ckpt)
        print(f"Saved checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
