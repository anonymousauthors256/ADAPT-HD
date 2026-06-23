# AdaptHD

Adaptive, quantized **Hyperdimensional Computing (HDC)** classifiers with a
**learnable per-sample sparsity mask**. AdaptHD encodes inputs into
high-dimensional bipolar hypervectors, then learns a data-dependent gating mask
that switches off blocks of dimensions per sample — reducing the effective
dimensionality at inference while preserving accuracy. Weights, encodings, and
the class memory are quantized end-to-end using a straight-through estimator
(STE), and trained models can be exported to a compact, bit-packed binary format
suitable for embedded inference.

## Highlights

- **AdaptHD model** — quantized HDC encoder + class memory with a learnable
  `AdaptiveMaxPool1d` → threshold mask that gates blocks of the hypervector.
- **Switching-loss training** — cross-entropy until validation accuracy crosses
  a target, then a sparsity penalty on the mask kicks in to reduce dimensions
  without sacrificing accuracy.
- **STE quantization** — global and row-wise symmetric quantization for 1-bit
  (bipolar), 2-bit (4-level), and n-bit weights/activations.
- **Bipolar baseline** — `BipolarHDC` for a plain quantized HDC classifier.
- **Compact serialization** — bit-pack bipolar parameters to `.bpt` (8× smaller),
  and repack to a 64-byte-aligned, mmap-friendly `.bptx` for runtimes.
- **Config-driven** — per-dataset, per-dimension hyperparameter presets in YAML.

## Installation

```bash
git clone <your-repo-url> adapthd
cd adapthd
pip install -e .
```

Requires Python ≥ 3.9, PyTorch, scikit-learn, scipy, and PyYAML.

## Repository layout

```
adapthd/
├── adapthd/
│   ├── config.py              # YAML config loader → AdaptHDConfig
│   ├── models/
│   │   ├── quant.py           # quant_global_STE, quant_rowwise_STE
│   │   ├── layers.py          # SimpleLinear, LearnableSTEAbsoluteThreshold
│   │   └── adapthd.py         # AdaptHD, BipolarHDC, AdaptHDInference
│   ├── training/
│   │   ├── train.py           # train_model_switching_loss, train_pure_bipolar
│   │   └── evaluate.py        # test_model, test_model_clean, test_model_bipolar
│   ├── utils/
│   │   └── metrics.py         # accuracy, per-class accuracy, class weights
│   ├── data/
│   │   └── preprocessing.py   # standardize / normalize helpers
│   └── io/
│       ├── serialization.py   # save/load bit-packed .bpt
│       └── bptx.py            # .bpt → .bptx converter, save_testset
├── configs/
│   └── configs.yaml           # dataset hyperparameter presets
├── scripts/
│   ├── train.py               # CLI: train from a config preset
│   └── export.py              # CLI: export checkpoint → .bpt/.bptx
├── pyproject.toml
└── requirements.txt
```

## Quick start (library)

```python
import numpy as np
import adapthd as a

# 1. Resolve hyperparameters for a dataset / dimension preset
cfg = a.get_config("isolet", D=2000)        # or index=0 (compact) / index=1 (large)

# 2. Build the model (f = #features, m = #classes)
model = a.AdaptHD(
    f=617, m=26, k=cfg.k, D=cfg.D,
    threshold=cfg.mask_threshold,
    dtype=cfg.torch_dtype,
)

# 3. Train with the switching loss (X_* are float32 numpy arrays, y_* int)
val_acc, sparsity = a.train_model_switching_loss(
    model, X_train, y_train, X_val, y_val,
    epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr,
    acc_th=cfg.acc_th, lambda_norm=cfg.lambda_norm, device="cuda",
)

# 4. Evaluate (returns accuracy and avg. zeroed dimensions per sample)
acc, avg_zeros = a.test_model_clean(model, X_test, y_test, device="cuda")
print(acc, avg_zeros)
```

## Quick start (CLI)

The training script expects a data directory containing NumPy arrays
`X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`,
`y_test.npy`. Adapt `load_arrays` in `scripts/train.py` to plug in your own
data pipeline.

```bash
# Train the compact ISOLET model
python -m scripts.train --dataset isolet --D 2000 \
    --data-dir ./data/isolet --device cuda --out runs/isolet_2000

# PAMAP2 has differently-scaled features — apply standardization
python -m scripts.train --dataset pamap2 --index 1 \
    --data-dir ./data/pamap2 --standardize --class-weights \
    --device cuda --out runs/pamap2_large

# Export a trained checkpoint to bit-packed .bpt and aligned .bptx
python -m scripts.export --ckpt runs/isolet_2000/isolet_D2000.pt \
    --out runs/isolet_2000/model
```

## How AdaptHD works

1. **Encoding.** A quantized linear layer (`SimpleLinear`) maps the `f`-dim input
   to a `D`-dim hypervector, which is quantized to bipolar via STE.
2. **Mask generation.** `AdaptiveMaxPool1d(k)` pools the input into `k` block
   logits; a `LearnableSTEAbsoluteThreshold` turns them into a hard 0/1 mask with
   a sigmoid surrogate gradient. The mask is interpolated back to `D` so each of
   the `k` blocks gates `D/k` dimensions.
3. **Classification.** The masked, quantized encoding is matched against a
   quantized class memory `M` (`m × D`) by dot product to produce logits.
4. **Training.** `train_model_switching_loss` runs plain cross-entropy until
   validation accuracy exceeds `acc_th`, then adds `lambda_norm · ‖mask‖₂` to
   push the mask toward sparsity — trimming dimensions only once the model is
   already accurate.

## Configuration

Presets live in `configs/configs.yaml`. Each dataset provides a compact
(`D = 2000`) and a large (`D = 10000`) variant. Selected fields:

| Field            | Meaning                                                        |
|------------------|----------------------------------------------------------------|
| `D`              | Hypervector dimensionality                                     |
| `k`              | Number of pooled mask blocks (`AdaptiveMaxPool1d` size)        |
| `lr`             | Learning rate                                                  |
| `epochs`         | Training epochs                                                |
| `acc_th`         | Val accuracy at which the sparsity penalty activates           |
| `mask_threshold` | Initial threshold (`init_ratio`) of the learnable mask         |
| `batch_size`     | Minibatch size                                                 |
| `lambda_norm`    | Weight of the L2 mask-sparsity penalty                         |

### Included presets

| Dataset             | D = 2000 (lr / epochs / batch / λ / thr)     | D = 10000 (lr / epochs / batch / λ / thr)    |
|---------------------|----------------------------------------------|----------------------------------------------|
| `isolet`            | 1e-3 / 300 / 64 / 0.05 / 0.01                | 2e-3 / 200 / 64 / 0.05 / 0.01                |
| `emg_hand_gestures` | 2e-3 / 200 / 64 / 0.05 / 0.01                | 1e-3 / 275 / 64 / 0.05 / 0.01                |
| `mnist`             | 2e-3 / 300 / 64 / 0.05 / 0.01                | 2e-3 / 200 / 64 / 0.05 / 0.01                |
| `fashion_mnist`     | 1e-3 / 200 / 128 / 0.05 / 0.01               | 1e-3 / 200 / 128 / 0.05 / 0.01               |
| `ucihar`            | 2e-3 / 300 / 64 / 0.01 / 0.005               | 2e-3 / 200 / 64 / 0.01 / 0.005               |
| `pamap2`            | 1e-3 / 70 / 1024 / 0.05 / 0.01               | 1e-3 / 70 / 1024 / 0.05 / 0.01               |

All presets use `acc_th = 0.93`. PAMAP2 features have differing ranges — use the
`--standardize` flag (or `adapthd.data.standardize`) when training it.

Load a preset programmatically:

```python
from adapthd import get_config, list_datasets

print(list_datasets())
cfg = get_config("ucihar", D=10000)   # by dimension
cfg = get_config("ucihar", index=0)   # or by index (0 compact, 1 large)
```

## Serialization formats

- **`.bpt`** — a simple length-prefixed format. Parameters named in
  `packed_param_names` are bipolarized to `{-1, +1}` and bit-packed (8×
  reduction); all others are stored as raw typed bytes alongside their dtype.
- **`.bptx`** — repacks a `.bpt` into a fixed 256-byte metadata slot per tensor
  plus 64-byte-aligned data blobs, so an inference runtime can memory-map and
  index tensors directly.

```python
from adapthd import save_bipolar_model, load_bipolar_model, convert_bpt_to_bptx
from adapthd.models import quant_global_STE

save_bipolar_model(model, {"M"}, "model.bpt", quant_global_STE)
load_bipolar_model(model, "model.bpt")
convert_bpt_to_bptx("model.bpt", "model.bptx")
```

## License

MIT — see [LICENSE](LICENSE).
