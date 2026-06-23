"""Load and resolve AdaptHD hyperparameter configs from YAML."""
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "configs", "configs.yaml")

_STR_TO_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "float64": torch.float64,
}


@dataclass
class AdaptHDConfig:
    """Resolved hyperparameters for a single (dataset, D) preset."""
    dataset: str
    D: int
    k: int
    lr: float
    epochs: int
    acc_th: float
    mask_threshold: float
    batch_size: int
    lambda_norm: float
    # shared / defaulted
    bit_width: int = 8
    weight_bit_width: int = 8
    inp_bit_width: Optional[int] = None
    bias_linear: Optional[bool] = None
    dropout_p: float = 0.0
    dtype: str = "float32"

    @property
    def torch_dtype(self):
        return _STR_TO_DTYPE[self.dtype]

    def to_dict(self):
        return asdict(self)


def _load_yaml(path=None):
    path = path or _DEFAULT_CONFIG
    with open(path, "r") as f:
        return yaml.safe_load(f)


def list_datasets(path=None):
    """Return the available dataset keys."""
    return list(_load_yaml(path)["datasets"].keys())


def get_config(dataset, D=None, index=None, path=None):
    """Resolve a config for ``dataset``.

    Select a preset either by ``D`` (e.g. 2000 or 10000) or by ``index``
    (0 = compact, 1 = large). Defaults are merged underneath the preset.
    """
    raw = _load_yaml(path)
    defaults = raw.get("defaults", {})
    datasets = raw["datasets"]

    if dataset not in datasets:
        raise KeyError(
            f"Unknown dataset '{dataset}'. Available: {list(datasets.keys())}"
        )

    presets = datasets[dataset]

    if index is not None:
        preset = presets[index]
    elif D is not None:
        matches = [p for p in presets if p["D"] == D]
        if not matches:
            avail = [p["D"] for p in presets]
            raise ValueError(f"No preset with D={D} for '{dataset}'. Available D: {avail}")
        preset = matches[0]
    else:
        preset = presets[0]

    merged = {**defaults, **preset}
    return AdaptHDConfig(dataset=dataset, **merged)
