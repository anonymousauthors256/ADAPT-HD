"""AdaptHD: adaptive, quantized hyperdimensional computing classifiers.

Public API re-exports the most commonly used pieces.
"""
from .config import AdaptHDConfig, get_config, list_datasets
from .models import (
    AdaptHD,
    BipolarHDC,
    AdaptHDInference,
    SimpleLinear,
    LearnableSTEAbsoluteThreshold,
    quant_global_STE,
    quant_rowwise_STE,
)
from .training import (
    train_model_switching_loss,
    train_pure_bipolar,
    test_model,
    test_model_clean,
    test_model_bipolar,
)
from .utils import (
    accuracy,
    accuracy_bipolar,
    per_class_accuracy,
    adapthd_predict,
    get_class_weights,
)
from .data import standardize, standardize_x_set, normalize_x_set
from .io import (
    save_bipolar_model,
    load_bipolar_model,
    convert_bpt_to_bptx,
    save_testset,
)

__version__ = "0.1.0"

__all__ = [
    "AdaptHDConfig", "get_config", "list_datasets",
    "AdaptHD", "BipolarHDC", "AdaptHDInference",
    "SimpleLinear", "LearnableSTEAbsoluteThreshold",
    "quant_global_STE", "quant_rowwise_STE",
    "train_model_switching_loss", "train_pure_bipolar",
    "test_model", "test_model_clean", "test_model_bipolar",
    "accuracy", "accuracy_bipolar", "per_class_accuracy",
    "adapthd_predict", "get_class_weights",
    "standardize", "standardize_x_set", "normalize_x_set",
    "save_bipolar_model", "load_bipolar_model",
    "convert_bpt_to_bptx", "save_testset",
]
