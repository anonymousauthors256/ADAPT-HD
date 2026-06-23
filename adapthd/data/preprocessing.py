"""Feature normalization / standardization helpers."""
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


def standardize_x_set(all_x_train, x_test, val_x=[]):
    """Zero-mean / unit-variance scaling fit on train (sklearn StandardScaler)."""
    scaler = StandardScaler().fit(all_x_train)
    train_x = scaler.transform(all_x_train)
    test_x = scaler.transform(x_test)
    if len(val_x) != 0:
        val_x = scaler.transform(val_x)
        return train_x, val_x, test_x
    return train_x, test_x


def normalize_x_set(all_x_train, x_test, val_x=[]):
    """L2-normalize samples (sklearn Normalizer)."""
    scaler = preprocessing.Normalizer().fit(all_x_train)
    train_x, test_x = scaler.transform(all_x_train), scaler.transform(x_test)
    if len(val_x) != 0:
        val_x = scaler.transform(val_x)
        return train_x, val_x, test_x
    return train_x, test_x


def standardize(train_x, test_x, val_x=[]):
    """Manual mean/std standardization. Use for datasets whose features have
    different ranges (e.g. PAMAP2)."""
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-8

    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    if len(val_x) != 0:
        val_x = (val_x - mean) / std
        return train_x, val_x, test_x
    return train_x, test_x
