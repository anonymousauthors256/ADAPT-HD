"""Accuracy / evaluation utilities for AdaptHD and bipolar models."""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(y_train, device):
    """Balanced class weights tensor for imbalanced datasets."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def adapthd_predict(model, X, y, batch_size=256, device=None, return_probs=False):
    """Batched prediction for AdaptHD-style models returning (logits, mask)."""
    if device is None:
        device = next(model.parameters()).device

    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.numpy())
            if return_probs:
                all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    if return_probs:
        return y_true, y_pred, np.concatenate(all_probs)
    return y_true, y_pred


def accuracy(model, X, y, batch_size=512):
    """Accuracy for models that return (logits, g_bin)."""
    model.eval()
    device = next(model.parameters()).device
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i + batch_size].to(device)
            yb = y[i:i + batch_size].to(device)
            logits, _ = model(xb)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += yb.shape[0]
    return total_correct / total_samples


def accuracy_bipolar(model, X, y, batch_size=512):
    """Accuracy for models that return logits only (e.g. BipolarHDC)."""
    model.eval()
    device = next(model.parameters()).device
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i + batch_size].to(device)
            yb = y[i:i + batch_size].to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += yb.shape[0]
    return total_correct / total_samples


def per_class_accuracy(model, X, y, batch_size=512):
    """Print and return per-class accuracy from the confusion matrix."""
    y_true, y_pred = adapthd_predict(model, X, y, batch_size=batch_size)
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    for cls, acc in enumerate(class_acc):
        print(f"Class {cls}: {acc:.4f} ({cm[cls, cls]}/{cm.sum(axis=1)[cls]})")
    return class_acc
