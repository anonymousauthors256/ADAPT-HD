"""Training loops for AdaptHD (switching-loss sparsity) and pure bipolar models."""
import copy
import torch
import torch.nn as nn

from ..utils.metrics import accuracy, accuracy_bipolar


def train_pure_bipolar(model, X_train, y_train, X_val, y_val, epochs=20,
                       batch_size=64, lr=1e-3, device="cuda",
                       dtype=torch.float32, verbose=True):
    """Plain cross-entropy training for BipolarHDC."""
    X_train = torch.from_numpy(X_train).to(dtype=dtype)
    y_train = torch.from_numpy(y_train).long()
    X_val = torch.from_numpy(X_val).to(dtype=dtype)
    y_val = torch.from_numpy(y_val).long()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    N = X_train.shape[0]
    val_acc = 0
    best_val_acc = 0
    best_state_dict = None
    val_acc_array = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

        epoch_loss /= N
        val_acc = accuracy_bipolar(model, X_val, y_val)
        train_acc = accuracy_bipolar(model, X_train, y_train)
        val_acc_array.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())

        if verbose and (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Train Acc: {train_acc:.4f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return val_acc_array


def train_model_switching_loss(model, X_train, y_train, X_val, y_val,
                               epochs=20, batch_size=64, lr=1e-3, acc_th=0.92,
                               lambda_norm=0.01, lambda_acc=2, device="cuda",
                               dtype=torch.float32, verbose=True,
                               class_weights=None, th_lr=None):
    """AdaptHD training with a two-phase switching loss.

    Until validation accuracy crosses ``acc_th`` the loss is plain
    cross-entropy. Past the threshold, an L2 sparsity penalty on the mask
    ``g`` (weighted by ``lambda_norm``) is added to encourage dimension
    reduction without sacrificing accuracy.
    """
    X_train = torch.from_numpy(X_train).to(dtype=dtype)
    y_train = torch.from_numpy(y_train).long()
    X_val = torch.from_numpy(X_val).to(dtype=dtype)
    y_val = torch.from_numpy(y_val).long()

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    if th_lr:
        optimizer = torch.optim.Adam([
            {"params": [model.bin_threshold.logit_ratio], "lr": th_lr},
            {"params": [p for p in model.parameters()
                        if p is not model.bin_threshold.logit_ratio], "lr": lr},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    N = X_train.shape[0]
    val_acc = 0
    best_val_acc = 0
    best_state_dict = None
    val_acc_array = []
    sparsity_train = []

    for epoch in range(epochs):
        sparsity_count = 0
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)

            optimizer.zero_grad()
            logits, g = model(xb)
            sparsity_count += torch.sum(g == 0).item()

            ce_loss = criterion(logits, yb)
            norm_loss = torch.norm(g, dim=1).mean()
            loss1 = ce_loss + lambda_norm * norm_loss
            loss2 = ce_loss

            if val_acc > acc_th:
                loss1.backward()
                epoch_loss += loss1.item()
            else:
                loss2.backward()
                epoch_loss += loss2.item()

            optimizer.step()

        epoch_loss /= N
        val_acc = accuracy(model, X_val, y_val)
        train_acc = accuracy(model, X_train, y_train)
        val_acc_array.append(val_acc)
        sparsity_train.append(sparsity_count / N)

        current_ratios = None
        if hasattr(model, "bin_threshold"):
            current_ratios = model.bin_threshold.logit_ratio.detach().clone()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())

        if verbose and (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Avg. Sparsity: {sparsity_count/N:.4f} | "
                  f"Bin Ratio: {current_ratios}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return val_acc_array, sparsity_train
