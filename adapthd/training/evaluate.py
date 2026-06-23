"""Test-set evaluation loops, including sparsity statistics for AdaptHD."""
import torch


def test_model(model, X_test, y_test, dtype=torch.float32, device="cpu",
               batch_size=512):
    """Evaluate AdaptHD and print detailed mask-sparsity diagnostics.

    Returns the full per-sample mask tensor ``g_bin``.
    """
    X_test = torch.from_numpy(X_test).to(dtype=dtype)
    y_test = torch.from_numpy(y_test).long()

    model.to(device)
    model.eval()

    total_correct = 0
    total_samples = 0
    all_g_bin = []

    with torch.no_grad():
        for i in range(0, X_test.shape[0], batch_size):
            xb = X_test[i:i + batch_size].to(device)
            yb = y_test[i:i + batch_size].to(device)
            logits, g_bin = model(xb)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += yb.shape[0]
            all_g_bin.append(g_bin.cpu())

    g_bin = torch.cat(all_g_bin, dim=0)
    acc = total_correct / total_samples
    zeros_per_sample = (g_bin == 0).sum(dim=1)
    total_zeros = zeros_per_sample.sum().item()
    max_zeros = zeros_per_sample.max().item()
    min_zeros = zeros_per_sample.min().item()
    zero_ratio = (g_bin == 0).sum(dim=0) / g_bin.shape[0]

    print("Max diff", torch.max(g_bin[0] - g_bin[1]))
    print(zeros_per_sample.shape)
    print(f"col sums shape: {(g_bin == 0).sum(dim=0).shape}, "
          f"unique col sums: {torch.unique((g_bin == 0).sum(dim=0))}")
    print(zero_ratio[:20])
    print("Test Accuracy:", acc)
    print("Total zeros in g over test set:", total_zeros)
    print(f"Average Reduced Dimension: {total_zeros / total_samples}")
    print(f"Min zeros: {min_zeros}, Max zeros: {max_zeros}")
    return g_bin


def test_model_clean(model, X_test, y_test, dtype=torch.float32, device="cpu",
                     batch_size=512):
    """Quiet AdaptHD evaluation. Returns (accuracy, avg_zeros_per_sample)."""
    X_test = torch.from_numpy(X_test).to(dtype=dtype)
    y_test = torch.from_numpy(y_test).long()

    model.to(device)
    model.eval()

    total_correct = 0
    total_samples = 0
    all_g_bin = []

    with torch.no_grad():
        for i in range(0, X_test.shape[0], batch_size):
            xb = X_test[i:i + batch_size].to(device)
            yb = y_test[i:i + batch_size].to(device)
            logits, g_bin = model(xb)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += yb.shape[0]
            all_g_bin.append(g_bin.cpu())

    g_bin = torch.cat(all_g_bin, dim=0)
    acc = total_correct / total_samples
    avg_zeros = (g_bin == 0).sum(dim=1).sum().item() / total_samples
    return acc, avg_zeros


def test_model_bipolar(model, X_test, y_test, dtype=torch.float32, device="cpu",
                       batch_size=512):
    """Evaluate a logits-only bipolar model. Returns accuracy."""
    X_test = torch.from_numpy(X_test).to(dtype=dtype)
    y_test = torch.from_numpy(y_test).long()

    model.to(device)
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i in range(0, X_test.shape[0], batch_size):
            xb = X_test[i:i + batch_size].to(device)
            yb = y_test[i:i + batch_size].to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += yb.shape[0]

    acc = total_correct / total_samples
    print("Test Accuracy:", acc)
    return acc
