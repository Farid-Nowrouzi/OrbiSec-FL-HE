import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

def set_global_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def binary_accuracy(preds, labels):
    preds = (preds > 0.5).float()
    return (preds == labels).float().mean().item()

def estimate_bytes(tensor_dict):
    total = 0
    for _, v in tensor_dict.items():
        total += v.nelement() * v.element_size()
    return total

def append_to_csv(path, row_dict):
    df = pd.DataFrame([row_dict])
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", index=False, header=False)

def plot_metrics(csv_path, save_dir="results"):
    import matplotlib
    matplotlib.use("Agg")  # headless

    df = pd.read_csv(csv_path)

    # Accuracy vs rounds
    plt.figure(figsize=(6, 4))
    plt.plot(df["round"], df["acc"], marker="o")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Rounds")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "acc_vs_rounds.png"))
    plt.close()

    # Bytes vs rounds
    plt.figure(figsize=(6, 4))
    plt.plot(df["round"], df["bytes_up"], label="uplink")
    plt.plot(df["round"], df["bytes_down"], label="downlink")
    plt.xlabel("Round")
    plt.ylabel("Bytes")
    plt.legend()
    plt.title("Bytes vs Rounds")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "bytes_vs_rounds.png"))
    plt.close()
