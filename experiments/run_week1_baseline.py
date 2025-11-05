# Week-1 baseline: single-process FedAvg with “satellite dropout”
# Saves CSVs + two PNG plots under ./results/

import os, csv, time
from typing import List, Tuple

# Use a headless backend so saving PNGs works in Codespaces/CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


# ===== Config =====
NUM_CLIENTS   = 8
ROUNDS        = 10
LOCAL_EPOCHS  = 1

DROPOUT_BASE  = 0.0   # baseline: everyone connects
DROPOUT_SAT   = 0.3   # satellite-like: intermittent contact

RANDOM_SEED   = 42    # single seed for full determinism

# Global seeding (NumPy + a dedicated RNG for dropout)
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)


# ===== Data (non-IID-ish split) =====
digits = load_digits()
# sklearn 1.6.x expects float64 in several paths; keep 64-bit throughout
X, y = shuffle(digits.data.astype(np.float64), digits.target, random_state=RANDOM_SEED)

# split by label then partition across clients (slight non-IID)
label_groups = [np.where(y == d)[0] for d in range(10)]
client_idx: List[np.ndarray] = [np.array([], dtype=int) for _ in range(NUM_CLIENTS)]
for idxs in label_groups:
    parts = np.array_split(idxs, NUM_CLIENTS)
    for c in range(NUM_CLIENTS):
        client_idx[c] = np.concatenate([client_idx[c], parts[c]])

clients_data: List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = []
for c in range(NUM_CLIENTS):
    ci = client_idx[c]
    Xc, yc = X[ci], y[ci]
    k = int(0.8 * len(Xc))
    clients_data.append(((Xc[:k], yc[:k]), (Xc[k:], yc[k:])))

# server-held test set (kept fixed)
Xt, yt = X[-300:], y[-300:]


# ===== Model helpers (float64 all the way) =====
def new_model() -> SGDClassifier:
    # online-style logistic regression to highlight FL dynamics
    return SGDClassifier(
        loss="log_loss",
        alpha=5e-4,
        max_iter=1,
        tol=None,
        random_state=0,   # classifier’s own RNG paths
    )

def warm_init_if_needed(clf: SGDClassifier) -> None:
    """Ensure classes_/coef_/intercept_ exist before reading params."""
    if not hasattr(clf, "coef_"):
        # tiny seed batch (one sample per class) from the test set
        X0, y0 = [], []
        for d in range(10):
            idx = int(np.where(yt == d)[0][0])
            X0.append(Xt[idx])
            y0.append(yt[idx])
        clf.partial_fit(np.stack(X0).astype(np.float64), np.array(y0), classes=np.arange(10))

def get_params(clf: SGDClassifier) -> List[np.ndarray]:
    warm_init_if_needed(clf)
    return [clf.coef_.astype(np.float64), clf.intercept_.astype(np.float64)]

def set_params(clf: SGDClassifier, params: List[np.ndarray]) -> SGDClassifier:
    clf.classes_       = np.arange(10)
    clf.coef_          = params[0].astype(np.float64)
    clf.intercept_     = params[1].astype(np.float64)
    clf.n_features_in_ = Xt.shape[1]
    return clf

def params_size_bytes(p: List[np.ndarray]) -> int:
    return sum(arr.nbytes for arr in p)


# ===== Single-process FedAvg simulator =====
def simulate(csv_path: str, dropout_prob: float, rounds: int = ROUNDS) -> str:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    # server/global model
    gclf    = new_model()
    gparams = get_params(gclf)
    one_client_down_bytes = params_size_bytes(gparams)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "bytes_up", "bytes_down", "server_acc", "active_clients"])

        for r in range(1, rounds + 1):
            t0 = time.time()

            client_params, sizes = [], []
            active, bytes_up = 0, 0

            for cid in range(NUM_CLIENTS):
                # satellite intermittency
                if rng.random() < dropout_prob:
                    continue
                active += 1

                (Xtr, ytr), _ = clients_data[cid]

                # start client from global
                clf = new_model()
                set_params(clf, gparams)

                # local training
                for _ in range(LOCAL_EPOCHS):
                    clf.partial_fit(Xtr.astype(np.float64), ytr, classes=np.arange(10))

                p = get_params(clf)
                client_params.append(p)
                sizes.append(len(Xtr))
                bytes_up += params_size_bytes(p)

            if active > 0:
                # FedAvg (size-weighted)
                wts = np.array(sizes, dtype=np.float64)
                wts /= wts.sum()
                agg_coef = np.average(np.stack([p[0] for p in client_params]), axis=0, weights=wts)
                agg_int  = np.average(np.stack([p[1] for p in client_params]), axis=0, weights=wts)
                gparams  = [agg_coef, agg_int]

            # evaluate on server test set
            eval_clf = new_model()
            set_params(eval_clf, gparams)
            y_pred = eval_clf.predict(Xt)
            acc = accuracy_score(yt, y_pred)

            w.writerow([
                r,
                int(bytes_up),
                int(one_client_down_bytes * active),  # approx. downlink bytes
                float(acc),
                active
            ])

    print(f"Wrote {csv_path}")
    return csv_path


# ===== Run two scenarios: baseline vs satellite-like dropout =====
b_csv = simulate("results/results_baseline.csv",  DROPOUT_BASE)
s_csv = simulate("results/results_satellite.csv", DROPOUT_SAT)

# ===== Plot / save artifacts =====
b = pd.read_csv(b_csv)
s = pd.read_csv(s_csv)

plt.figure(figsize=(6, 4))
plt.plot(b["round"], b["server_acc"], label="baseline")
plt.plot(s["round"], s["server_acc"], label=f"satellite (dropout={DROPOUT_SAT})")
plt.xlabel("Round"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Rounds")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("results/acc_vs_rounds.png", dpi=160)

plt.figure(figsize=(6, 4))
plt.plot(b["round"], b["bytes_up"], label="baseline uplink")
plt.plot(s["round"], s["bytes_up"], label="satellite uplink")
plt.xlabel("Round"); plt.ylabel("Bytes (approx)"); plt.title("Uplink Bytes vs Rounds")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("results/bytes_vs_rounds.png", dpi=160)

print(
    "Artifacts:\n"
    " - results/results_baseline.csv\n"
    " - results/results_satellite.csv\n"
    " - results/acc_vs_rounds.png\n"
    " - results/bytes_vs_rounds.png"
)
