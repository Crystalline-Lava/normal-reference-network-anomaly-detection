# Adapted in part from EURECOM USAD, BSD 3-Clause License.
# Source and full upstream notice: ../../THIRD_PARTY_NOTICES.md

"""USAD baseline for the CIC-UNSW-NB15 dataset."""

from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import mstats
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass


# =============================================================================
# Configuration
# =============================================================================

# Fixed five-seed set reported in the manuscript.
RUN_SEEDS = (42, 456, 7, 789, 1024)
SEED = RUN_SEEDS[0]  # Active model seed; reassigned by run_experiment().
STRICT_DETERMINISM = False
FIX_TRAIN_SHUFFLE_ORDER = False

LABEL_TO_ID = {
    "Benign": 0,
    "Analysis": 1,
    "Backdoor": 2,
    "DoS": 3,
    "Exploits": 4,
    "Fuzzers": 5,
    "Generic": 6,
    "Reconnaissance": 7,
    "Shellcode": 8,
    "Worms": 9,
}
CIC_LABEL_MAP = {v: k for k, v in LABEL_TO_ID.items()}
HOLDOUT_FAMILIES = [8, 9]

# Keep Google Drive paths. These are intentionally not local paths.
BASE_DIR = os.getenv("CIC_BASE_DIR", "/content/drive/MyDrive/15SOTA")
RAW_CSV = os.getenv("CIC_UNSW_RAW_CSV", "/content/drive/MyDrive/CIC-UNSW-NB15/CICFlowMeter_out.csv")
OUTPUT_ROOT = os.path.join(BASE_DIR, "usad_paper_baseline")

TRAIN_CUTOFF = pd.Timestamp("2015-01-23")
DROP_COLS = [
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port",
    "Protocol", "Timestamp", "Label",
]
ENTITY_COLS = ["Src IP"]
WINDOW_SIZE = 50
TRAIN_STRIDE = 25
TEST_STRIDE = 10
MAX_TRAIN_WINDOWS = 150_000

BATCH_SIZE = 256
EPOCHS = 180
USAD_ALPHA = 0.5
USAD_BETA = 0.5
USAD_ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Readout comparison
# The same per-feature residual tensor is read three ways:
#   raw  = unweighted mean over features (reported USAD score)
#   std  = ÷ benign-val per-feature variance, then mean  — calibrated
#   maha = benign-val covariance Mahalanobis             — calibrated
# Selection is val-AP only (epochs >= MIN_SELECT_EPOCH), one best epoch PER readout.
MIN_SELECT_EPOCH = 20     # match the main pipeline's selection warmup
TPR_BUDGET = 0.01         # protocol: TPR @ 1% budget, threshold from benign val only
READOUTS = ["raw", "std", "maha"]


def set_seed(seed=None):
    if seed is None:
        seed = SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if STRICT_DETERMINISM:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# =============================================================================
# Shared Dataset Loader
# =============================================================================

def build_windows(X, y, entity_ids, window_size, stride):
    windows, family_labels = [], []
    for eid in np.unique(entity_ids):
        mask = entity_ids == eid
        X_e = X[mask]
        y_e = y[mask]
        for start in range(0, len(X_e) - window_size + 1, stride):
            win = X_e[start:start + window_size]
            windows.append(win)
            family_labels.append(int(y_e[start:start + window_size].max()))

    if not windows:
        return (
            np.zeros((0, window_size, X.shape[1]), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )

    windows = np.stack(windows).astype(np.float32)
    family_labels = np.array(family_labels, dtype=np.int64)
    attack_labels = (family_labels != 0).astype(np.int64)
    return windows, attack_labels, family_labels


def load_and_preprocess():
    """Load CICFlowMeter CSV from Google Drive and build shared windows."""
    set_seed()

    print("Loading raw CICFlowMeter CSV ...")
    raw = pd.read_csv(RAW_CSV)
    print(f"  Raw shape: {raw.shape}")

    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], dayfirst=True)
    raw.sort_values("Timestamp", inplace=True)
    raw.reset_index(drop=True, inplace=True)

    label_str = raw["Label"].str.strip()
    y_all = label_str.map(LABEL_TO_ID).values.astype(int)
    print(f"  Timestamp range: {raw['Timestamp'].min()} -> {raw['Timestamp'].max()}")
    print(f"  Label distribution:\n{label_str.value_counts().to_string()}")

    entity_str = raw[ENTITY_COLS].astype(str).agg("_".join, axis=1)
    entity_ids = pd.Categorical(entity_str).codes
    print(f"  Entities ({', '.join(ENTITY_COLS)}): {len(np.unique(entity_ids))} unique")

    data = raw.drop(columns=[c for c in DROP_COLS if c in raw.columns])
    for col in ENTITY_COLS:
        if col in data.columns:
            data = data.drop(columns=[col])

    var = data.var(numeric_only=True)
    low_var = var[var < 1e-10].index.tolist()
    data = data.drop(columns=low_var)
    print(f"  Removed {len(low_var)} near-constant features -> {data.shape[1]} remain")

    X_all = data.values.astype(np.float64)
    X_all[~np.isfinite(X_all)] = np.nan
    X_all = SimpleImputer(strategy="median").fit_transform(X_all)
    for j in range(X_all.shape[1]):
        X_all[:, j] = mstats.winsorize(X_all[:, j], limits=[0.001, 0.001])

    ts = raw["Timestamp"].values
    train_period = ts < np.datetime64(TRAIN_CUTOFF)
    test_period = ~train_period
    benign = y_all == 0

    train_indices, val_indices = [], []
    for eid in np.unique(entity_ids[train_period & benign]):
        eid_idx = np.where(train_period & benign & (entity_ids == eid))[0]
        split = int(len(eid_idx) * 0.85)
        train_indices.extend(eid_idx[:split].tolist())
        val_indices.extend(eid_idx[split:].tolist())

    train_indices = np.array(train_indices, dtype=int)
    val_indices = np.array(val_indices, dtype=int)
    test_indices = np.where(test_period)[0]

    print(f"\n  Train period: {len(train_indices)} benign flows (85%)")
    print(f"  Val period:   {len(val_indices)} benign flows (15%)")
    print(f"  Test period:  {len(test_indices)} all flows")

    scaler = MinMaxScaler()
    scaler.fit(X_all[train_indices])
    X_scaled = scaler.transform(X_all)
    X_scaled = np.clip(X_scaled, 0.0, 1.0).astype(np.float32)

    print(f"\n  Building windows (W={WINDOW_SIZE}) ...")
    X_train_w, _, _ = build_windows(
        X_scaled[train_indices],
        y_all[train_indices],
        entity_ids[train_indices],
        WINDOW_SIZE,
        TRAIN_STRIDE,
    )
    X_val_w, _, _ = build_windows(
        X_scaled[val_indices],
        y_all[val_indices],
        entity_ids[val_indices],
        WINDOW_SIZE,
        TRAIN_STRIDE,
    )
    X_test_w, y_test_attack_w, y_test_family_w = build_windows(
        X_scaled[test_indices],
        y_all[test_indices],
        entity_ids[test_indices],
        WINDOW_SIZE,
        TEST_STRIDE,
    )

    if len(X_train_w) > MAX_TRAIN_WINDOWS:
        idx = np.random.choice(len(X_train_w), MAX_TRAIN_WINDOWS, replace=False)
        X_train_w = X_train_w[idx]
        print(f"  Subsampled train windows to {MAX_TRAIN_WINDOWS}")

    holdout_w = np.isin(y_test_family_w, HOLDOUT_FAMILIES)
    known_attack = (y_test_attack_w == 1) & ~holdout_w
    known_attack_idx = np.where(known_attack)[0]
    val_attack_idx = np.zeros(0, dtype=int)

    if len(known_attack_idx) > 0:
        try:
            val_attack_idx, _ = train_test_split(
                known_attack_idx,
                test_size=0.90,
                random_state=SEED,
                stratify=y_test_family_w[known_attack_idx],
            )
        except ValueError:
            n_val_attack = max(1, len(known_attack_idx) // 10)
            val_attack_idx = np.random.choice(
                known_attack_idx, n_val_attack, replace=False)
        X_val_mixed_w = np.concatenate([X_val_w, X_test_w[val_attack_idx]])
        y_val_mixed_w = np.concatenate([
            np.zeros(len(X_val_w), dtype=int),
            np.ones(len(val_attack_idx), dtype=int),
        ])
        test_keep = np.ones(len(X_test_w), dtype=bool)
        test_keep[val_attack_idx] = False
        X_test_w = X_test_w[test_keep]
        y_test_attack_w = y_test_attack_w[test_keep]
        y_test_family_w = y_test_family_w[test_keep]
    else:
        X_val_mixed_w = X_val_w.copy()
        y_val_mixed_w = np.zeros(len(X_val_w), dtype=int)

    print("\n  Windows built:")
    print(f"    Train:     {X_train_w.shape}")
    print(f"    Val:       {X_val_w.shape}")
    print(f"    Val mixed: {X_val_mixed_w.shape} (attack rate={y_val_mixed_w.mean():.4f})")
    print(f"      Known-attack val windows removed from test: {len(val_attack_idx)}")
    print(f"    Test:      {X_test_w.shape} (attack rate={y_test_attack_w.mean():.4f})")

    return (
        X_train_w,
        X_val_w,
        X_val_mixed_w,
        y_val_mixed_w,
        X_test_w,
        y_test_attack_w,
        y_test_family_w,
    )


# =============================================================================
# USAD Paper Baseline
# =============================================================================

class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size / 2))
        self.linear2 = nn.Linear(int(in_size / 2), int(in_size / 4))
        self.linear3 = nn.Linear(int(in_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return self.relu(out)


class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size / 4))
        self.linear2 = nn.Linear(int(out_size / 4), int(out_size / 2))
        self.linear3 = nn.Linear(int(out_size / 2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return self.sigmoid(out)


class UsadModel(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)

    def training_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + \
            (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - \
            (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        return loss1, loss2

    @torch.no_grad()
    def validation_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + \
            (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - \
            (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        return {"val_loss1": loss1, "val_loss2": loss2}


def make_usad_loader(X, batch_size, shuffle=False, drop_last=False):
    X_flat = X.reshape(len(X), -1).astype(np.float32, copy=False)
    dataset = TensorDataset(torch.tensor(X_flat, dtype=torch.float32))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
    }
    if shuffle and FIX_TRAIN_SHUFFLE_ORDER:
        generator = torch.Generator()
        generator.manual_seed(SEED)
        kwargs["generator"] = generator
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def evaluate_usad(model, val_loader, n, device):
    model.eval()
    outputs = []
    for (batch,) in val_loader:
        outputs.append(model.validation_step(batch.to(device), n))
    return {
        "val_loss1": torch.stack([x["val_loss1"] for x in outputs]).mean().item(),
        "val_loss2": torch.stack([x["val_loss2"] for x in outputs]).mean().item(),
    }


def train_usad(epochs, model, train_loader, val_loader, device,
               X_val_w, X_val_mixed_w, y_val_mixed_w, X_test_w, y_test_attack_w,
               opt_func=torch.optim.Adam):
    """Adversarial USAD training with per-epoch readout tracking.

    Each epoch, ONE per-feature residual pass (usad_feature_resid) is read as
    raw / std / maha. Each readout keeps its own val-AP-best epoch (>= warmup)
    with the test and benign-val scores captured AT that epoch — selection on
    val only; the per-epoch TEST-AP curve is stored for the ORACLE/selection-gap
    report ONLY. The raw-val-AP-best model state is also kept so the alpha-swept
    "paper" row can run at the end (their original selection flow, AUPRC not
    AUROC under the reported evaluation protocol)."""
    history = []
    optimizer1 = opt_func(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()))
    optimizer2 = opt_func(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()))
    track = {r: {"val_ap": -np.inf, "epoch": None,
                 "scores_test": None, "scores_valb": None,
                 "val_curve": [], "test_curve": []} for r in READOUTS}
    best_state, best_state_ap = None, -np.inf
    y_test = y_test_attack_w.astype(int)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)

            loss1, _ = model.training_step(batch, epoch + 1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            _, loss2 = model.training_step(batch, epoch + 1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()

        result = evaluate_usad(model, val_loader, epoch + 1, device)

        # ── per-epoch readout tracking (selection on val only) ──
        E_b = usad_feature_resid(model, X_val_w, device)
        stats = fit_benign_stats(E_b)
        E_vm = usad_feature_resid(model, X_val_mixed_w, device)
        E_te = usad_feature_resid(model, X_test_w, device)
        sel_ok = (epoch + 1) >= MIN_SELECT_EPOCH
        for r in READOUTS:
            s_vm = apply_readout(E_vm, stats, r)
            v_ap = float(average_precision_score(y_val_mixed_w, s_vm))
            s_te = apply_readout(E_te, stats, r)
            t_ap = float(average_precision_score(y_test, s_te))
            track[r]["val_curve"].append(v_ap)
            track[r]["test_curve"].append(t_ap)
            if sel_ok and v_ap > track[r]["val_ap"]:
                track[r].update(
                    val_ap=v_ap, epoch=epoch + 1,
                    scores_test=s_te.astype(np.float32),
                    scores_valb=apply_readout(E_b, stats, r).astype(np.float32))
        if sel_ok and track["raw"]["val_curve"][-1] > best_state_ap:
            best_state_ap = track["raw"]["val_curve"][-1]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        result["epoch"] = epoch + 1
        result["time_s"] = time.time() - t0
        for r in READOUTS:
            result[f"val_ap_{r}"] = track[r]["val_curve"][-1]
        history.append(result)
        if epoch < 5 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print("  Epoch [{}] loss1 {:.4f} loss2 {:.4f}  val_AP raw {:.4f} "
                  "std {:.4f} maha {:.4f}  ({:.0f}s)".format(
                      epoch + 1, result["val_loss1"], result["val_loss2"],
                      result["val_ap_raw"], result["val_ap_std"],
                      result["val_ap_maha"], result["time_s"]))

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Restored raw-val-AP-best state (val_AP={best_state_ap:.6f}) "
              f"for the alpha-sweep paper row")
    return history, track


@torch.no_grad()
def score_usad(model, X, device, alpha=0.5, beta=0.5, batch_size=256):
    model.eval()
    loader = make_usad_loader(X, batch_size=batch_size, shuffle=False)
    scores = []
    for (batch,) in loader:
        batch = batch.to(device)
        w1 = model.decoder1(model.encoder(batch))
        w2 = model.decoder2(model.encoder(w1))
        score = alpha * torch.mean((batch - w1) ** 2, axis=1) + \
            beta * torch.mean((batch - w2) ** 2, axis=1)
        scores.append(score.cpu().numpy())
    return np.concatenate(scores)


@torch.no_grad()
def usad_feature_resid(model, X, device, batch_size=256):
    """Per-window per-feature residual E (N, F): mean over the W positions of
    0.5·(x−AE1(x))² + 0.5·(x−AE2(AE1(x)))². Note E.mean(axis=1) IS the paper score
    at alpha=beta=0.5 — raw and the calibrated readouts aggregate the SAME residual
    tensor, so any difference is attributable to the readout alone."""
    model.eval()
    _, W, F = X.shape
    loader = make_usad_loader(X, batch_size=batch_size, shuffle=False)
    out = []
    for (batch,) in loader:
        batch = batch.to(device)
        w1 = model.decoder1(model.encoder(batch))
        w2 = model.decoder2(model.encoder(w1))
        r = 0.5 * (batch - w1) ** 2 + 0.5 * (batch - w2) ** 2
        out.append(r.view(-1, W, F).mean(1).cpu().numpy())
    return np.concatenate(out)


def fit_benign_stats(E_benign):
    return {
        "mu": E_benign.mean(0),
        "var": E_benign.var(0) + 1e-8,
        "Sinv": np.linalg.pinv(
            np.cov(E_benign, rowvar=False) + 1e-6 * np.eye(E_benign.shape[1])),
    }


def apply_readout(E, stats, readout):
    if readout == "raw":
        return E.mean(1)
    if readout == "std":
        return (E / stats["var"]).mean(1)
    if readout == "maha":
        d = E - stats["mu"]
        return ((d @ stats["Sinv"]) * d).sum(1)
    raise ValueError(readout)


def operating_point(scores_val_benign, scores_test, y_attack, y_family):
    """TPR @ TPR_BUDGET with the threshold calibrated on benign val ONLY."""
    thr = float(np.quantile(scores_val_benign, 1.0 - TPR_BUDGET))
    flag = scores_test >= thr
    atk = y_attack == 1
    hold = np.isin(y_family, HOLDOUT_FAMILIES)
    return {
        "tpr": float(flag[atk].mean()),
        "fpr_actual": float(flag[~atk].mean()),
        "tpr_holdout": (float(flag[atk & hold].mean())
                        if (atk & hold).any() else float("nan")),
    }


def tune_usad_score_weights(model, X_val_mixed, y_val_mixed, device):
    if len(np.unique(y_val_mixed)) < 2:
        print("  Val mixed has one class; using default USAD score weights.")
        return USAD_ALPHA, USAD_BETA, None

    print("\n" + "=" * 80)
    print("  USAD Score Weight Sweep on Val Mixed")
    print("=" * 80)
    best_alpha = USAD_ALPHA
    best_beta = USAD_BETA
    best_val_auprc = -np.inf
    best_val_auroc = 0.0

    for alpha in USAD_ALPHA_GRID:
        beta = 1.0 - alpha
        scores_val = score_usad(
            model, X_val_mixed, device,
            alpha=alpha, beta=beta, batch_size=BATCH_SIZE)
        val_auprc = float(average_precision_score(y_val_mixed, scores_val))
        val_auroc = float(roc_auc_score(y_val_mixed, scores_val))
        marker = ""
        if val_auprc > best_val_auprc:
            best_alpha = alpha
            best_beta = beta
            best_val_auprc = val_auprc
            best_val_auroc = val_auroc
            marker = "  <-- best"
        print(
            f"  alpha={alpha:.1f}, beta={beta:.1f}  "
            f"val_AUROC={val_auroc:.6f}  val_AUPRC={val_auprc:.6f}{marker}"
        )

    return best_alpha, best_beta, {
        "best_alpha": float(best_alpha),
        "best_beta": float(best_beta),
        "best_val_auprc": float(best_val_auprc),
        "best_val_auroc": float(best_val_auroc),
    }


# =============================================================================
# Shared Final Metric Reporting
# =============================================================================

def bootstrap_auprc(y, scores, n_boot=1000, seed=None):
    # CI resampling is fixed independently of the active model run.
    seed = RUN_SEEDS[0] if seed is None else seed
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(average_precision_score(y[idx], scores[idx]))
    vals = np.array(vals)
    return (
        float(np.median(vals)),
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


def evaluate_model(name, scores_test, y_attack, y_family):
    res = {"model": name}
    res["auroc_overall"] = float(roc_auc_score(y_attack, scores_test))
    res["auprc_overall"] = float(average_precision_score(y_attack, scores_test))

    holdout_mask = np.isin(y_family, HOLDOUT_FAMILIES)
    eval_mask = (y_family == 0) | holdout_mask
    y_h = (y_family[eval_mask] != 0).astype(int)
    s_h = scores_test[eval_mask]
    if len(np.unique(y_h)) >= 2:
        auprc_h, ci_lo, ci_hi = bootstrap_auprc(y_h, s_h)
    else:
        auprc_h, ci_lo, ci_hi = 0.0, 0.0, 0.0
    res["auprc_holdout"] = auprc_h
    res["holdout_ci_low"] = ci_lo
    res["holdout_ci_high"] = ci_hi

    benign_mask = y_family == 0
    med_b = float(np.median(scores_test[benign_mask])) if benign_mask.any() else 0.0
    med_a = float(np.median(scores_test[~benign_mask])) if (~benign_mask).any() else 0.0
    res["median_benign"] = med_b
    res["median_attack"] = med_a
    res["direction_correct"] = bool(med_a >= med_b)

    per_family = []
    for fam in sorted(np.unique(y_family)):
        mask = y_family == fam
        scores = scores_test[mask]
        per_family.append({
            "name": CIC_LABEL_MAP.get(fam, f"fam_{fam}"),
            "n": int(mask.sum()),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "std": float(scores.std()),
            "p90": float(np.percentile(scores, 90)),
            "p99": float(np.percentile(scores, 99)),
        })
    res["per_family"] = per_family
    return res


def print_result(res):
    print(f"\n{'=' * 80}")
    print(f"  {res['model']}")
    print(f"{'=' * 80}")
    print(f"  AUROC (overall)     : {res['auroc_overall']:.6f}")
    print(f"  AUPRC (overall)     : {res['auprc_overall']:.6f}")
    print(f"  AUPRC (holdout 8+9) : {res['auprc_holdout']:.6f}  "
          f"95% CI [{res['holdout_ci_low']:.6f}, {res['holdout_ci_high']:.6f}]")
    direction = "CORRECT" if res["direction_correct"] else "INVERTED"
    print(f"  Score direction     : {direction}  "
          f"(benign={res['median_benign']:.2e}, attack={res['median_attack']:.2e})")

    print(f"\n  {'family':<20s} {'n':>7s} {'mean':>10s} {'median':>10s} "
          f"{'std':>10s} {'p90':>10s} {'p99':>10s}")
    print("  " + "-" * 79)
    for row in res["per_family"]:
        print(f"  {row['name']:<20s} {row['n']:>7d} {row['mean']:>10.2e} "
              f"{row['median']:>10.2e} {row['std']:>10.2e} "
              f"{row['p90']:>10.2e} {row['p99']:>10.2e}")


def deep_convert(obj):
    if isinstance(obj, dict):
        return {k: deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_convert(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# =============================================================================
# Main Pipeline
# =============================================================================

def run_experiment(run_seed):
    global SEED
    SEED = run_seed
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(OUTPUT_ROOT, f"seed_{SEED}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")

    (
        X_train_w,
        X_val_w,
        X_val_mixed_w,
        y_val_mixed_w,
        X_test_w,
        y_test_attack_w,
        y_test_family_w,
    ) = load_and_preprocess()

    train_loader = make_usad_loader(
        X_train_w, BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = make_usad_loader(X_val_w, BATCH_SIZE, shuffle=False)

    w_size = X_train_w.shape[1] * X_train_w.shape[2]
    z_size = max(1, int(w_size / 4))
    model = UsadModel(w_size, z_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 80)
    print("  USAD Paper Baseline")
    print("=" * 80)
    print(f"  Parameters: {n_params:,}")
    print(f"  Flattened window size: {w_size}")
    print(f"  Latent size: {z_size}")
    print(f"  Checkpoint score: alpha={USAD_ALPHA}, beta={USAD_BETA}")

    history, track = train_usad(
        EPOCHS, model, train_loader, val_loader, device,
        X_val_w, X_val_mixed_w, y_val_mixed_w, X_test_w, y_test_attack_w)

    # "paper" row: alpha-swept raw score at the raw-val-AP-best state
    # (the original USAD selection flow, kept as the faithful published baseline).
    best_alpha, best_beta, score_selection = tune_usad_score_weights(
        model, X_val_mixed_w, y_val_mixed_w, device)
    scores_test_paper = score_usad(
        model, X_test_w, device, alpha=best_alpha, beta=best_beta,
        batch_size=BATCH_SIZE)
    scores_valb_paper = score_usad(
        model, X_val_w, device, alpha=best_alpha, beta=best_beta,
        batch_size=BATCH_SIZE)

    # Readout comparison
    print("\n" + "=" * 80)
    print("  USAD READOUT COMPARISON — same residuals, different readout")
    print("  (selection: val AP, epochs >= {}, per readout; gap = sel-window "
          "test-oracle − selected)".format(MIN_SELECT_EPOCH))
    print("=" * 80)
    print(f"  {'readout':<14}{'sel_ep':>7}{'val_AP':>9}{'test_AP':>9}"
          f"{'AUROC':>8}{'ho_AP':>8}{'gap':>8}{'TPR@1%':>8}{'FPR_act':>9}{'hoTPR':>7}")
    print("  " + "-" * 86)
    rows = {}
    for r in READOUTS:
        tr = track[r]
        res = evaluate_model(f"USAD[{r}]", tr["scores_test"],
                             y_test_attack_w, y_test_family_w)
        op = operating_point(tr["scores_valb"], tr["scores_test"],
                             y_test_attack_w, y_test_family_w)
        sel_curve = tr["test_curve"][MIN_SELECT_EPOCH - 1:]
        gap = (max(sel_curve) - res["auprc_overall"]) if sel_curve else float("nan")
        rows[r] = {"readout": r, "sel_epoch": tr["epoch"], "val_ap": tr["val_ap"],
                   "gap": gap, "operating_point": op, **res}
        print(f"  {r:<14}{tr['epoch']:>7}{tr['val_ap']:>9.4f}"
              f"{res['auprc_overall']:>9.4f}{res['auroc_overall']:>8.4f}"
              f"{res['auprc_holdout']:>8.4f}{gap:>+8.4f}{op['tpr']:>8.4f}"
              f"{op['fpr_actual']:>9.4f}{op['tpr_holdout']:>7.4f}")
    res_p = evaluate_model("USAD[paper]", scores_test_paper,
                           y_test_attack_w, y_test_family_w)
    op_p = operating_point(scores_valb_paper, scores_test_paper,
                           y_test_attack_w, y_test_family_w)
    rows["paper"] = {"readout": "paper_alpha_swept", "sel_epoch": None,
                     "val_ap": (score_selection or {}).get("best_val_auprc"),
                     "gap": float("nan"), "operating_point": op_p,
                     "score_selection": score_selection, **res_p}
    print(f"  {'paper(a-swept)':<14}{'—':>7}{'':>9}"
          f"{res_p['auprc_overall']:>9.4f}{res_p['auroc_overall']:>8.4f}"
          f"{res_p['auprc_holdout']:>8.4f}{'':>8}{op_p['tpr']:>8.4f}"
          f"{op_p['fpr_actual']:>9.4f}{op_p['tpr_holdout']:>7.4f}")
    d_std = rows["std"]["auprc_overall"] - rows["raw"]["auprc_overall"]
    d_maha = rows["maha"]["auprc_overall"] - rows["raw"]["auprc_overall"]
    print(f"\n  CALIBRATION DELTA (this seed): std−raw {d_std:+.4f}  "
          f"maha−raw {d_maha:+.4f}   (holdout: "
          f"{rows['std']['auprc_holdout'] - rows['raw']['auprc_holdout']:+.4f} / "
          f"{rows['maha']['auprc_holdout'] - rows['raw']['auprc_holdout']:+.4f})")
    print_result(rows["raw"])
    print_result(rows["maha"] if d_maha >= d_std else rows["std"])

    torch.save(
        {
            "model": model.state_dict(),
            "seed": SEED,
            "w_size": w_size,
            "z_size": z_size,
            "score_selection": score_selection,
            "history": history,
        },
        os.path.join(output_dir, "checkpoint.pt"),
    )
    curves = {r: {"val": track[r]["val_curve"], "test": track[r]["test_curve"]}
              for r in READOUTS}
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(deep_convert({"history": history, "curves": curves}), f, indent=2)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(deep_convert(rows), f, indent=2)

    print(f"\nSaved USAD results to {output_dir}")
    return rows


def main():
    all_results = []
    for seed in RUN_SEEDS:
        print(f"\n\n{'=' * 80}\n  STARTING USAD RUN WITH SEED {seed}\n{'=' * 80}")
        all_results.append(run_experiment(seed))

    print(f"\n\n{'=' * 80}\n  AGGREGATED USAD RESULTS OVER {len(RUN_SEEDS)} SEEDS\n{'=' * 80}")
    keys = READOUTS + ["paper"]
    for metric in ["auprc_overall", "auprc_holdout", "auroc_overall"]:
        print(f"\n  {metric}:")
        for k in keys:
            vals = np.array([r[k][metric] for r in all_results], dtype=float)
            print(f"    {k:<16s}: {vals.mean():.4f} +/- {vals.std():.4f}   "
                  f"per-seed {np.round(vals, 4).tolist()}")
    print("\n  PAIRED CALIBRATION DELTA (the systematic-under-reading test):")
    for cal in ("std", "maha"):
        for metric, lbl in (("auprc_overall", "test"), ("auprc_holdout", "ho")):
            d = np.array([r[cal][metric] - r["raw"][metric]
                          for r in all_results], dtype=float)
            print(f"    {cal}−raw [{lbl:<4}]: mean {d.mean():+.4f}  "
                  f"positive on {(d > 0).sum()}/{len(d)} seeds  "
                  f"per-seed {np.round(d, 4).tolist()}")
    print("\n  TPR@1% (val-calibrated budget):")
    for k in keys:
        t = np.array([r[k]["operating_point"]["tpr"] for r in all_results])
        f_ = np.array([r[k]["operating_point"]["fpr_actual"] for r in all_results])
        print(f"    {k:<16s}: TPR {t.mean():.4f}  actual FPR {f_.mean():.4f}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUTPUT_ROOT, "aggregated_results.json")
    with open(out_path, "w") as f:
        json.dump(deep_convert(all_results), f, indent=2)
    print(f"\nSaved aggregated USAD results to {out_path}")


if __name__ == "__main__":
    main()
