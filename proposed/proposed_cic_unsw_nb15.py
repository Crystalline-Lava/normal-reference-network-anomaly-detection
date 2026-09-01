"""Proposed normal-reference detector for the CIC-UNSW-NB15 dataset."""


from __future__ import annotations
import json, math, time, os, random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
# Required by PyTorch for deterministic cublas kernels on CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
from scipy.stats import gaussian_kde, mstats
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ModuleNotFoundError:
    drive = None

# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Fixed five-seed set reported in the manuscript.
RUN_SEEDS = (42, 456, 7, 789, 1024)
SEED = RUN_SEEDS[0]  # Active model seed; reassigned by run_experiment().
STRICT_DETERMINISM = False
FIX_TRAIN_SHUFFLE_ORDER = False

# ── GPU acceleration (bit-for-bit numerics-preserving) ──────────────────────
# Keep train/val window tensors resident on the GPU so the per-batch CPU→GPU
# copy disappears from the hot training loop. The DataLoader sampler is left
# untouched, so the shuffle order / RNG stream and every arithmetic op are
# identical to the CPU-fed path — same trajectory, just no longer input-starved.
# This does NOT change results; safe for publication runs.
GPU_RESIDENT_DATA = True

# Label encoding: string → integer
LABEL_TO_ID = {
    "Benign": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3,
    "Exploits": 4, "Fuzzers": 5, "Generic": 6,
    "Reconnaissance": 7, "Shellcode": 8, "Worms": 9,
}
CIC_LABEL_MAP = {v: k for k, v in LABEL_TO_ID.items()}  # int → string
HOLDOUT_FAMILIES = [8, 9]  # Shellcode, Worms

# Paths (adjust for your environment: Colab / local)
BASE_DIR = os.getenv("CIC_BASE_DIR", "/content/drive/MyDrive/cic-reproduce")
RAW_CSV = os.getenv("CIC_UNSW_RAW_CSV", "/content/drive/MyDrive/CIC-UNSW-NB15/CICFlowMeter_out.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "ablation_outputs")

# Time-based split
TRAIN_CUTOFF = pd.Timestamp("2015-01-23")

# Columns to drop from features (metadata)
DROP_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port",
             "Protocol", "Timestamp", "Label"]

# Reported training conditions. S1 is the retained detector; P0 removes
# train-time SVDD, and S1_nodelta removes delta inputs.
ABLATIONS_TO_RUN = ["S1"]
ABLATIONS = {
    "P0": {
        "name": "prediction_only",
        "description": "Prediction-only control: same global-z predictor, delta features, "
                    "masking, data protocol, and training schedule as S1, but no "
                    "train-time SVDD loss.",
        "use_delta_features": True,
        "error_signal": "prediction",
        "train_svdd": False,
        "lambda_var": 0.0,
        "lambda_cov": 0.0,
        "cov_target": "projection",
    },
    "S1": {
        "name": "svdd_only",
        "description": "Retained detector: prediction with delta inputs, KDE loss "
                    "weighting, and train-time SVDD, without VICReg.",
        "use_delta_features": True,
        "error_signal": "prediction",
        "train_svdd": True,
        "lambda_var": 0.0,
        "lambda_cov": 0.0,
        "cov_target": "projection",
    },
    "S1_nodelta": {
        "name": "svdd_only_nodelta",
        "description": "S1 paired control with delta inputs removed.",
        "use_delta_features": False,
        "error_signal": "prediction",
        "train_svdd": True,
        "lambda_var": 0.0,
        "lambda_cov": 0.0,
        "cov_target": "projection",
    },
}

# Entity grouping & windowing
ENTITY_COLS = ["Src IP"]        # group flows by this key
WINDOW_SIZE = 50                # flows per window
TRAIN_STRIDE = 25               # stride for training windows (50% overlap)
TEST_STRIDE = 10                # stride for test windows (dense overlap)
MAX_TRAIN_WINDOWS = 150_000     # memory safety cap
USE_DELTA_FEATURES = True       # overwritten per ablation

# Architecture
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 4
D_FF = 256
DROPOUT = 0.1
BOTTLENECK_DIM = 64

# Training
COMPARE_READOUTS = False         # True re-scores saved checkpoints by readout.
SKIP_TRAINING = False
RECOMPUTE_CENTER_ON_LOAD = True  # avoid stale checkpoint centers after best-state restore
# Prediction readout used throughout the per-epoch trajectory: "raw"
# (0.7 max + 0.3 mean MSE),
# "standardized" (÷ benign per-feature variance — strong for global_z, cheap), or "maha"
# (benign-cov decorrelated). The fusion/corr/holdout analysis then uses
# this stronger pred signal automatically. Benign stats fit on X_val_w only (no leakage).
PRED_READOUT = "standardized"
# Sphere readout (latent-space deviation): "euclid" (‖z−c‖²) or "maha" (benign-cov
# whitened Mahalanobis in latent space — the calibrated counterpart of the pred readout, so
# pred and sphere are compared on equal footing). Only used when PRED_READOUT != "raw".
SPHERE_READOUT = "maha"
# When True, save the per-epoch signal trajectory and leave fusion selection
# and metric computation to the separate trajectory-analysis workflow.
TRAJECTORY_ONLY = True
EPOCHS = 120
BATCH_SIZE = 256
LR = 5e-5
WEIGHT_DECAY = 1e-5
LAMBDA_VAR = 1.0                # variance regularization strength
MIN_VARIANCE = 0.5              # minimum z variance to maintain
LAMBDA_COV = 0.04               # covariance regularization (off-diagonal decorrelation)
LAMBDA_SVDD = 3.0               # train-time sphere strength
COV_WARMUP = 0                  # ramp λ_cov linearly over this many epochs (0=disabled)
MASK_RATIO = 0.15               # fraction of flows masked during training

# Optional contrastive pseudo-anomaly augmentation (disabled in reported runs).
LAMBDA_CONTRAST = 0.0           # set >0 to enable; tested at 0.05–0.5, destabilised training
CONTRAST_MARGIN = 50.0          # min sphere distance for pseudo-anomalies
CONTRAST_WARMUP = 30            # delay until model stabilises
ENABLE_LEARNED_FUSION = True    # compare alpha blend against learned logreg fusion on val_mixed

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
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading & Preprocessing — Sequence Windows
# ═══════════════════════════════════════════════════════════════════════════════

def build_windows(X, y, entity_ids, window_size, stride, use_deltas=False):
    """Build sliding windows per entity (data must be timestamp-sorted).

    Args:
        X: (N, F) scaled feature matrix
        y: (N,) integer family labels
        entity_ids: (N,) integer entity codes
        window_size: W
        stride: S
        use_deltas: if True, append flow[t]-flow[t-1] as extra features

    Returns:
        windows: (M, W, F) or (M, W, 2F) if use_deltas, float32
        attack_labels: (M,) binary
        family_labels: (M,) integer (max family in window)
    """
    windows, family_labels = [], []
    for eid in np.unique(entity_ids):
        mask = (entity_ids == eid)
        X_e = X[mask]
        y_e = y[mask]
        n = len(X_e)
        for start in range(0, n - window_size + 1, stride):
            win = X_e[start:start + window_size]
            if use_deltas:
                deltas = np.diff(win, axis=0, prepend=win[:1])  # (W, F)
                win = np.concatenate([win, deltas], axis=1)     # (W, 2F)
            windows.append(win)
            family_labels.append(int(y_e[start:start + window_size].max()))
    if len(windows) == 0:
        F = X.shape[1] * (2 if use_deltas else 1)
        return (np.zeros((0, window_size, F), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64))
    windows = np.stack(windows).astype(np.float32)
    family_labels = np.array(family_labels, dtype=np.int64)
    attack_labels = (family_labels != 0).astype(np.int64)
    return windows, attack_labels, family_labels


def load_and_preprocess():
    """Load raw CICFlowMeter CSV → entity-grouped windows, time-split, scale.

    Time-based split strategy:
      - Train: windows from Jan 22 benign (earlier 85% per entity)
      - Val (pure): windows from Jan 22 benign (later 15% per entity)
      - Val (mixed): val benign windows + 10% known-attack windows
      - Test: remaining Feb 17+18 windows after removing val_mixed attack windows
      - Holdout: families 8+9 excluded from val_mixed
    """
    set_seed()

    # ── Load raw CSV ──
    print("Loading raw CICFlowMeter CSV ...")
    raw = pd.read_csv(RAW_CSV)
    print(f"  Raw shape: {raw.shape}")

    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], dayfirst=True)
    raw.sort_values("Timestamp", inplace=True)
    raw.reset_index(drop=True, inplace=True)

    # Labels
    label_str = raw["Label"].str.strip()
    y_all = label_str.map(LABEL_TO_ID).values.astype(int)
    print(f"  Timestamp range: {raw['Timestamp'].min()} → {raw['Timestamp'].max()}")
    print(f"  Label distribution:\n{label_str.value_counts().to_string()}")

    # Entity IDs (extract BEFORE dropping metadata)
    entity_str = raw[ENTITY_COLS].astype(str).agg("_".join, axis=1)
    entity_ids = pd.Categorical(entity_str).codes
    entity_map = dict(enumerate(pd.Categorical(entity_str).categories))
    print(f"  Entities ({', '.join(ENTITY_COLS)}): {len(entity_map)} unique")

    # ── Extract features ──
    data = raw.drop(columns=[c for c in DROP_COLS if c in raw.columns])
    for c in ENTITY_COLS:
        if c in data.columns:
            data = data.drop(columns=[c])

    var = data.var(numeric_only=True)
    low_var = var[var < 1e-10].index.tolist()
    data = data.drop(columns=low_var)
    print(f"  Removed {len(low_var)} near-constant features → {data.shape[1]} remain")

    X_all = data.values.astype(np.float64)

    # Clean
    X_all[~np.isfinite(X_all)] = np.nan
    imputer = SimpleImputer(strategy="median")
    X_all = imputer.fit_transform(X_all)
    for j in range(X_all.shape[1]):
        X_all[:, j] = mstats.winsorize(X_all[:, j], limits=[0.001, 0.001])

    # ── Time-based split ──
    ts = raw["Timestamp"].values
    train_period = ts < np.datetime64(TRAIN_CUTOFF)
    test_period = ~train_period
    benign = (y_all == 0)

    # Per-entity train/val split (85/15 by position within each entity)
    train_indices, val_indices = [], []
    for eid in np.unique(entity_ids[train_period & benign]):
        eid_idx = np.where(train_period & benign & (entity_ids == eid))[0]
        n = len(eid_idx)
        split = int(n * 0.85)
        train_indices.extend(eid_idx[:split].tolist())
        val_indices.extend(eid_idx[split:].tolist())
    train_indices = np.array(train_indices, dtype=int)
    val_indices = np.array(val_indices, dtype=int)
    test_indices = np.where(test_period)[0]

    print(f"\n  Train period: {len(train_indices)} benign flows (85%)")
    print(f"  Val period:   {len(val_indices)} benign flows (15%)")
    print(f"  Test period:  {len(test_indices)} all flows")

    # ── Scale (fit on train flows) ──
    scaler = StandardScaler()
    scaler.fit(X_all[train_indices])
    X_scaled = scaler.transform(X_all).astype(np.float32)

    # Keep individual train flows for KDE fitting
    X_train_flows = X_scaled[train_indices].copy()

    # ── Build windows ──
    print(f"\n  Building windows (W={WINDOW_SIZE}) ...")

    X_train_w, _, _ = build_windows(
        X_scaled[train_indices], y_all[train_indices],
        entity_ids[train_indices], WINDOW_SIZE, TRAIN_STRIDE,
        use_deltas=USE_DELTA_FEATURES)
    X_val_w, _, _ = build_windows(
        X_scaled[val_indices], y_all[val_indices],
        entity_ids[val_indices], WINDOW_SIZE, TRAIN_STRIDE,
        use_deltas=USE_DELTA_FEATURES)
    X_test_w, y_test_atk_w, y_test_fam_w = build_windows(
        X_scaled[test_indices], y_all[test_indices],
        entity_ids[test_indices], WINDOW_SIZE, TEST_STRIDE,
        use_deltas=USE_DELTA_FEATURES)

    # Cap train windows for memory
    if len(X_train_w) > MAX_TRAIN_WINDOWS:
        idx = np.random.choice(len(X_train_w), MAX_TRAIN_WINDOWS, replace=False)
        X_train_w = X_train_w[idx]
        print(f"  ⚠ Subsampled train windows: {len(X_train_w)} → {MAX_TRAIN_WINDOWS}")

    # ── Val mixed: val benign windows + 10% known-attack windows ──
    # The sampled known-attack validation windows are removed from test below.
    holdout_w = np.isin(y_test_fam_w, HOLDOUT_FAMILIES)
    known_atk = (y_test_atk_w == 1) & ~holdout_w
    known_atk_idx = np.where(known_atk)[0]
    val_atk_idx = np.zeros(0, dtype=int)

    if len(known_atk_idx) > 0:
        try:
            val_atk_idx, _ = train_test_split(
                known_atk_idx, test_size=0.90, random_state=SEED,
                stratify=y_test_fam_w[known_atk_idx])
        except ValueError:
            val_atk_idx = np.random.choice(
                known_atk_idx, max(1, len(known_atk_idx) // 10), replace=False)
        X_val_mixed_w = np.concatenate([X_val_w, X_test_w[val_atk_idx]])
        y_val_mixed_w = np.concatenate([
            np.zeros(len(X_val_w), dtype=int),
            np.ones(len(val_atk_idx), dtype=int)])
        test_keep = np.ones(len(X_test_w), dtype=bool)
        test_keep[val_atk_idx] = False
        X_test_w = X_test_w[test_keep]
        y_test_atk_w = y_test_atk_w[test_keep]
        y_test_fam_w = y_test_fam_w[test_keep]
    else:
        X_val_mixed_w = X_val_w.copy()
        y_val_mixed_w = np.zeros(len(X_val_w), dtype=int)

    print("\n  Windows built:")
    print(f"    Train:     {X_train_w.shape}")
    print(f"    Val:       {X_val_w.shape}")
    print(f"    Val mixed: {X_val_mixed_w.shape} (attack rate={y_val_mixed_w.mean():.4f})")
    print(f"      Known-attack val windows removed from test: {len(val_atk_idx)}")
    print(f"    Test:      {X_test_w.shape} (attack rate={y_test_atk_w.mean():.4f})")
    print(f"    KDE train flows: {X_train_flows.shape}")

    feature_names = data.columns.tolist()
    if USE_DELTA_FEATURES:
        delta_names = [f"Δ_{c}" for c in feature_names]
        feature_names_full = feature_names + delta_names
    else:
        feature_names_full = feature_names

    return (X_train_w, X_val_w, X_val_mixed_w, y_val_mixed_w,
            X_test_w, y_test_atk_w, y_test_fam_w,
            X_train_flows, feature_names, feature_names_full, scaler)


# ═══════════════════════════════════════════════════════════════════════════════
#  KDE Explainability Module (operates on individual flows)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureKDE:
    """Per-feature Gaussian KDE for explainability."""

    def __init__(self):
        self.kdes = []
        self.n_features = 0
        self.feature_names = []

    def fit(self, X_train, feature_names=None, max_samples=5000):
        """Fit one KDE per feature on benign training flows."""
        self.n_features = X_train.shape[1]
        self.feature_names = feature_names or [f"f{i}" for i in range(self.n_features)]

        if len(X_train) > max_samples:
            idx = np.random.choice(len(X_train), max_samples, replace=False)
            X_sub = X_train[idx]
            print(f"  Subsampled {len(X_train)} → {max_samples} for KDE fitting")
        else:
            X_sub = X_train

        self.kdes = []
        print(f"Fitting KDE on {self.n_features} features ...")
        for i in range(self.n_features):
            col = X_sub[:, i].astype(np.float64)
            if np.std(col) < 1e-10:
                col = col + np.random.normal(0, 1e-8, len(col))
            kde = gaussian_kde(col, bw_method='scott')
            self.kdes.append(kde)
        print("  KDE fitting complete.")

    def score_features(self, X):
        """Per-feature anomaly scores (higher = more anomalous)."""
        scores = np.zeros((X.shape[0], self.n_features))
        for i in range(self.n_features):
            log_pdf = self.kdes[i].logpdf(X[:, i].astype(np.float64))
            scores[:, i] = -log_pdf
        return scores

    def top_k_features(self, X, k=5):
        """Top-k most anomalous features per sample."""
        scores = self.score_features(X)
        return np.argsort(scores, axis=1)[:, -k:][:, ::-1]

    def compute_feature_weights(self):
        """Per-feature weights based on benign density tightness."""
        weights = np.zeros(self.n_features)
        x_grid = np.linspace(-3, 3, 200)
        for i, kde_i in enumerate(self.kdes):
            density = kde_i.evaluate(x_grid)
            weights[i] = density.max()
        weights = weights / (weights.mean() + 1e-8)
        return weights

    def explain_samples(self, X, k=5):
        """Per-sample KDE explanation for individual flows."""
        scores = self.score_features(X)
        explanations = []
        for i in range(len(X)):
            top_idx = np.argsort(scores[i])[-k:][::-1]
            explanations.append([
                {"feature": self.feature_names[j],
                 "kde_score": float(scores[i, j]),
                 "value": float(X[i, j])}
                for j in top_idx
            ])
        return explanations

    def explain_family(self, X, y_family, k=10, max_per_family=1000):
        """Per-family KDE explanation (operates on individual flows)."""
        results = {}
        for fam in sorted(np.unique(y_family)):
            if fam == 0:
                continue
            mask = (y_family == fam)
            X_fam = X[mask]
            if len(X_fam) > max_per_family:
                idx = np.random.choice(len(X_fam), max_per_family, replace=False)
                X_fam = X_fam[idx]
            scores = self.score_features(X_fam)
            mean_scores = scores.mean(axis=0)
            top_idx = np.argsort(mean_scores)[-k:][::-1]
            results[CIC_LABEL_MAP.get(fam, f"family_{fam}")] = [
                {"rank": r+1, "feature": self.feature_names[idx],
                 "mean_kde_score": float(mean_scores[idx])}
                for r, idx in enumerate(top_idx)
            ]
        return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Architecture — Sequence-Window Transformer-AE
# ═══════════════════════════════════════════════════════════════════════════════

class FlowTokenizer(nn.Module):
    """Project each flow's feature vector (F dims) → d_model token."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        # x: (B, W, F) → (B, W, d_model)
        return self.proj(x)




class SequenceTransformerAE(nn.Module):
    """Temporal Transformer-AE over entity flow windows.

    Encoder: FlowTokenizer + [CLS] token + positional encoding + Transformer
             → CLS output → bottleneck z
    Decoder: position-aware heavier MLP from z → reconstruct each flow
    """

    def __init__(self, n_features, window_size, d_model=64, n_heads=4,
                 n_layers=4, d_ff=256, dropout=0.1, bottleneck_dim=64):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

        # Encoder
        self.tokenizer = FlowTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # W+1 positions: [CLS] + W flow tokens
        self.pos_encoding = nn.Parameter(
            torch.randn(1, window_size + 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # Bottleneck
        self.bottleneck_proj = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim), nn.GELU(),
        )

        self.dec_pos_emb = nn.Parameter(
            torch.randn(1, window_size, bottleneck_dim) * 0.02)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_features),
        )
        # Projection head: used only for covariance regularization during training
        # (discarded at inference, following SimCLR/VICReg convention)
        self.proj_head = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
        )

    def encode(self, x):
        # x: (B, W, F)
        B = x.size(0)
        tokens = self.tokenizer(x)                          # (B, W, d_model)
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)            # (B, W+1, d_model)
        tokens = tokens + self.pos_encoding                 # (B, W+1, d_model)
        encoded = self.encoder_norm(self.encoder(tokens))   # (B, W+1, d_model)
        cls_out = encoded[:, 0, :]                          # (B, d_model)
        z = self.bottleneck_proj(cls_out)                    # (B, bottleneck_dim)
        return z, encoded

    def project(self, z):
        """Project z → p for covariance regularization (training only)."""
        return self.proj_head(z)                             # (B, bottleneck_dim)

    def decode(self, z):
        # z: (B, bottleneck_dim)
        z_exp = z.unsqueeze(1).expand(-1, self.window_size, -1)  # (B, W, bd)
        z_pos = z_exp + self.dec_pos_emb                         # (B, W, bd)
        return self.decoder(z_pos)                               # (B, W, F)

    def forward(self, x):
        z, _ = self.encode(x)
        p = self.project(z)
        return self.decode(z), z, p


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss Function
# ═══════════════════════════════════════════════════════════════════════════════

class PredictionLoss(nn.Module):
    """Prediction loss with optional regularization and train-time SVDD."""

    def __init__(self, lambda_var=1.0, min_variance=0.5, lambda_cov=0.04,
                 feature_weights=None, error_signal="prediction",
                 cov_target="projection", train_svdd=False, lambda_svdd=1.0):
        super().__init__()
        self.lambda_var = lambda_var
        self.min_variance = min_variance
        self.lambda_cov = lambda_cov
        self.error_signal = error_signal
        self.cov_target = cov_target
        self.train_svdd = train_svdd
        self.lambda_svdd = lambda_svdd
        if feature_weights is not None:
            self.register_buffer("feature_weights", feature_weights)
        else:
            self.feature_weights = None

    def forward(self, x, x_hat, z, p, center=None):
        if self.error_signal == "prediction":
            # Next-flow prediction: decoder[t] should predict input[t+1]
            x_target = x[:, 1:, :]       # (B, W-1, F)
            x_pred = x_hat[:, :-1, :]    # (B, W-1, F)
        elif self.error_signal == "reconstruction":
            # Standard AE reconstruction: decoder[t] should reconstruct input[t]
            x_target = x
            x_pred = x_hat
        else:
            raise ValueError(f"Unknown error_signal: {self.error_signal}")

        if self.feature_weights is not None:
            pred_err = torch.mean(self.feature_weights * (x_pred - x_target) ** 2)
        else:
            pred_err = F.mse_loss(x_pred, x_target)

        # Variance regularization on z: prevent embedding collapse
        z_var = z.var(dim=0).mean()   # mean variance across bottleneck dims
        var_loss = F.relu(self.min_variance - z_var)

        # Covariance regularization on projected representation p.
        cov_source = z if self.cov_target == "z" else p
        p_centered = cov_source - cov_source.mean(dim=0)
        cov_matrix = (p_centered.T @ p_centered) / max(cov_source.shape[0] - 1, 1)
        d = cov_source.shape[1]
        off_diag_mask = ~torch.eye(d, device=p.device, dtype=torch.bool)
        cov_loss = cov_matrix[off_diag_mask].pow(2).sum() / d

        svdd_loss = torch.tensor(0.0, device=x.device)
        if self.train_svdd:
            if center is None:
                raise ValueError("center is required when train_svdd=True")
            svdd_loss = torch.sum((z - center.detach())**2, dim=1).mean()

        total = (
            pred_err
            + self.lambda_var * var_loss
            + self.lambda_cov * cov_loss
            + self.lambda_svdd * svdd_loss
        )
        return total, pred_err, var_loss, cov_loss, svdd_loss


# ═══════════════════════════════════════════════════════════════════════════════
#  Training Utilities
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def init_center(model, dataloader, device, eps=0.1):
    """Initialize Deep SVDD center as mean of z over benign training windows."""
    model.eval()
    z_list = []
    for (batch_x,) in dataloader:
        z, _ = model.encode(batch_x.to(device))
        z_list.append(z)
    center = torch.cat(z_list).mean(dim=0)
    center[(center.abs() < eps) & (center < 0)] = -eps
    center[(center.abs() < eps) & (center >= 0)] = eps
    return center


def generate_pseudo_anomalies(bx):
    """Generate pseudo-anomalous windows from a batch of normal windows.

    Two strategies (randomly mixed per window):
      1. Temporal shuffle — randomly permute flow order within the window,
         breaking the temporal patterns the model is learning to predict.
      2. Gaussian noise — inject strong noise (σ=2.0) into random features,
         creating out-of-distribution windows.

    Args:
        bx: (B, W, F) batch of normal windows (already on device)
    Returns:
        bx_pseudo: (B, W, F) batch of pseudo-anomalous windows
    """
    B, W, F = bx.shape
    bx_pseudo = bx.clone()

    # Strategy mask: 50% shuffle, 50% noise
    use_shuffle = torch.rand(B, device=bx.device) < 0.5

    # Strategy 1: Temporal shuffle — permute flow order per window
    shuffle_idx = torch.argsort(torch.rand(B, W, device=bx.device), dim=1)
    shuffled = torch.gather(bx, 1, shuffle_idx.unsqueeze(-1).expand(-1, -1, F))
    bx_pseudo[use_shuffle] = shuffled[use_shuffle]

    # Strategy 2: Gaussian noise injection on ~30% of features
    noise_mask = (torch.rand(B, 1, F, device=bx.device) < 0.3).float()
    noise = torch.randn_like(bx) * 2.0 * noise_mask
    bx_pseudo[~use_shuffle] = (bx + noise)[~use_shuffle]

    return bx_pseudo


def train_one_epoch(model, loader, optimizer, loss_fn, device,
                    mask_ratio=0.0, center=None, lambda_contrast=0.0,
                    contrast_margin=10.0):
    """Train with temporal masking + optional contrastive pseudo-anomaly loss.

    When lambda_contrast > 0 and center is provided:
      1. Generate pseudo-anomalous windows from each normal batch
      2. Encode both normal and pseudo windows
      3. Contrastive loss pushes pseudo embeddings away from center:
         L_contrast = mean(max(0, margin - ||z_pseudo - c||²))
         where c is DETACHED (no gradient through center)
    """
    model.train()
    sums = {"total": 0, "pred": 0, "var": 0, "cov": 0, "svdd": 0, "contrast": 0}
    n = 0
    for (bx,) in loader:
        bx = bx.to(device)
        # Temporal masking: randomly zero out some flows in the window
        if mask_ratio > 0:
            mask = (torch.rand(bx.shape[0], bx.shape[1], 1,
                               device=device) > mask_ratio).float()
            bx_masked = bx * mask  # zero out masked flows
        else:
            bx_masked = bx
        optimizer.zero_grad()
        x_hat, z, p = model(bx_masked)
        # Prediction loss against ORIGINAL (unmasked) input
        total, pred, var, cov, svdd = loss_fn(bx, x_hat, z, p, center=center)

        # Contrastive loss on pseudo-anomalies
        contrast_loss = torch.tensor(0.0, device=device)
        if lambda_contrast > 0 and center is not None:
            bx_pseudo = generate_pseudo_anomalies(bx)
            z_pseudo, _ = model.encode(bx_pseudo)
            # Push pseudo embeddings to be at least `margin` from center
            c_detached = center.detach()
            dist_pseudo = torch.sum((z_pseudo - c_detached)**2, dim=1)
            contrast_loss = F.relu(contrast_margin - dist_pseudo).mean()
            total = total + lambda_contrast * contrast_loss

        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sums["total"] += total.item()
        sums["pred"] += pred.item()
        sums["var"] += var.item()
        sums["cov"] += cov.item()
        sums["svdd"] += svdd.item()
        sums["contrast"] += contrast_loss.item()
        n += 1
    return {k: v/n for k, v in sums.items()}


@torch.no_grad()
def validate(model, loader, loss_fn, device, center=None):
    model.eval()
    sums = {"total": 0, "pred": 0, "var": 0, "cov": 0, "svdd": 0}
    n = 0
    for (bx,) in loader:
        bx = bx.to(device)
        x_hat, z, p = model(bx)
        total, pred, var, cov, svdd = loss_fn(bx, x_hat, z, p, center=center)
        sums["total"] += total.item()
        sums["pred"] += pred.item()
        sums["var"] += var.item()
        sums["cov"] += cov.item()
        sums["svdd"] += svdd.item()
        n += 1
    return {k: v/n for k, v in sums.items()}


def compute_window_error(bx, x_hat, error_signal):
    if error_signal == "prediction":
        err = torch.mean((bx[:, 1:, :] - x_hat[:, :-1, :])**2, dim=2)
    elif error_signal == "reconstruction":
        err = torch.mean((bx - x_hat)**2, dim=2)
    else:
        raise ValueError(f"Unknown error_signal: {error_signal}")
    err_max = torch.max(err, dim=1).values
    err_mean = torch.mean(err, dim=1)
    return 0.7 * err_max + 0.3 * err_mean


@torch.no_grad()
def compute_raw_signals(model, X, center, device, batch_size=256,
                        error_signal="prediction"):
    """Compute pred_score and sphere_score separately (no blending).

    Returns:
        pred_scores: (N,)  blended prediction error per window
        sphere_scores: (N,)  squared distance to center per window
    """
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    pred_list, sphere_list = [], []
    for (bx,) in loader:
        bx = bx.to(device)
        x_hat, z, _ = model(bx)
        pred = compute_window_error(bx, x_hat, error_signal)
        sphere = torch.sum((z - center)**2, dim=1)
        pred_list.append(pred.cpu().numpy())
        sphere_list.append(sphere.cpu().numpy())
    return np.concatenate(pred_list), np.concatenate(sphere_list)


def build_fusion_features(pred_scores: np.ndarray, sphere_scores: np.ndarray) -> np.ndarray:
    pred_feat = np.log1p(np.clip(np.asarray(pred_scores, dtype=np.float64), 0.0, None))
    sphere_feat = np.log1p(np.clip(np.asarray(sphere_scores, dtype=np.float64), 0.0, None))
    return np.column_stack([pred_feat, sphere_feat])


@torch.no_grad()
def _per_feature_resid(model, X, device, error_signal, batch_size=512):
    """Per-window, per-feature squared residual (mean over positions): (N, F).

    This is the raw material every pred readout aggregates differently. The current
    scoring (compute_window_error) collapses features by an UNWEIGHTED mean — the
    suspected reason pred_only caps at ~0.70 while the sphere (latent-space deviation)
    reaches ~0.90. Here we keep the per-feature error so we can try better feature
    aggregations from the SAME forward pass (no retraining).
    """
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    out = []
    for (bx,) in loader:
        bx = bx.to(device)
        x_hat, _, _ = model(bx)
        if error_signal == "prediction":
            r2 = (bx[:, 1:, :] - x_hat[:, :-1, :]) ** 2     # (B, W-1, F)
        else:
            r2 = (bx - x_hat) ** 2
        out.append(r2.mean(dim=1).cpu().numpy())            # mean over positions → (B, F)
    return np.concatenate(out)


def _apply_readout(E, stats, readout):
    """Map per-feature residual energy E (N,F) → scalar pred score per window.

    "standardized": mean over features of E ÷ benign per-feature variance (cheap, O(NF)).
    "maha":         Mahalanobis distance of E under benign (mu, Sigma^-1) — decorrelated.
    """
    if readout == "standardized":
        return (E / stats["var"]).mean(1)
    if readout == "maha":
        d = E - stats["mu"]
        return ((d @ stats["Sinv"]) * d).sum(1)   # BLAS quadratic form
    raise ValueError(f"Unknown PRED_READOUT: {readout}")


@torch.no_grad()
def _resid_and_z(model, X, device, error_signal, batch_size=256):
    """One forward pass → (per-feature residual energy E (N,F), latent Z (N,bottleneck))."""
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    E_list, Z_list = [], []
    for (bx,) in loader:
        bx = bx.to(device)
        x_hat, z, _ = model(bx)
        if error_signal == "prediction":
            r2 = (bx[:, 1:, :] - x_hat[:, :-1, :]) ** 2
        else:
            r2 = (bx - x_hat) ** 2
        E_list.append(r2.mean(dim=1).cpu().numpy())
        Z_list.append(z.cpu().numpy())
    return np.concatenate(E_list), np.concatenate(Z_list)


def _sphere_readout(Z, center_np, z_stats, readout):
    """Latent-space deviation. "euclid": ‖z−center‖². "maha": benign-cov whitened
    Mahalanobis distance in latent space — the calibrated counterpart of the pred readout."""
    if readout == "euclid":
        d = Z - center_np
        return (d * d).sum(1)
    if readout == "maha":
        d = Z - z_stats["mu"]
        return ((d @ z_stats["Sinv"]) * d).sum(1)
    raise ValueError(f"Unknown SPHERE_READOUT: {readout}")


def compare_pred_readouts(model, X_val_benign, X_test, y_attack, y_family,
                          center, device, error_signal, feat_w=None):
    """Compare pred_only AUPRC under several feature-aggregation readouts, one forward pass.

    Tests the hypothesis that prediction's ~0.70 ceiling is a READOUT artifact (raw
    unweighted MSE dominated by noisy features), not intrinsic weakness — by re-reading
    the SAME residuals with: raw_orig (0.7max+0.3mean), raw_mean, KDE-weighted,
    per-feature standardized (÷ benign var), and Mahalanobis (benign-cov decorrelated).
    sphere is printed as the reference upper bound (latent-space deviation).
    """
    pred_orig, sphere = compute_raw_signals(
        model, X_test, center, device, error_signal=error_signal)
    Ev = _per_feature_resid(model, X_val_benign, device, error_signal)  # benign → stats
    Et = _per_feature_resid(model, X_test, device, error_signal)
    mu, var = Ev.mean(0), Ev.var(0) + 1e-8
    w = (feat_w.detach().cpu().numpy().ravel() if feat_w is not None
         else np.ones(Et.shape[1]))
    Sigma = np.cov(Ev, rowvar=False) + 1e-6 * np.eye(Ev.shape[1])
    Sinv = np.linalg.pinv(Sigma)
    d = Et - mu
    scores = {
        "raw_orig(.7max+.3mean)": pred_orig,
        "raw_mean":               Et.mean(1),
        "weighted(KDE)":          (Et * w).sum(1),
        "standardized(÷benvar)":  (Et / var).mean(1),
        "mahalanobis(bencov)":    np.einsum("nf,fg,ng->n", d, Sinv, d),
        "sphere (reference)":     sphere,
    }
    holdout = np.isin(y_family, HOLDOUT_FAMILIES)
    emask = (y_family == 0) | holdout
    yh = (y_family[emask] != 0).astype(int)
    print("\n" + "=" * 72)
    print("  PRED READOUT COMPARISON  (pred_only signal on test; sphere = reference)")
    print("=" * 72)
    print(f"  {'readout':<26}{'AUPRC':>9}{'AUROC':>9}{'ho_AUPRC':>10}")
    print("  " + "-" * 54)
    for name, s in scores.items():
        ap = average_precision_score(y_attack, s)
        au = roc_auc_score(y_attack, s)
        hp = (average_precision_score(yh, s[emask])
              if len(np.unique(yh)) >= 2 else float("nan"))
        print(f"  {name:<26}{ap:>9.4f}{au:>9.4f}{hp:>10.4f}")
    print("=" * 72)


def score_with_fusion(
    pred_scores: np.ndarray,
    sphere_scores: np.ndarray,
    score_cfg: Dict[str, Any],
) -> np.ndarray:
    method = score_cfg["method"]
    if method == "alpha":
        alpha = float(score_cfg["alpha"])
        return (alpha * pred_scores + (1 - alpha) * sphere_scores).astype(np.float32, copy=False)
    if method == "logreg":
        X = build_fusion_features(pred_scores, sphere_scores)
        X_scaled = score_cfg["feature_scaler"].transform(X)
        return score_cfg["classifier"].decision_function(X_scaled).astype(np.float32, copy=False)
    raise ValueError(f"Unknown score fusion method: {method}")


def summarize_fusion_cfg(score_cfg: Dict[str, Any]) -> Dict[str, Any]:
    method = score_cfg["method"]
    if method == "alpha":
        return {
            "method": "alpha",
            "alpha": float(score_cfg["alpha"]),
            "val_auroc": float(score_cfg["val_auroc"]),
            "val_auprc": float(score_cfg["val_auprc"]),
        }
    if method == "logreg":
        clf = score_cfg["classifier"]
        return {
            "method": "logreg",
            "val_auroc": float(score_cfg["val_auroc"]),
            "val_auprc": float(score_cfg["val_auprc"]),
            "coef": clf.coef_.astype(np.float64).ravel().tolist(),
            "intercept": clf.intercept_.astype(np.float64).ravel().tolist(),
        }
    return {"method": method}


def fit_learned_fusion(
    y_attack: np.ndarray,
    pred_scores: np.ndarray,
    sphere_scores: np.ndarray,
) -> Optional[Dict[str, Any]]:
    y_attack = np.asarray(y_attack, dtype=np.int64)
    n_pos = int(y_attack.sum())
    n_neg = int(len(y_attack) - n_pos)
    if n_pos < 20 or n_neg < 20:
        return None

    X = build_fusion_features(pred_scores, sphere_scores)
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)
    classifier = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED,
    )
    classifier.fit(X_scaled, y_attack)
    fused_scores = classifier.decision_function(X_scaled).astype(np.float32, copy=False)
    return {
        "method": "logreg",
        "feature_scaler": feature_scaler,
        "classifier": classifier,
        "val_auroc": float(roc_auc_score(y_attack, fused_scores)),
        "val_auprc": float(average_precision_score(y_attack, fused_scores)),
    }


def select_score_fusion(
    y_attack: np.ndarray,
    pred_scores: np.ndarray,
    sphere_scores: np.ndarray,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_alpha_auprc, best_alpha = -1.0, 0.5
    for a in alphas:
        s_val = a * pred_scores + (1 - a) * sphere_scores
        auprc = average_precision_score(y_attack, s_val)
        auroc = roc_auc_score(y_attack, s_val)
        if auprc > best_alpha_auprc:
            best_alpha_auprc, best_alpha = float(auprc), float(a)
        if verbose:
            marker = "  ← best" if a == best_alpha else ""
            print(f"  α={a:.1f}  val_AUROC={auroc:.6f}  val_AUPRC={auprc:.6f}{marker}")

    alpha_cfg = {
        "method": "alpha",
        "alpha": float(best_alpha),
        "val_auroc": float(roc_auc_score(y_attack, best_alpha * pred_scores + (1 - best_alpha) * sphere_scores)),
        "val_auprc": float(best_alpha_auprc),
    }
    candidate_cfgs = [alpha_cfg]
    if verbose:
        print(f"\n  Best α = {best_alpha:.1f}  (val AUPRC={best_alpha_auprc:.6f})")

    if ENABLE_LEARNED_FUSION:
        learned_cfg = fit_learned_fusion(y_attack, pred_scores, sphere_scores)
        if learned_cfg is not None:
            candidate_cfgs.append(learned_cfg)
            if verbose:
                learned_coef = learned_cfg["classifier"].coef_.astype(np.float64).ravel()
                learned_intercept = float(learned_cfg["classifier"].intercept_.astype(np.float64).ravel()[0])
                print(
                    f"  Learned fusion (logreg)  val_AUROC={learned_cfg['val_auroc']:.6f}  "
                    f"val_AUPRC={learned_cfg['val_auprc']:.6f}"
                )
                print(
                    "    logreg coef [log1p(pred), log1p(sphere)] "
                    f"= [{learned_coef[0]:.6f}, {learned_coef[1]:.6f}]  "
                    f"intercept={learned_intercept:.6f}"
                )
        elif verbose:
            print("  Learned fusion skipped: insufficient mixed-validation positives/negatives.")

    score_cfg = max(
        candidate_cfgs,
        key=lambda cfg: (float(cfg["val_auprc"]), float(cfg["val_auroc"])),
    )
    fusion_summary = summarize_fusion_cfg(score_cfg)
    if verbose:
        if score_cfg["method"] == "alpha":
            print(f"  Selected fusion      : alpha (α={score_cfg['alpha']:.1f})")
        else:
            print("  Selected fusion      : learned logreg over [log1p(pred), log1p(sphere)]")
    return score_cfg, fusion_summary, float(best_alpha)


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

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
    return float(np.median(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize_binary_signal(name: str, y_attack: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    y_attack = np.asarray(y_attack, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    benign = scores[y_attack == 0]
    attack = scores[y_attack == 1]
    if len(benign) == 0 or len(attack) == 0:
        return {
            "name": name,
            "n_benign": int(len(benign)),
            "n_attack": int(len(attack)),
            "auroc": None,
            "auprc": None,
        }

    benign_mean = float(np.mean(benign))
    attack_mean = float(np.mean(attack))
    benign_std = float(np.std(benign))
    attack_std = float(np.std(attack))
    pooled_std = math.sqrt((benign_std**2 + attack_std**2) / 2.0)
    return {
        "name": name,
        "n_benign": int(len(benign)),
        "n_attack": int(len(attack)),
        "auroc": float(roc_auc_score(y_attack, scores)),
        "auprc": float(average_precision_score(y_attack, scores)),
        "benign_mean": benign_mean,
        "attack_mean": attack_mean,
        "benign_median": float(np.median(benign)),
        "attack_median": float(np.median(attack)),
        "benign_p90": float(np.percentile(benign, 90)),
        "attack_p10": float(np.percentile(attack, 10)),
        "attack_to_benign_mean_ratio": float(attack_mean / (benign_mean + 1e-12)),
        "median_gap": float(np.median(attack) - np.median(benign)),
        "cohens_d": float((attack_mean - benign_mean) / (pooled_std + 1e-12)),
    }


def print_signal_summary(summary: Dict[str, Any]):
    print(f"\n  {summary['name']}")
    print("  " + "-" * len(summary["name"]))
    if summary.get("auprc") is None:
        print(
            f"  skipped: need both classes "
            f"(benign={summary['n_benign']}, attack={summary['n_attack']})"
        )
        return
    print(f"  AUROC={summary['auroc']:.6f}  AUPRC={summary['auprc']:.6f}")
    print(
        f"  mean    benign={summary['benign_mean']:.6e}  "
        f"attack={summary['attack_mean']:.6e}  "
        f"ratio={summary['attack_to_benign_mean_ratio']:.3f}"
    )
    print(
        f"  median  benign={summary['benign_median']:.6e}  "
        f"attack={summary['attack_median']:.6e}  "
        f"gap={summary['median_gap']:.6e}"
    )
    print(
        f"  overlap benign_p90={summary['benign_p90']:.6e}  "
        f"attack_p10={summary['attack_p10']:.6e}  "
        f"cohens_d={summary['cohens_d']:.3f}"
    )


def evaluate_model(name, scores_test, y_attack, y_family):
    """Full evaluation suite (window-level) with TPR@FPR metrics."""
    res = {"model": name}
    res["auroc_overall"] = float(roc_auc_score(y_attack, scores_test))
    res["auprc_overall"] = float(average_precision_score(y_attack, scores_test))

    # ── TPR @ FPR operating points ──
    fpr_arr, tpr_arr, _ = roc_curve(y_attack, scores_test)
    tpr_at_fpr = {}
    for target_fpr in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        idx = np.searchsorted(fpr_arr, target_fpr, side="right") - 1
        idx = max(0, min(idx, len(tpr_arr) - 1))
        tpr_at_fpr[f"TPR@FPR={target_fpr:.1%}"] = float(tpr_arr[idx])
    res["tpr_at_fpr"] = tpr_at_fpr

    # Holdout families
    holdout_mask = np.isin(y_family, HOLDOUT_FAMILIES)
    eval_mask = (y_family == 0) | holdout_mask
    y_h = (y_family[eval_mask] != 0).astype(int)
    s_h = scores_test[eval_mask]
    if len(np.unique(y_h)) >= 2:
        auprc_h, ci_lo, ci_hi = bootstrap_auprc(y_h, s_h)
        # Holdout TPR@FPR
        fpr_h, tpr_h, _ = roc_curve(y_h, s_h)
        holdout_tpr_at_fpr = {}
        for target_fpr in [0.01, 0.05, 0.10]:
            idx = np.searchsorted(fpr_h, target_fpr, side="right") - 1
            idx = max(0, min(idx, len(tpr_h) - 1))
            holdout_tpr_at_fpr[f"TPR@FPR={target_fpr:.0%}"] = float(tpr_h[idx])
        res["holdout_tpr_at_fpr"] = holdout_tpr_at_fpr
    else:
        auprc_h, ci_lo, ci_hi = 0.0, 0.0, 0.0
        res["holdout_tpr_at_fpr"] = {}
    res["auprc_holdout"] = auprc_h
    res["holdout_ci_low"] = ci_lo
    res["holdout_ci_high"] = ci_hi

    # Direction
    benign_mask = (y_family == 0)
    med_b = float(np.median(scores_test[benign_mask])) if benign_mask.any() else 0
    med_a = float(np.median(scores_test[~benign_mask])) if (~benign_mask).any() else 0
    res["median_benign"] = med_b
    res["median_attack"] = med_a
    res["direction_correct"] = bool(med_a >= med_b)

    # Per-family stats
    per_fam = []
    for fam in sorted(np.unique(y_family)):
        m = (y_family == fam)
        s = scores_test[m]
        per_fam.append({
            "name": CIC_LABEL_MAP.get(fam, f"fam_{fam}"),
            "n": int(m.sum()),
            "mean": float(s.mean()), "median": float(np.median(s)),
            "std": float(s.std()),
            "p90": float(np.percentile(s, 90)),
            "p99": float(np.percentile(s, 99)),
        })
    res["per_family"] = per_fam
    return res


def print_result(res):
    print(f"\n{'='*80}")
    print(f"  {res['model']}")
    print(f"{'='*80}")
    print(f"  AUROC (overall)     : {res['auroc_overall']:.6f}")
    print(f"  AUPRC (overall)     : {res['auprc_overall']:.6f}")
    print(f"  AUPRC (holdout 8+9) : {res['auprc_holdout']:.6f}  "
          f"95% CI [{res['holdout_ci_low']:.6f}, {res['holdout_ci_high']:.6f}]")
    d = "✓ CORRECT" if res["direction_correct"] else "✗ INVERTED"
    print(f"  Score direction     : {d}  "
          f"(benign={res['median_benign']:.2e}, attack={res['median_attack']:.2e})")

    # TPR@FPR operating points
    print("\n  TPR @ FPR operating points (overall):")
    for k, v in res.get("tpr_at_fpr", {}).items():
        print(f"    {k:<20s} : {v:.4f}")
    if res.get("holdout_tpr_at_fpr"):
        print("\n  TPR @ FPR operating points (holdout 8+9):")
        for k, v in res["holdout_tpr_at_fpr"].items():
            print(f"    {k:<20s} : {v:.4f}")

    print(f"\n  {'family':<20s} {'n':>7s} {'mean':>10s} {'median':>10s} "
          f"{'std':>10s} {'p90':>10s} {'p99':>10s}")
    print("  " + "-" * 79)
    for row in res["per_family"]:
        print(f"  {row['name']:<20s} {row['n']:>7d} {row['mean']:>10.2e} "
              f"{row['median']:>10.2e} {row['std']:>10.2e} "
              f"{row['p90']:>10.2e} {row['p99']:>10.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(all_results, models_info, y_test_attack, y_test_family, output_dir):
    """Generate ROC, PR, score distribution, and per-family boxplots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ["#e74c3c"]

    ax = axes[0, 0]
    for i, (name, scores) in enumerate(models_info):
        fpr, tpr, _ = roc_curve(y_test_attack, scores)
        auroc = all_results[i]["auroc_overall"]
        ax.plot(fpr, tpr, color=colors[i], label=f"{name} (AUROC={auroc:.4f})")
    ax.plot([0,1],[0,1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("(a) ROC Curve"); ax.legend()

    ax = axes[0, 1]
    for i, (name, scores) in enumerate(models_info):
        prec, rec, _ = precision_recall_curve(y_test_attack, scores)
        auprc = all_results[i]["auprc_overall"]
        ax.plot(rec, prec, color=colors[i], label=f"{name} (AUPRC={auprc:.4f})")
    rate = y_test_attack.mean()
    ax.axhline(rate, color="gray", ls="--", alpha=0.5, label=f"Random (rate={rate:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("(b) Precision-Recall Curve"); ax.legend()

    ax = axes[1, 0]
    name, scores = models_info[0]
    benign_s = scores[y_test_attack == 0]
    attack_s = scores[y_test_attack == 1]
    ax.hist(benign_s, bins=100, alpha=0.6, density=True, color="green", label="Benign")
    ax.hist(attack_s, bins=100, alpha=0.6, density=True, color="red", label="Attack")
    ax.set_title(f"(c) Score Distribution — {name}")
    ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Density"); ax.legend()

    ax = axes[1, 1]
    families = sorted(np.unique(y_test_family))
    fam_data = [scores[y_test_family == f] for f in families]
    fam_labels = [f"{CIC_LABEL_MAP.get(f,'?')}\n(n={len(scores[y_test_family==f])})"
                  for f in families]
    bp = ax.boxplot(fam_data, labels=fam_labels, patch_artist=True,
                    showfliers=False, medianprops={"color":"black"})
    for patch in bp["boxes"]:
        patch.set_facecolor("#74b9ff")
    ax.set_title(f"(d) Per-Family Window Scores — {name}")
    ax.set_ylabel("Anomaly Score")
    plt.xticks(rotation=45, ha="right")

    plt.suptitle("Sequence-Window Transformer-AE Results", fontsize=16, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "ablation_outputs_results.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved plot to {output_dir}/ablation_outputs_results.png")


def plot_kde_heatmap(kde_results, output_dir, top_n=10):
    """Heatmap of top anomalous features per attack family."""
    families = list(kde_results.keys())
    all_features = set()
    for fam_data in kde_results.values():
        for entry in fam_data[:top_n]:
            all_features.add(entry["feature"])
    features = sorted(all_features)

    matrix = np.zeros((len(families), len(features)))
    for i, fam in enumerate(families):
        scores_dict = {e["feature"]: e["mean_kde_score"] for e in kde_results[fam]}
        for j, feat in enumerate(features):
            matrix[i, j] = scores_dict.get(feat, 0)

    fig, ax = plt.subplots(figsize=(max(12, len(features)*0.8), max(6, len(families)*0.6)))
    sns.heatmap(matrix, xticklabels=features, yticklabels=families,
                cmap="YlOrRd", annot=True, fmt=".1f", ax=ax)
    ax.set_title("KDE Feature Anomaly Scores by Attack Family")
    ax.set_xlabel("Feature"); ax.set_ylabel("Attack Family")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "kde_explainability_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved KDE heatmap to {output_dir}/kde_explainability_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(run_seed, ablation_key="S1"):
    global SEED, OUTPUT_DIR, USE_DELTA_FEATURES, LAMBDA_VAR, LAMBDA_COV
    if ablation_key not in ABLATIONS:
        raise ValueError(f"Unknown ablation key: {ablation_key}")
    ablation = ABLATIONS[ablation_key]
    SEED = run_seed
    USE_DELTA_FEATURES = bool(ablation["use_delta_features"])
    LAMBDA_VAR = float(ablation["lambda_var"])
    LAMBDA_COV = float(ablation["lambda_cov"])
    OUTPUT_DIR = os.path.join(
        BASE_DIR, "ablation_outputs", ablation_key,
        f"{ablation['name']}_seed_{SEED}",
    )
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ablation: {ablation_key} — {ablation['name']}")
    print(f"  {ablation['description']}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data loading and preprocessing
    print("\n" + "="*80)
    print("  Data Loading & Windowing")
    print("="*80)
    (X_train_w, X_val_w, X_val_mixed_w, y_val_mixed_w,
     X_test_w, y_test_attack_w, y_test_family_w,
     X_train_flows, feature_names, _, _) = load_and_preprocess()

    n_features = X_train_w.shape[2]  # F

    # KDE fitting on individual training flows
    print("\n" + "="*80)
    print("  KDE Explainability — Fitting")
    print("="*80)
    kde = FeatureKDE()
    kde.fit(X_train_flows, feature_names)

    kde_weights_np = kde.compute_feature_weights()
    # Extend weights for delta features (uniform weight = 1.0)
    if USE_DELTA_FEATURES:
        delta_weights = np.ones_like(kde_weights_np)
        kde_weights_np_full = np.concatenate([kde_weights_np, delta_weights])
    else:
        kde_weights_np_full = kde_weights_np
    kde_weights = torch.tensor(kde_weights_np_full, dtype=torch.float32).to(device)
    print(f"  KDE feature weights: min={kde_weights_np.min():.3f}, "
          f"max={kde_weights_np.max():.3f}, mean={kde_weights_np.mean():.3f}"
          f"{' (+delta uniform)' if USE_DELTA_FEATURES else ''}")

    # Model construction
    print("\n" + "="*80)
    print("  Model Construction")
    print("="*80)
    model = SequenceTransformerAE(
        n_features=n_features, window_size=WINDOW_SIZE,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, dropout=DROPOUT, bottleneck_dim=BOTTLENECK_DIM,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Window: {WINDOW_SIZE} flows × {n_features} features")
    print(f"  Bottleneck: {BOTTLENECK_DIM}-dim")

    # Dataloaders
    train_loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "drop_last": True,
    }
    if FIX_TRAIN_SHUFFLE_ORDER:
        train_generator = torch.Generator()
        train_generator.manual_seed(SEED)
        train_loader_kwargs["generator"] = train_generator
    # Keep window tensors resident on the GPU (num_workers stays 0, so the
    # sampler draws identical indices from the same RNG — batches become plain
    # GPU slices and the per-batch .to(device) below is a no-op). Bit-identical
    # to the CPU-fed path, but removes the input-pipeline stall.
    train_tensor = torch.tensor(X_train_w)
    val_tensor = torch.tensor(X_val_w)
    if GPU_RESIDENT_DATA and device.type == "cuda":
        train_tensor = train_tensor.to(device)
        val_tensor = val_tensor.to(device)
    train_loader = DataLoader(
        TensorDataset(train_tensor), **train_loader_kwargs)
    val_loader = DataLoader(
        TensorDataset(val_tensor), batch_size=BATCH_SIZE, shuffle=False)

    # Center initialization, objective construction, and training
    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint.pt")

    if SKIP_TRAINING:
        print("\n" + "="*80)
        print("  Loading from checkpoint (SKIP_TRAINING=True)")
        print("="*80)
        # Own training checkpoints include history/metrics with NumPy scalar types.
        # PyTorch 2.6 defaults torch.load to weights_only=True, which rejects them.
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        center = ckpt["center"].to(device)
        history = ckpt.get("history", [])
        best_val_auroc = ckpt.get("best_val_auroc", 0)
        print(f"  ✓ Loaded epoch {ckpt['epoch']}, val_AUROC={best_val_auroc:.6f}")
        print(f"  Checkpoint ||center|| = {center.norm():.4f}")
        if RECOMPUTE_CENTER_ON_LOAD:
            center = init_center(model, train_loader, device)
            print(f"  Recomputed ||center|| = {center.norm():.4f}")
    else:
        loss_fn = PredictionLoss(lambda_var=LAMBDA_VAR, min_variance=MIN_VARIANCE,
                                 lambda_cov=LAMBDA_COV,
                                 feature_weights=kde_weights,
                                 error_signal=ablation["error_signal"],
                                 cov_target=ablation["cov_target"],
                                 train_svdd=ablation["train_svdd"],
                                 lambda_svdd=LAMBDA_SVDD).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        # Linear warmup for 5 epochs, then cosine decay
        warmup_epochs = 5
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        print("\n" + "="*80)
        train_mode = "SVDD sphere training" if ablation["train_svdd"] else "decoupled AE training"
        print(f"  Training ({train_mode})")
        print(f"  Error signal: {ablation['error_signal']}  "
              f"delta_features={USE_DELTA_FEATURES}  cov_target={ablation['cov_target']}")
        print(f"  Masking: {MASK_RATIO:.0%}, λ_var={LAMBDA_VAR}, min_var={MIN_VARIANCE}, "
              f"λ_cov={LAMBDA_COV} (warmup={COV_WARMUP}), λ_svdd={LAMBDA_SVDD}")
        print(f"  Contrastive: λ={LAMBDA_CONTRAST}, margin={CONTRAST_MARGIN}, "
              f"warmup={CONTRAST_WARMUP} epochs")
        print("="*80)
        history = []
        # Record validation and test signals at every epoch without applying
        # fusion or early stopping; selection is performed from the saved trajectory.
        traj_val_pred, traj_val_sphere = [], []
        traj_test_pred, traj_test_sphere = [], []
        # Raw euclid sphere = SVDD's NATIVE (scale-sensitive) readout. SPHERE_READOUT="maha"
        # whitens the latent by benign cov, which structurally cancels SVDD's isotropic radius
        # shrink. Dump the Euclidean sphere as well for the reported latent-geometry
        # audit. zgeom tracks the benign latent radius per epoch.
        traj_val_sphere_eu, traj_test_sphere_eu = [], []
        traj_zgeom = []   # per epoch: (trace cov(z), mean‖z−c‖²) on benign val (X_val_w)

        # Initial center for val scoring (will be recomputed post-training)
        center = init_center(model, train_loader, device)

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()

            # Covariance warmup: ramp λ_cov linearly over COV_WARMUP epochs
            if COV_WARMUP > 0 and epoch <= COV_WARMUP:
                loss_fn.lambda_cov = LAMBDA_COV * (epoch / COV_WARMUP)
            else:
                loss_fn.lambda_cov = LAMBDA_COV

            train_center = center if loss_fn.train_svdd else None

            # Contrastive loss activates after warmup
            lc = LAMBDA_CONTRAST if epoch > CONTRAST_WARMUP else 0.0
            tm = train_one_epoch(model, train_loader, optimizer, loss_fn,
                                 device, mask_ratio=MASK_RATIO,
                                 center=train_center, lambda_contrast=lc,
                                 contrast_margin=CONTRAST_MARGIN)
            vm = validate(model, val_loader, loss_fn, device, center=train_center)
            scheduler.step()
            elapsed = time.time() - t0

            # Recompute center periodically so all conditions use the same schedule.
            if epoch % 5 == 0:
                center = init_center(model, train_loader, device)

            # ── Per-epoch signals on val_mixed AND test (no fusion here) ──
            # pred readout per PRED_READOUT; benign stats fit on X_val_w (pure benign) only.
            es = ablation["error_signal"]
            if PRED_READOUT == "raw":
                ep_val_pred, ep_val_sphere = compute_raw_signals(
                    model, X_val_mixed_w, center, device, batch_size=256, error_signal=es)
                ep_test_pred, ep_test_sphere = compute_raw_signals(
                    model, X_test_w, center, device, batch_size=256, error_signal=es)
                # raw branch already reads sphere as euclid ‖z−c‖²; mirror it, no z-geometry.
                ep_val_sphere_eu, ep_test_sphere_eu = ep_val_sphere, ep_test_sphere
                ep_zgeom = (float("nan"), float("nan"))
            else:
                # Fit benign (X_val_w) stats for BOTH calibrated readouts; no leakage.
                E_b, Z_b = _resid_and_z(model, X_val_w, device, es)
                pred_stats = {"var": E_b.var(0) + 1e-8, "mu": E_b.mean(0)}
                if PRED_READOUT == "maha":
                    pred_stats["Sinv"] = np.linalg.pinv(
                        np.cov(E_b, rowvar=False) + 1e-6 * np.eye(E_b.shape[1]))
                z_stats = {"mu": Z_b.mean(0),
                           "Sinv": np.linalg.pinv(
                               np.cov(Z_b, rowvar=False) + 1e-6 * np.eye(Z_b.shape[1]))}
                center_np = center.detach().cpu().numpy()

                E_vm, Z_vm = _resid_and_z(model, X_val_mixed_w, device, es)
                E_te, Z_te = _resid_and_z(model, X_test_w, device, es)
                ep_val_pred = _apply_readout(E_vm, pred_stats, PRED_READOUT).astype(np.float32)
                ep_test_pred = _apply_readout(E_te, pred_stats, PRED_READOUT).astype(np.float32)
                ep_val_sphere = _sphere_readout(Z_vm, center_np, z_stats, SPHERE_READOUT).astype(np.float32)
                ep_test_sphere = _sphere_readout(Z_te, center_np, z_stats, SPHERE_READOUT).astype(np.float32)
                # SVDD's native readout: raw euclid ‖z−c‖² (uncalibrated, scale-sensitive).
                ep_val_sphere_eu = _sphere_readout(Z_vm, center_np, z_stats, "euclid").astype(np.float32)
                ep_test_sphere_eu = _sphere_readout(Z_te, center_np, z_stats, "euclid").astype(np.float32)
                _zc = Z_b - center_np
                ep_zgeom = (float(Z_b.var(0).sum()), float((_zc * _zc).sum(1).mean()))
            traj_val_pred.append(ep_val_pred.astype(np.float32))
            traj_val_sphere.append(ep_val_sphere.astype(np.float32))
            traj_test_pred.append(ep_test_pred.astype(np.float32))
            traj_test_sphere.append(ep_test_sphere.astype(np.float32))
            traj_val_sphere_eu.append(ep_val_sphere_eu.astype(np.float32))
            traj_test_sphere_eu.append(ep_test_sphere_eu.astype(np.float32))
            traj_zgeom.append(ep_zgeom)

            # Console diagnostic ONLY (not used to select or stop): the stronger
            # single-signal AUROC, so you can watch discrimination evolve / collapse.
            val_auroc_pred = roc_auc_score(y_val_mixed_w, ep_val_pred)
            val_auroc_sphere = roc_auc_score(y_val_mixed_w, ep_val_sphere)
            val_auroc = max(val_auroc_pred, val_auroc_sphere)

            history.append({"epoch": epoch,
                            **{f"train_{k}": v for k,v in tm.items()},
                            **{f"val_{k}": v for k,v in vm.items()},
                            "val_auroc": val_auroc,
                            "val_auroc_pred": val_auroc_pred,
                            "val_auroc_sphere": val_auroc_sphere,
                            "lr": optimizer.param_groups[0]["lr"], "time_s": elapsed})

            if epoch <= 5 or epoch % 10 == 0 or epoch == EPOCHS:
                print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                      f"train={tm['total']:.6f}  val={vm['total']:.6f}  "
                      f"pred={vm['pred']:.6f}  var={vm['var']:.6f}  "
                      f"cov={vm['cov']:.6f}  svdd={vm['svdd']:.6f}  "
                      f"ctr={tm['contrast']:.6f}  "
                      f"AUROC[pred={val_auroc_pred:.4f} sph={val_auroc_sphere:.4f}]  "
                      f"({elapsed:.1f}s)")

        # Full schedule trained — NO early stopping. The best (epoch, fusion) is
        # chosen later from the stored trajectory, not here.
        best_val_auroc = max((h["val_auroc"] for h in history), default=0.0)
        center = init_center(model, train_loader, device)
        torch.save({"model": model.state_dict(), "center": center.cpu(),
                    "epoch": EPOCHS, "best_val_auroc": best_val_auroc,
                    "history": history}, ckpt_path)
        print(f"\n  Trained {EPOCHS} epochs (no early stopping; selection deferred).")
        print(f"  Best monitored single-signal val_AUROC = {best_val_auroc:.6f}")
        print(f"  Post-training ||center|| = {center.norm():.4f}")

        # Save the per-epoch signal trajectory for epoch/fusion selection.
        traj_path = os.path.join(OUTPUT_DIR, f"signal_trajectory_{ablation_key}.npz")
        np.savez(
            traj_path,
            epochs=np.arange(1, len(traj_val_pred) + 1, dtype=np.int32),
            val_pred=np.stack(traj_val_pred), val_sphere=np.stack(traj_val_sphere),
            test_pred=np.stack(traj_test_pred), test_sphere=np.stack(traj_test_sphere),
            val_sphere_eu=np.stack(traj_val_sphere_eu),
            test_sphere_eu=np.stack(traj_test_sphere_eu),
            zgeom=np.asarray(traj_zgeom, dtype=np.float32),
            y_val=y_val_mixed_w.astype(np.int8), y_test=y_test_attack_w.astype(np.int8),
            y_test_family=y_test_family_w.astype(np.int8),
        )
        print(f"  Dumped signal trajectory ({len(traj_val_pred)} epochs) -> {traj_path}")

        if TRAJECTORY_ONLY and not COMPARE_READOUTS:
            print("\n  TRAJECTORY_ONLY=True: skipping optional in-pipeline fusion and evaluation.")
            print("  Use the stored trajectory for epoch/fusion selection.")
            return {
                "ablation": {"key": ablation_key, "name": ablation["name"]},
                "trajectory_path": traj_path,
                "best_val_auroc": best_val_auroc,
            }

    if COMPARE_READOUTS:
        print(f"\n  [{ablation_key}] readout study")
        compare_pred_readouts(
            model, X_val_w, X_test_w, y_test_attack_w, y_test_family_w,
            center, device, ablation["error_signal"], feat_w=kde_weights)
        return {
            "ablation": {"key": ablation_key, "name": ablation["name"]},
            "readout_comparison": True,
        }

    # Fusion selection on mixed validation data (no test-set leakage)
    # Raw alpha blending is retained as a fallback candidate.
    print("\n" + "="*80)
    print("  Anomaly Scoring (fusion selection on val_mixed)")
    print("="*80)

    val_pred, val_sphere = compute_raw_signals(
        model, X_val_mixed_w, center, device, batch_size=256,
        error_signal=ablation["error_signal"])
    score_cfg, fusion_summary, _ = select_score_fusion(
        y_val_mixed_w, val_pred, val_sphere, verbose=True)

    # Apply val-chosen fusion to score test set (unbiased evaluation)
    test_pred, test_sphere = compute_raw_signals(
        model, X_test_w, center, device, batch_size=256,
        error_signal=ablation["error_signal"])
    # Store raw signals so standalone fusion analyses do not rerun the model.
    fusion_signals_path = os.path.join(OUTPUT_DIR, f"fusion_signals_{ablation_key}.npz")
    np.savez(
        fusion_signals_path,
        val_pred=val_pred, val_sphere=val_sphere, y_val=y_val_mixed_w,
        test_pred=test_pred, test_sphere=test_sphere, y_test=y_test_attack_w,
    )
    print(f"  Dumped raw fusion signals -> {fusion_signals_path}")

    sphere_signal_summary = {
        "val_mixed": summarize_binary_signal("Sphere-only signal on val_mixed", y_val_mixed_w, val_sphere),
        "test": summarize_binary_signal("Sphere-only signal on test", y_test_attack_w, test_sphere),
    }
    print_signal_summary(sphere_signal_summary["val_mixed"])
    print_signal_summary(sphere_signal_summary["test"])
    scores_val_mixed = score_with_fusion(val_pred, val_sphere, score_cfg)
    scores_test = score_with_fusion(test_pred, test_sphere, score_cfg)
    test_auprc = average_precision_score(y_test_attack_w, scores_test)
    test_auroc = roc_auc_score(y_test_attack_w, scores_test)
    print(
        f"  Test AUROC={test_auroc:.6f}  Test AUPRC={test_auprc:.6f}  "
        f"(fusion={fusion_summary['method']})"
    )

    # Final evaluation
    print("\n" + "="*80)
    print("  Evaluation")
    print("="*80)
    result = evaluate_model(f"SeqTransformerAE_{ablation_key}", scores_test,
                            y_test_attack_w, y_test_family_w)
    result["ablation"] = {
        "key": ablation_key,
        "name": ablation["name"],
        "description": ablation["description"],
        "use_delta_features": USE_DELTA_FEATURES,
        "error_signal": ablation["error_signal"],
        "train_svdd": bool(ablation["train_svdd"]),
        "lambda_var": LAMBDA_VAR,
        "lambda_cov": LAMBDA_COV,
        "lambda_svdd": LAMBDA_SVDD if ablation["train_svdd"] else 0.0,
        "cov_target": ablation["cov_target"],
    }
    result["fusion"] = fusion_summary
    result["sphere_signal"] = sphere_signal_summary
    print_result(result)

    # Threshold calibration
    print("\n" + "="*80)
    print("  Threshold calibration — data-driven threshold determination")
    print("="*80)
    print("  Calibrating threshold on mixed validation windows ...")

    prec_v, rec_v, thresholds_v = precision_recall_curve(y_val_mixed_w, scores_val_mixed)
    # F2 score: weights recall 2× higher than precision for better detection
    beta = 2.0
    fbeta_scores = ((1 + beta**2) * prec_v[:-1] * rec_v[:-1]) / \
                   (beta**2 * prec_v[:-1] + rec_v[:-1] + 1e-8)
    best_fb_idx = np.argmax(fbeta_scores)
    optimal_threshold = float(thresholds_v[best_fb_idx])
    best_fb = float(fbeta_scores[best_fb_idx])
    best_prec = float(prec_v[best_fb_idx])
    best_rec = float(rec_v[best_fb_idx])

    print(f"  Optimal threshold : {optimal_threshold:.6f}  (F2 score)")
    print(f"  Val F2            : {best_fb:.4f}")
    print(f"  Val Precision     : {best_prec:.4f}")
    print(f"  Val Recall        : {best_rec:.4f}")

    y_pred_test = (scores_test >= optimal_threshold).astype(int)
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, confusion_matrix
    )
    test_prec = precision_score(y_test_attack_w, y_pred_test, zero_division=0)
    test_rec = recall_score(y_test_attack_w, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test_attack_w, y_pred_test, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test_attack_w, y_pred_test).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n  Test set @ threshold={optimal_threshold:.6f}:")
    print(f"  Precision  : {test_prec:.4f}")
    print(f"  Recall     : {test_rec:.4f}")
    print(f"  F1         : {test_f1:.4f}")
    print(f"  FPR        : {fpr:.4f}")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    print("\n  Per-family detection rate @ threshold:")
    print(f"  {'Family':<20s} {'N':>7s} {'Detected':>10s} {'Rate':>8s}")
    print("  " + "-"*50)
    for fam in sorted(np.unique(y_test_family_w)):
        mask = (y_test_family_w == fam)
        n_fam = mask.sum()
        detected = y_pred_test[mask].sum()
        rate = detected / n_fam if n_fam > 0 else 0
        fname = CIC_LABEL_MAP.get(fam, f"fam_{fam}")
        marker = " (holdout)" if fam in HOLDOUT_FAMILIES else ""
        print(f"  {fname:<20s} {n_fam:>7d} {detected:>10d} {rate:>8.4f}{marker}")

    ddtd_result = {
        "optimal_threshold": optimal_threshold,
        "val_f2": best_fb, "val_precision": best_prec, "val_recall": best_rec,
        "test_f1": test_f1, "test_precision": test_prec, "test_recall": test_rec,
        "test_fpr": fpr, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    result["ddtd"] = ddtd_result

    # KDE explanations for flagged windows
    flagged_mask = y_pred_test == 1
    n_flagged = flagged_mask.sum()
    print(f"\n  Explaining {n_flagged} flagged windows with KDE...")
    # Use last flow in each flagged window as representative
    max_explain = 500
    flagged_idx = np.where(flagged_mask)[0]
    if len(flagged_idx) > max_explain:
        explain_rng = np.random.RandomState(SEED)
        explain_idx = explain_rng.choice(flagged_idx, max_explain, replace=False)
    else:
        explain_idx = flagged_idx
    X_explain_flows = X_test_w[explain_idx, -1, :]  # last flow per window
    explanations = kde.explain_samples(X_explain_flows, k=5)
    print("  Sample explanations (first 3):")
    for i, exp in enumerate(explanations[:3]):
        fam_label = CIC_LABEL_MAP.get(y_test_family_w[explain_idx[i]], "?")
        print(f"    Window {explain_idx[i]} (family={fam_label}, "
              f"score={scores_test[explain_idx[i]]:.4f}):")
        for feat in exp:
            print(f"      → {feat['feature']:<30s} kde={feat['kde_score']:.2f}")

    # ── Save results ──
    def deep_convert(obj):
        if isinstance(obj, dict): return {k: deep_convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [deep_convert(v) for v in obj]
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(deep_convert(result), f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Results + DDTD saved to {OUTPUT_DIR}/")

    # KDE explanation summary
    print("\n" + "="*80)
    print("  KDE Explainability Analysis")
    print("="*80)
    # Flatten test windows → individual flows for per-family KDE analysis
    X_test_flows_flat = X_test_w.reshape(-1, n_features)
    y_test_flows_family = np.repeat(y_test_family_w, WINDOW_SIZE)
    kde_results = kde.explain_family(X_test_flows_flat, y_test_flows_family, k=10)
    for fam, entries in kde_results.items():
        print(f"\n  {fam}:")
        for e in entries[:5]:
            print(f"    #{e['rank']} {e['feature']:<30s} score={e['mean_kde_score']:.2f}")

    # Visualizations
    print("\n" + "="*80)
    print("  Visualizations")
    print("="*80)
    models_info = [("SeqTransformerAE", scores_test)]
    plot_results([result], models_info, y_test_attack_w, y_test_family_w, OUTPUT_DIR)
    plot_kde_heatmap(kde_results, OUTPUT_DIR)

    # Save KDE results
    print("\n" + "="*80)
    print("  Saving KDE Results")
    print("="*80)
    with open(os.path.join(OUTPUT_DIR, "kde_explainability.json"), "w") as f:
        json.dump(deep_convert(kde_results), f, indent=2)
    print(f"  All results saved to {OUTPUT_DIR}/")
    print("\nDone!")
    return result

def main():
    all_results = []

    for ablation_key in ABLATIONS_TO_RUN:
        if ablation_key not in ABLATIONS:
            raise ValueError(f"Unknown ablation in ABLATIONS_TO_RUN: {ablation_key}")
        for s in RUN_SEEDS:
            print(f"\n\n{'='*80}\n  STARTING {ablation_key} WITH SEED {s}\n{'='*80}")
            res = run_experiment(s, ablation_key=ablation_key)
            all_results.append(res)

    if COMPARE_READOUTS:
        print(f"\n\n{'='*80}\n  READOUT STUDY — {len(all_results)} checkpoints re-scored "
              f"(see PRED READOUT COMPARISON tables above)\n{'='*80}")
        return

    if TRAJECTORY_ONLY:
        print(f"\n\n{'='*80}\n  TRAJECTORY_ONLY — {len(all_results)} runs dumped signal "
              f"trajectories\n{'='*80}")
        for r in all_results:
            print(f"  {r['ablation']['key']:<14}-> {r.get('trajectory_path', '(readout study)')}")
        print("\n  Next: use the stored trajectory for epoch/fusion selection")
        return

    print(f"\n\n{'='*80}\n  AGGREGATED RESULTS OVER {len(RUN_SEEDS)} SEEDS PER ABLATION\n{'='*80}")
    metrics = ['auroc_overall', 'auprc_overall', 'auprc_holdout']
    for ablation_key in ABLATIONS_TO_RUN:
        rows = [r for r in all_results if r["ablation"]["key"] == ablation_key]
        print(f"\n  {ablation_key} — {ABLATIONS[ablation_key]['name']}")
        for m in metrics:
            vals = [r[m] for r in rows]
            print(f"    {m:<20s}: {np.mean(vals):.6f} ± {np.std(vals):.6f}")

    # Save aggregated results to root base dir
    out_path = os.path.join(BASE_DIR, "ablation_outputs", "ablation_aggregated_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved aggregated results to {out_path}")

if __name__ == "__main__":
    main()
