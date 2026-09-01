# Adapted in part from THUML Time-Series-Library (TimesNet), MIT License.
# Source and full upstream notice: ../../THIRD_PARTY_NOTICES.md

"""
TimesNet CIC-UNSW-NB15 baseline -- proposed split + TimesNet's native processing.

Cloned from the native ModernTCN pipeline (same split, same native SegLoader-
style stream windowing, same reporting) with the model + training recipe
swapped for TimesNet as in thuml/Time-Series-Library anomaly_detection:

  Split (= proposed pipeline, time-split only): cutoff 2015-01-23, train = benign
  pre-cutoff (per-entity first 85%), val = benign pre-cutoff (last 15%),
  test = ALL post-cutoff flows; no val_mixed injection/removal.

  Processing (= TSLib native): NO entity grouping (continuous per-split
  streams), nan_to_num + StandardScaler, per-TIMESTEP reconstruction energy,
  percentile(train+test) threshold, point-adjustment.

  Recipe (= TSLib): Adam lr=1e-4 halved per epoch (lradj type1), 10 epochs,
  early stop patience=3 on val recon loss.

  Metrics: native point-adjusted P/R/F1 for the appendix, raw per-flow
  AUROC/AUPRC, and the flow-level score export consumed by the aggregation
  script. Point-adjusted metrics are not used in the main tables.

Fully self-contained: TimesNet model code is inlined below (identical to
the official TimesNet implementation); no sibling imports.
"""

from __future__ import annotations

import copy
import math
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

try:
    from google.colab import drive

    drive.mount("/content/drive")
except Exception:
    pass

# =============================================================================
# TimesNet architecture hyperparameters (must precede the model classes:
# TimesNetConfig dataclass defaults bind at class-definition time)
# =============================================================================

D_MODEL = 64
D_FF = 64
E_LAYERS = 2
TOP_K = 3
NUM_KERNELS = 6
DROPOUT = 0.1


# Inlined TimesNet anomaly-detection architecture from Time-Series-Library.
# The temporal embedding type is inactive because anomaly detection supplies
# no timestamp markers to ``enc_embedding``.

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model: int, embed_type: str = "fixed",
                 freq: str = "h"):
        super().__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = embed(minute_size, d_model)
        self.hour_embed = embed(hour_size, d_model)
        self.weekday_embed = embed(weekday_size, d_model)
        self.day_embed = embed(day_size, d_model)
        self.month_embed = embed(month_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, "minute_embed") else 0.0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model: int, embed_type: str = "timeF",
                 freq: str = "h"):
        super().__init__()
        freq_map = {
            "h": 4, "t": 5, "s": 6, "m": 1, "a": 1,
            "w": 2, "d": 3, "b": 3,
        }
        self.embed = nn.Linear(freq_map[freq], d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, embed_type: str = "fixed",
                 freq: str = "h", dropout: float = 0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        if embed_type != "timeF":
            self.temporal_embedding = TemporalEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor,
                x_mark: torch.Tensor | None) -> torch.Tensor:
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (self.value_embedding(x)
                 + self.temporal_embedding(x_mark)
                 + self.position_embedding(x))
        return self.dropout(x)


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 num_kernels: int = 6, init_weight: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(
                in_channels, out_channels, kernel_size=2 * i + 1,
                padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        return torch.stack(res_list, dim=-1).mean(-1)


def fft_for_period(x: torch.Tensor, k: int = 2) -> Tuple[np.ndarray, torch.Tensor]:
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


@dataclass
class TimesNetConfig:
    seq_len: int
    pred_len: int
    enc_in: int
    c_out: int
    embed: str = "fixed"
    freq: str = "h"
    d_model: int = D_MODEL
    d_ff: int = D_FF
    e_layers: int = E_LAYERS
    top_k: int = TOP_K
    num_kernels: int = NUM_KERNELS
    dropout: float = DROPOUT


class TimesBlock(nn.Module):
    def __init__(self, configs: TimesNetConfig):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time_steps, channels = x.size()
        period_list, period_weight = fft_for_period(x, self.k)

        res = []
        for period in period_list:
            total_len = self.seq_len + self.pred_len
            if total_len % period != 0:
                length = ((total_len // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], length - total_len, x.shape[2]],
                    device=x.device,
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_len
                out = x

            out = out.reshape(bsz, length // period, period, channels)
            out = out.permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(bsz, -1, channels)
            res.append(out[:, :total_len, :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(
            1, time_steps, channels, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x


class TimesNet(nn.Module):
    def __init__(self, configs: TimesNetConfig):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([
            TimesBlock(configs) for _ in range(configs.e_layers)
        ])
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq,
            configs.dropout)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True,
                                     unbiased=False) + 1e-5)
        x = x.div(stdev)

        enc_out = self.enc_embedding(x, None)
        for block in self.model:
            enc_out = self.layer_norm(block(enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
        return dec_out


class Model(nn.Module):
    """Standalone TimesNet anomaly-detection wrapper."""

    def __init__(self, configs):
        super().__init__()
        self.model = TimesNet(configs)

    def forward(self, x):
        return self.model(x)


# =============================================================================
# Shared constants (inlined; data split + labels match the proposed pipeline)
# =============================================================================

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

BASE_DIR = os.getenv("CIC_BASE_DIR", "/content/drive/MyDrive/15SOTA")
RAW_CSV = os.getenv("CIC_UNSW_RAW_CSV", "/content/drive/MyDrive/CIC-UNSW-NB15/CICFlowMeter_out.csv")

TRAIN_CUTOFF = pd.Timestamp("2015-01-23")
DROP_COLS = [
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Label",
]
ENTITY_COLS = ["Src IP"]

# Official PSM-style TimesNet training configuration. ``ANOMALY_RATIO`` only
# affects the appendix point-adjusted exhibit.
EPOCHS = 3
BATCH_SIZE = 128
LR = 1e-4
PATIENCE = 3
ANOMALY_RATIO = 0.5



# =============================================================================
# Shared helpers (inlined)
# =============================================================================

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


def make_timesnet_config(n_features):
    return TimesNetConfig(
        seq_len=WIN_SIZE,
        pred_len=0,
        enc_in=n_features,
        c_out=n_features,
    )


def make_loader(data, batch_size, shuffle=False, drop_last=False):
    # `data` may be a Dataset (lazy windows, see WindowDataset) or a raw ndarray
    # of pre-materialized windows (smoke test). Wrapping an ndarray here would
    # copy it into a tensor in full, so prefer passing a Dataset for real runs.
    dataset = data if isinstance(data, Dataset) else TensorDataset(
        torch.tensor(data, dtype=torch.float32)
    )
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


def adjust_learning_rate(optimizer, epoch):
    """TSLib lradj type1: lr = LR * 0.5 ** (epoch - 1)."""
    lr = LR * (0.5 ** ((epoch - 1) // 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def run_epoch(model, loader, optimizer, criterion, scheduler, device, log_every=200):
    model.train()
    losses = []
    n_batches = len(loader)
    # Heartbeat + crude data-wait vs compute split to diagnose loader starvation.
    data_wait = 0.0
    t_prev = time.time()
    for i, (batch_x,) in enumerate(loader, 1):
        data_wait += time.time() - t_prev
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_x)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
        if log_every and i % log_every == 0:
            print(f"    batch {i}/{n_batches} loss={np.mean(losses[-log_every:]):.6f} "
                  f"cum_data_wait={data_wait:.1f}s (high => GPU starved by loader)", flush=True)
        t_prev = time.time()
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        losses.append(criterion(outputs, batch_x).item())
    return float(np.mean(losses)) if losses else 0.0


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


def deep_convert(obj):
    if isinstance(obj, dict):
        return {k: deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_convert(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

# =============================================================================
# Configuration (native-pipeline specific)
# =============================================================================

# Fixed five-seed set reported in the manuscript.
RUN_SEEDS = (42, 456, 7, 789, 1024)
SEED = RUN_SEEDS[0]  # Active model seed; reassigned by run_experiment().

# All artifacts for this experiment live under ONE clearly-named folder.
# In a Colab notebook you can redirect everything in a single line BEFORE calling
# main() writes results below CIC_BASE_DIR; set that environment variable before running.
#   <RESULTS_DIR>/
#     seed_42/                 -> checkpoint.pth, results.json, history.json, test_energy.npy
#     aggregated_results.json  -> summary over all seeds
RESULTS_DIR = os.path.join(BASE_DIR, "results_timesnet_native")
OUTPUT_DIR = os.path.join(RESULTS_DIR, f"seed_{SEED}")

WIN_SIZE = 100      # TimesNet AD default; set 50 for tighter parity with proposed method
TRAIN_STEP = 1      # faithful PSM dense windows; raise to speed up training
SKIP_TRAINING = False


# =============================================================================
# Proposed split + native TimesNet preprocessing
# =============================================================================

def load_and_preprocess_native():
    """Proposed flow split with TimesNet-native cleaning/scaling, returned as
    continuous (entity-agnostic) per-split streams."""
    set_seed(SEED)

    print("Loading raw CICFlowMeter CSV ...")
    raw = pd.read_csv(RAW_CSV)
    print(f"  Raw shape: {raw.shape}")

    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], dayfirst=True)
    raw.sort_values("Timestamp", inplace=True)
    raw.reset_index(drop=True, inplace=True)

    label_str = raw["Label"].str.strip()
    y_all = label_str.map(LABEL_TO_ID).values.astype(int)
    y_bin = (y_all != 0).astype(int)
    print(f"  Timestamp range: {raw['Timestamp'].min()} -> {raw['Timestamp'].max()}")
    print(f"  Label distribution:\n{label_str.value_counts().to_string()}")

    # Entity ids — needed ONLY to reproduce the proposed per-entity 85/15 split.
    entity_str = raw[ENTITY_COLS].astype(str).agg("_".join, axis=1)
    entity_ids = pd.Categorical(entity_str).codes
    print(f"  Entities ({', '.join(ENTITY_COLS)}): {entity_ids.max() + 1} unique "
          f"(used only for the split, NOT for windowing)")

    # Features: drop metadata + entity cols, keep ALL remaining columns (native).
    data = raw.drop(columns=[c for c in DROP_COLS if c in raw.columns])
    for c in ENTITY_COLS:
        if c in data.columns:
            data = data.drop(columns=[c])
    feature_names = data.columns.tolist()
    print(f"  Features kept (no low-variance drop): {len(feature_names)}")

    # ── Split: identical to the proposed pipeline (time-split only, no val_mixed) ──
    ts = raw["Timestamp"].values
    train_period = ts < np.datetime64(TRAIN_CUTOFF)
    benign = y_all == 0

    train_indices, val_indices = [], []
    for eid in np.unique(entity_ids[train_period & benign]):
        eid_idx = np.where(train_period & benign & (entity_ids == eid))[0]
        split = int(len(eid_idx) * 0.85)
        train_indices.extend(eid_idx[:split].tolist())
        val_indices.extend(eid_idx[split:].tolist())
    train_indices = np.sort(np.array(train_indices, dtype=int))
    val_indices = np.sort(np.array(val_indices, dtype=int))
    test_indices = np.where(~train_period)[0]

    print(f"\n  Train: {len(train_indices)} benign flows (85%)")
    print(f"  Val:   {len(val_indices)} benign flows (15%)")
    print(f"  Test:  {len(test_indices)} all flows (attack rate="
          f"{y_bin[test_indices].mean():.4f})")

    # ── Native cleaning/scaling: nan_to_num + StandardScaler (fit on train) ──
    X_all = data.values.astype(np.float64)
    X_all[~np.isfinite(X_all)] = np.nan          # CIC infs -> nan (then -> 0 below)
    X_all = np.nan_to_num(X_all)                 # NaN -> 0, faithful to SegLoaders
    scaler = StandardScaler()
    scaler.fit(X_all[train_indices])
    X_scaled = scaler.transform(X_all).astype(np.float32)

    # Per-split continuous streams (time order preserved; entity-agnostic).
    train_stream = X_scaled[train_indices]
    val_stream = X_scaled[val_indices]
    test_stream = X_scaled[test_indices]
    y_test_bin = y_bin[test_indices]
    y_test_family = y_all[test_indices]

    return (train_stream, val_stream, test_stream,
            y_test_bin, y_test_family, feature_names, test_indices)


# =============================================================================
# Native windowing
# =============================================================================

def seg_windows(stream, win_size, step):
    """Sliding windows across a continuous stream -> (M, win_size, nvars).

    Matches the official SegLoader count: (n - win_size) // step + 1.

    WARNING: this MATERIALIZES every window as a copy, so dense (step=1) windows
    expand memory by ~win_size and OOM on large flow streams. The real pipeline
    uses `WindowDataset` (lazy, O(stream) memory); this helper is kept only for
    the small synthetic smoke test.
    """
    n = len(stream)
    if n < win_size:
        return np.zeros((0, win_size, stream.shape[1]), dtype=np.float32)
    starts = range(0, n - win_size + 1, step)
    windows = np.stack([stream[s:s + win_size] for s in starts])
    return windows.astype(np.float32)


class WindowDataset(Dataset):
    """Lazy sliding windows over a continuous stream.

    Holds only the base stream in memory and slices each (win_size, nvars)
    window on demand, so peak RAM is O(stream) instead of O(stream * win_size).
    Window count and order match `seg_windows` / the official SegLoader:
    (n - win_size) // step + 1.
    """

    def __init__(self, stream, win_size, step):
        # One contiguous float32 copy of the stream; windows are views into it.
        self.stream = torch.as_tensor(np.ascontiguousarray(stream), dtype=torch.float32)
        self.win_size = win_size
        self.step = step
        n = len(stream)
        self.n_windows = 0 if n < win_size else (n - win_size) // step + 1
        self.n_features = stream.shape[1]

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        s = idx * self.step
        return (self.stream[s:s + self.win_size],)

    @property
    def shape(self):
        return (self.n_windows, self.win_size, self.n_features)


# =============================================================================
# Scoring + threshold + point adjustment
# =============================================================================

@torch.no_grad()
def compute_timestep_energy(model, windows, device, batch_size=BATCH_SIZE):
    """Per-TIMESTEP reconstruction energy, flattened to 1-D.

    energy[t] = mean over features of MSE(x[t], recon[t]).
    Returns array of length n_windows * win_size (window-major order).
    """
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    loader = make_loader(windows, batch_size=batch_size, shuffle=False)
    energies = []
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        score = torch.mean(criterion(batch_x, outputs), dim=-1)   # (B, win_size)
        energies.append(score.detach().cpu().numpy().reshape(-1))
    if not energies:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(energies, axis=0)


def adjustment(gt, pred):
    """Anomaly-Transformer / TimesNet point-adjustment.

    If any point inside a ground-truth anomaly segment is detected, the whole
    segment is marked detected.
    """
    gt = np.asarray(gt).astype(int)
    pred = np.asarray(pred).astype(int).copy()
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if gt[j] == 0:
                    break
                if pred[j] == 0:
                    pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                if pred[j] == 0:
                    pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


# =============================================================================
# Reporting — both metric families
# =============================================================================

def _ranking_metrics(scores, y_bin, y_family):
    """Threshold-free ranking metrics: overall AUROC/AUPRC, TPR@FPR operating
    points, holdout (families 8+9 vs benign) AUPRC + bootstrap CI, and score
    direction. Used for BOTH per-flow and per-window granularities so the two are
    computed by identical code (mirrors the proposed evaluation protocol)."""
    scores = np.asarray(scores)
    y_bin = np.asarray(y_bin)
    y_family = np.asarray(y_family)
    out = {}
    out["auroc_overall"] = float(roc_auc_score(y_bin, scores))
    out["auprc_overall"] = float(average_precision_score(y_bin, scores))

    fpr_arr, tpr_arr, _ = roc_curve(y_bin, scores)
    tpr_at_fpr = {}
    for target in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        idx = np.searchsorted(fpr_arr, target, side="right") - 1
        idx = max(0, min(idx, len(tpr_arr) - 1))
        tpr_at_fpr[f"TPR@FPR={target:.1%}"] = float(tpr_arr[idx])
    out["tpr_at_fpr"] = tpr_at_fpr

    holdout_mask = np.isin(y_family, HOLDOUT_FAMILIES)
    eval_mask = (y_family == 0) | holdout_mask
    y_h = (y_family[eval_mask] != 0).astype(int)
    s_h = scores[eval_mask]
    if len(np.unique(y_h)) >= 2:
        auprc_h, ci_lo, ci_hi = bootstrap_auprc(y_h, s_h)
    else:
        auprc_h, ci_lo, ci_hi = 0.0, 0.0, 0.0
    out["auprc_holdout"] = auprc_h
    out["holdout_ci_low"] = ci_lo
    out["holdout_ci_high"] = ci_hi

    benign_mask = y_family == 0
    out["median_benign"] = float(np.median(scores[benign_mask])) if benign_mask.any() else 0.0
    out["median_attack"] = float(np.median(scores[~benign_mask])) if (~benign_mask).any() else 0.0
    out["direction_correct"] = bool(out["median_attack"] >= out["median_benign"])
    return out


def evaluate_both(test_energy, threshold, y_bin, y_family):
    """Threshold metrics (point-adjusted P/R/F1 + raw confusion) PLUS threshold-free
    ranking AUROC/AUPRC at TWO granularities, both reported for scientific rigor:
      - per-flow (top-level keys): TimesNet's native per-timestep unit
      - window-level (res["window_level"]): aggregated to the proposed per-window unit
    Threshold-based numbers are per-flow only (a threshold isn't comparable across
    methods anyway)."""
    res = {"model": "TimesNet (native pipeline)"}

    # ── Native: point-adjusted detection ──
    pred = (test_energy > threshold).astype(int)
    gt_adj, pred_adj = adjustment(y_bin, pred.copy())
    accuracy = accuracy_score(gt_adj, pred_adj)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_adj, pred_adj, average="binary", zero_division=0
    )
    res["point_adjusted"] = {
        "anomaly_ratio": ANOMALY_RATIO,
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    # ── Native (no point-adjust) raw confusion at threshold ──
    tp = int(((pred == 1) & (y_bin == 1)).sum())
    fp = int(((pred == 1) & (y_bin == 0)).sum())
    tn = int(((pred == 0) & (y_bin == 0)).sum())
    fn = int(((pred == 0) & (y_bin == 1)).sum())
    res["raw_threshold"] = {
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

    # ── Threshold-free ranking metrics, PER-FLOW (TimesNet native granularity) ──
    res.update(_ranking_metrics(test_energy, y_bin, y_family))

    # ── Same metrics at WINDOW granularity (proposed unit) ──
    # Aggregate per-timestep energy over each non-overlapping eval window; label the
    # window by its worst (max) flow — matching the proposed window aggregation.
    # Score = max energy in the window (a window is anomalous iff its peak is). Switch
    # `.max(axis=1)` to `.mean(axis=1)` for a mean-aggregated variant.
    nW = len(test_energy) // WIN_SIZE
    if nW > 0:
        e_w = np.asarray(test_energy)[:nW * WIN_SIZE].reshape(nW, WIN_SIZE)
        f_w = np.asarray(y_family)[:nW * WIN_SIZE].reshape(nW, WIN_SIZE)
        win_scores = e_w.max(axis=1)
        win_family = f_w.max(axis=1)
        win_bin = (win_family != 0).astype(int)
        if len(np.unique(win_bin)) >= 2:
            res["window_level"] = _ranking_metrics(win_scores, win_bin, win_family)
            res["window_level"]["n_windows"] = int(nW)
            res["window_level"]["aggregation"] = "score=max-over-timestep; family=max-over-window"

    # Per-family stats + detection rate at threshold
    per_fam = []
    for fam in sorted(np.unique(y_family)):
        m = y_family == fam
        s = test_energy[m]
        per_fam.append({
            "name": CIC_LABEL_MAP.get(fam, f"fam_{fam}"),
            "n": int(m.sum()),
            "detected": int(pred[m].sum()),
            "rate": float(pred[m].sum() / m.sum()) if m.sum() else 0.0,
            "holdout": bool(fam in HOLDOUT_FAMILIES),
            "mean": float(s.mean()), "median": float(np.median(s)),
            "std": float(s.std()),
            "p90": float(np.percentile(s, 90)), "p99": float(np.percentile(s, 99)),
        })
    res["per_family"] = per_fam
    return res


def print_result(res):
    print(f"\n{'=' * 80}")
    print(f"  {res['model']}")
    print(f"{'=' * 80}")

    pa = res["point_adjusted"]
    print(f"  Point-adjusted (TimesNet native, anomaly_ratio={pa['anomaly_ratio']}):")
    print(f"    threshold={pa['threshold']:.6f}")
    print(f"    accuracy={pa['accuracy']:.4f} precision={pa['precision']:.4f} "
          f"recall={pa['recall']:.4f} f1={pa['f1']:.4f}")

    rt = res["raw_threshold"]
    print("\n  Raw at-threshold (no point adjustment):")
    print(f"    precision={rt['precision']:.4f} recall={rt['recall']:.4f} "
          f"fpr={rt['fpr']:.4f}")
    print(f"    TP={rt['tp']}, FP={rt['fp']}, TN={rt['tn']}, FN={rt['fn']}")

    print("\n  Per-FLOW ranking (TimesNet native per-timestep unit):")
    print(f"    AUROC (overall)     : {res['auroc_overall']:.6f}")
    print(f"    AUPRC (overall)     : {res['auprc_overall']:.6f}")
    print(f"    AUPRC (holdout 8+9) : {res['auprc_holdout']:.6f}  "
          f"95% CI [{res['holdout_ci_low']:.6f}, {res['holdout_ci_high']:.6f}]")
    direction = "CORRECT" if res["direction_correct"] else "INVERTED"
    print(f"    Score direction     : {direction}  "
          f"(benign={res['median_benign']:.2e}, attack={res['median_attack']:.2e})")

    wl = res.get("window_level")
    if wl:
        print(f"\n  Per-WINDOW ranking (aggregated to proposed unit; n={wl['n_windows']}, "
              f"{wl['aggregation']}):")
        print(f"    AUROC (overall)     : {wl['auroc_overall']:.6f}")
        print(f"    AUPRC (overall)     : {wl['auprc_overall']:.6f}")
        print(f"    AUPRC (holdout 8+9) : {wl['auprc_holdout']:.6f}  "
              f"95% CI [{wl['holdout_ci_low']:.6f}, {wl['holdout_ci_high']:.6f}]")

    print("\n  TPR @ FPR operating points (per-flow):")
    for k, v in res.get("tpr_at_fpr", {}).items():
        print(f"    {k:<20s} : {v:.4f}")

    print(f"\n  {'family':<16s} {'n':>7s} {'det':>7s} {'rate':>6s} "
          f"{'mean':>10s} {'median':>10s} {'p99':>10s}")
    print("  " + "-" * 74)
    for row in res["per_family"]:
        tag = "*" if row["holdout"] else " "
        print(f"  {tag}{row['name']:<15s} {row['n']:>7d} {row['detected']:>7d} "
              f"{row['rate']:>6.3f} {row['mean']:>10.2e} {row['median']:>10.2e} "
              f"{row['p99']:>10.2e}")


# =============================================================================
# Model speed / cost benchmark (epoch-independent, comparable across models)
# =============================================================================

def benchmark_model(model, n_features, win_size, device,
                    batch_size=BATCH_SIZE, n_warmup=15, n_iter=100):
    """Epoch-independent model cost metrics for fair cross-model comparison.

    Reports params, FLOPs/MACs (if `thop` or `fvcore` is installed), inference
    latency/throughput, train-step latency, and peak inference memory. These stay
    comparable regardless of dataset size, windowing, or epoch budget — unlike raw
    training wall-clock, which conflates all three (see module discussion).

    Timing correctness: a GPU warmup primes cuDNN autotune/lazy alloc, and every
    measured region is bracketed by torch.cuda.synchronize() so we time actual
    kernel execution, not async launch overhead.

    Non-destructive: runs on a deep copy and saves/restores RNG state, so neither
    the real model's weights nor the global RNG (hence the faithful training run)
    are perturbed.
    """
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None

    work = copy.deepcopy(model).to(device)
    x = torch.randn(batch_size, win_size, n_features, device=device)
    cuda = device.type == "cuda"

    metrics = {
        "n_params": int(sum(p.numel() for p in work.parameters() if p.requires_grad)),
        "batch_size": batch_size,
        "win_size": win_size,
        "n_features": n_features,
        "device": str(device),
    }

    # ── FLOPs / MACs (optional dependency, graceful fallback) ──
    macs = None
    try:
        from thop import profile
        macs, _ = profile(copy.deepcopy(work), inputs=(x, None, None, None), verbose=False)
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            macs = float(FlopCountAnalysis(work, (x, None, None, None)).total())
        except Exception:
            macs = None
    if macs is not None:
        metrics["macs_per_window"] = float(macs) / batch_size
        metrics["gflops_per_window"] = float(macs) * 2 / batch_size / 1e9   # FLOPs = 2*MACs

    # ── Inference latency / throughput + peak memory (warmup + synchronize) ──
    work.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            work(x, None, None, None)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            work(x, None, None, None)
        if cuda:
            torch.cuda.synchronize()
        infer_per_batch = (time.perf_counter() - t0) / n_iter
    metrics["infer_ms_per_window"] = infer_per_batch / batch_size * 1e3
    metrics["infer_windows_per_s"] = batch_size / infer_per_batch
    metrics["infer_flows_per_s"] = batch_size * win_size / infer_per_batch  # non-overlap eval
    if cuda:
        metrics["peak_infer_mem_mb"] = torch.cuda.max_memory_allocated(device) / 1e6

    # ── Train-step latency (fwd+bwd+step), for the convergence-cost framing ──
    work.train()
    crit = nn.MSELoss()
    opt = torch.optim.Adam(work.parameters(), lr=LR)

    def _train_step():
        opt.zero_grad()
        loss = crit(work(x, None, None, None), x)
        loss.backward()
        opt.step()

    for _ in range(n_warmup):
        _train_step()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _train_step()
    if cuda:
        torch.cuda.synchronize()
    metrics["train_ms_per_batch"] = (time.perf_counter() - t0) / n_iter * 1e3

    # Restore RNG so the subsequent real training run is unaffected.
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)
    return metrics


def print_benchmark(b):
    print("  Model cost (epoch-independent; comparable across models):")
    print(f"    Params               : {b['n_params']:,}")
    if "gflops_per_window" in b:
        print(f"    Compute              : {b['gflops_per_window']:.4f} GFLOPs/window "
              f"({b['macs_per_window']:.3e} MACs/window)")
    else:
        print("    Compute              : FLOPs unavailable (pip install thop OR fvcore)")
    print(f"    Inference latency    : {b['infer_ms_per_window']:.4f} ms/window")
    print(f"    Inference throughput : {b['infer_windows_per_s']:,.0f} windows/s "
          f"({b['infer_flows_per_s']:,.0f} flows/s, non-overlap eval)")
    print(f"    Train-step latency   : {b['train_ms_per_batch']:.2f} ms/batch (fwd+bwd, bs={b['batch_size']})")
    if "peak_infer_mem_mb" in b:
        print(f"    Peak inference mem   : {b['peak_infer_mem_mb']:.1f} MB")


# =============================================================================
# Experiment
# =============================================================================

def run_experiment(run_seed):
    global SEED, OUTPUT_DIR
    SEED = run_seed
    OUTPUT_DIR = os.path.join(RESULTS_DIR, f"seed_{SEED}")
    set_seed(SEED)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Fixed input shapes (drop_last train batches are all 128xWIN_SIZExF) => let
    # cuDNN autotune the conv algorithm once. Free speedup, no effect on results.
    # Skipped under STRICT_DETERMINISM, which forces deterministic conv algos.
    if device.type == "cuda" and not STRICT_DETERMINISM:
        torch.backends.cudnn.benchmark = True

    print("\n" + "=" * 80)
    print("  Proposed split + TimesNet-native preprocessing")
    print("=" * 80)
    (train_stream, val_stream, test_stream,
     y_test_bin, y_test_family, feature_names,
     test_indices) = load_and_preprocess_native()
    n_features = train_stream.shape[1]

    print("\n" + "=" * 80)
    print(f"  Native windowing (WIN_SIZE={WIN_SIZE}, TRAIN_STEP={TRAIN_STEP})")
    print("=" * 80)
    X_train_w = WindowDataset(train_stream, WIN_SIZE, TRAIN_STEP)
    X_val_w = WindowDataset(val_stream, WIN_SIZE, TRAIN_STEP)
    X_test_w = WindowDataset(test_stream, WIN_SIZE, WIN_SIZE)   # non-overlapping eval
    n_test_scored = len(X_test_w) * WIN_SIZE
    print(f"    Train windows: {X_train_w.shape}")
    print(f"    Val   windows: {X_val_w.shape}")
    print(f"    Test  windows: {X_test_w.shape} (non-overlapping)")
    print(f"    Test timesteps scored: {n_test_scored} / {len(test_stream)} "
          f"(trailing {len(test_stream) - n_test_scored} truncated)")

    print("\n" + "=" * 80)
    print("  TimesNet model construction")
    print("=" * 80)
    configs = make_timesnet_config(n_features)
    configs.seq_len = WIN_SIZE          # use the native window size
    model = Model(configs).float().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Window: {WIN_SIZE} timesteps x {n_features} features (no entity grouping)")

    benchmark = benchmark_model(model, n_features, WIN_SIZE, device)
    print_benchmark(benchmark)

    train_loader = make_loader(X_train_w, BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = make_loader(X_val_w, BATCH_SIZE, shuffle=False)
    criterion = nn.MSELoss()
    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint.pth")

    print("\n" + "=" * 80)
    print("  Training (Adam + OneCycleLR, early stop on val recon loss)")
    print("=" * 80)
    history = []
    if SKIP_TRAINING:
        print("  Loading checkpoint because SKIP_TRAINING=True")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = None            # TSLib lradj type1: lr halved per epoch below
        best_val = float("inf")
        patience_ctr = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            adjust_learning_rate(optimizer, epoch)
            train_loss = run_epoch(model, train_loader, optimizer, criterion, scheduler, device)
            val_loss = validate(model, val_loader, criterion, device)
            elapsed = time.time() - t0
            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"], "time_s": elapsed,
            })
            if epoch <= 5 or epoch % 10 == 0 or epoch == EPOCHS:
                print(f"  Epoch {epoch:3d}/{EPOCHS} train={train_loss:.7f} "
                      f"val={val_loss:.7f} ({elapsed:.1f}s)")
            if val_loss < best_val:
                best_val = val_loss
                patience_ctr = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, ckpt_path)
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"\n  Early stopping at epoch {epoch} (best val_loss={best_val:.7f})")
                    break
        model.load_state_dict(best_state)
        print(f"\n  Best val_loss = {best_val:.7f}")

    print("\n" + "=" * 80)
    print("  Per-timestep energy + percentile threshold")
    print("=" * 80)
    train_energy = compute_timestep_energy(model, X_train_w, device)
    test_energy = compute_timestep_energy(model, X_test_w, device)
    combined = np.concatenate([train_energy, test_energy], axis=0)
    threshold = float(np.percentile(combined, 100 - ANOMALY_RATIO))
    print(f"  Train energy mean={train_energy.mean():.6f} std={train_energy.std():.6f}")
    print(f"  Test  energy mean={test_energy.mean():.6f} std={test_energy.std():.6f}")
    print(f"  Threshold (anomaly_ratio={ANOMALY_RATIO}) = {threshold:.6f}")

    # Align per-flow labels with the (truncated) per-timestep test energies.
    gt_bin = y_test_bin[:len(test_energy)]
    gt_family = y_test_family[:len(test_energy)]

    # Flow-level export: per-flow scores keyed by sorted-CSV row position plus
    # benign-validation flow energies (the threshold source specified by the
    # source). Shared test-universe filtering is applied by the aggregation
    # workflow, so the operating point printed here precedes that filtering.
    val_eval_w = WindowDataset(val_stream, WIN_SIZE, WIN_SIZE)
    val_energy = compute_timestep_energy(model, val_eval_w, device)
    thr1 = (float(np.quantile(val_energy, 0.99))
            if len(val_energy) else float("inf"))
    flag = test_energy >= thr1
    t2_tpr = float(flag[gt_bin == 1].mean()) if (gt_bin == 1).any() else float("nan")
    t2_fpr = float(flag[gt_bin == 0].mean()) if (gt_bin == 0).any() else float("nan")
    t2_hold = np.isin(gt_family, HOLDOUT_FAMILIES)
    t2_ho = float(flag[t2_hold].mean()) if t2_hold.any() else float("nan")
    print("\n  Flow-level operating point before shared-universe filtering "
          "(benign-val threshold, 1% budget):")
    print(f"    TPR {t2_tpr:.4f}  actual FPR {t2_fpr:.4f}  hoTPR {t2_ho:.4f}  "
          f"(val flows scored: {len(val_energy)}/{len(val_stream)})")
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "track2_flows.npz"),
        flow_pos=np.asarray(test_indices[:len(test_energy)], dtype=np.int64),
        score=test_energy.astype(np.float32),
        valb_score=val_energy.astype(np.float32))

    print("\n" + "=" * 80)
    print("  Evaluation — native point-adjusted + raw per-flow")
    print("=" * 80)
    result = evaluate_both(test_energy, threshold, gt_bin, gt_family)
    print_result(result)

    result["track2_operating_point"] = {
        "tpr": t2_tpr, "fpr_actual": t2_fpr, "tpr_holdout": t2_ho,
    }
    result["config"] = {
        "seed": SEED,
        "split": "proposed time-split (no val_mixed)",
        "processing": "TimesNet-native (no entity grouping, nan_to_num+StandardScaler)",
        "win_size": WIN_SIZE,
        "train_step": TRAIN_STEP,
        "n_features": n_features,
        "features": feature_names,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "patience": PATIENCE,
        "anomaly_ratio": ANOMALY_RATIO,
        "test_timesteps_scored": int(len(test_energy)),
        "test_timesteps_total": int(len(test_stream)),
    }
    result["benchmark"] = benchmark

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(deep_convert(result), f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(deep_convert(history), f, indent=2)
    np.save(os.path.join(OUTPUT_DIR, "test_energy.npy"), test_energy)

    print(f"\n  Results saved to {OUTPUT_DIR}/")
    return result


def main():
    all_results = []
    for s in RUN_SEEDS:
        print(f"\n\n{'=' * 80}\n  TIMESNET (NATIVE PIPELINE) — SEED {s}\n{'=' * 80}")
        all_results.append(run_experiment(s))

    print(f"\n\n{'=' * 80}\n  AGGREGATED OVER {len(RUN_SEEDS)} SEEDS\n{'=' * 80}")
    print("  Per-FLOW (native):")
    for metric in ["auroc_overall", "auprc_overall", "auprc_holdout"]:
        vals = [r[metric] for r in all_results]
        print(f"    {metric:<18s}: {np.mean(vals):.6f} +/- {np.std(vals):.6f}")
    if all("window_level" in r for r in all_results):
        print("  Per-WINDOW (proposed unit):")
        for metric in ["auroc_overall", "auprc_overall", "auprc_holdout"]:
            vals = [r["window_level"][metric] for r in all_results]
            print(f"    {metric:<18s}: {np.mean(vals):.6f} +/- {np.std(vals):.6f}")
    for metric in ["precision", "recall", "f1"]:
        vals = [r["point_adjusted"][metric] for r in all_results]
        print(f"  PA_{metric:<17s}: {np.mean(vals):.6f} +/- {np.std(vals):.6f}")

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "aggregated_results.json")
    with open(out_path, "w") as f:
        json.dump(deep_convert(all_results), f, indent=2)
    print(f"\nSaved aggregated results to {out_path}")


# =============================================================================
# Local smoke test (no dataset / no Drive needed):  python timesnet_cic_unsw_nb15_native.py --smoke
# =============================================================================

def _smoke_test():
    """Exercise windowing, energy, point-adjustment and metric math on synthetic data."""
    print("Running smoke test (synthetic, no dataset) ...")

    # seg_windows count + shape
    nvars = 6
    stream = np.random.randn(523, nvars).astype(np.float32)
    w = seg_windows(stream, WIN_SIZE, WIN_SIZE)
    assert w.shape == ((523 - WIN_SIZE) // WIN_SIZE + 1, WIN_SIZE, nvars), w.shape

    # compute_timestep_energy length == n_windows * WIN_SIZE
    device = torch.device("cpu")
    configs = make_timesnet_config(nvars)
    configs.seq_len = WIN_SIZE
    model = Model(configs).float().to(device)
    energy = compute_timestep_energy(model, w, device, batch_size=4)
    assert energy.shape == (w.shape[0] * WIN_SIZE,), energy.shape

    # adjustment: hand-checked reference example.
    gt = np.array([0, 1, 1, 1, 0, 1, 0])
    pred = np.array([0, 0, 1, 0, 0, 0, 0])
    # detection inside segment [1..3] -> whole segment flips to 1; segment [5] untouched.
    _, adj = adjustment(gt, pred)
    assert adj.tolist() == [0, 1, 1, 1, 0, 0, 0], adj.tolist()

    # adjustment never marks a true-negative region.
    gt2 = np.array([0, 0, 0, 1, 1])
    pred2 = np.array([1, 0, 0, 1, 0])
    _, adj2 = adjustment(gt2, pred2)
    assert adj2.tolist() == [1, 0, 0, 1, 1], adj2.tolist()

    # threshold + evaluate_both run end-to-end on synthetic per-flow energies.
    rng = np.random.RandomState(RUN_SEEDS[0])
    test_energy = np.concatenate([rng.randn(200) * 0.1, rng.randn(40) * 0.1 + 2.0])
    y_bin = np.concatenate([np.zeros(200, int), np.ones(40, int)])
    y_family = np.concatenate([np.zeros(200, int), rng.randint(1, 10, 40)])
    thr = float(np.percentile(test_energy, 100 - ANOMALY_RATIO))
    res = evaluate_both(test_energy, thr, y_bin, y_family)
    assert 0.0 <= res["auroc_overall"] <= 1.0
    assert res["direction_correct"]  # injected anomalies score higher
    print("  OK: shapes, energy length, point-adjustment, and metric math all pass.")
    return 0


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        sys.exit(_smoke_test())
    main()
