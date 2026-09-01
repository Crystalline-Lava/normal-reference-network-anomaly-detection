# Adapted in part from THUML Time-Series-Library (TimesNet), MIT License.
# Source and full upstream notice: ../../THIRD_PARTY_NOTICES.md

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import mstats
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# Fixed five-seed set reported in the manuscript.
RUN_SEEDS = (42, 456, 7, 789, 1024)
SEED = RUN_SEEDS[0]  # Active model seed; reassigned by run_experiment().

# Official PSM-style TimesNet configuration: d_model 64, d_ff 64, two encoder
# layers, top_k 3, batch 128, and Adam at 1e-4 with per-epoch halving. Training
# runs for six epochs and also reports native selection within the official
# three-epoch budget.
EPOCHS = 6
OFFICIAL_BUDGET_EPOCHS = 3
BATCH_SIZE = 128
LR = 1e-4
PATIENCE = 3

# Shared entity-window evaluation settings.
MIN_SELECT_EPOCH = 1
TPR_BUDGET = 0.01         # TPR @ 1% budget, threshold from benign val only
READOUTS = ["raw", "std", "maha"]  # raw = E.mean(1) IS the native TSLib score

D_MODEL = 64
D_FF = 64
E_LAYERS = 2
TOP_K = 3
NUM_KERNELS = 6
DROPOUT = 0.1

LABEL_TO_ID = {
    "Benign": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3,
    "Exploits": 4, "Fuzzers": 5, "Generic": 6,
    "Reconnaissance": 7, "Shellcode": 8, "Worms": 9,
}
CIC_LABEL_MAP = {v: k for k, v in LABEL_TO_ID.items()}
HOLDOUT_FAMILIES = [8, 9]

BASE_DIR = os.getenv("CIC_BASE_DIR", "/content/drive/MyDrive/cic_only")
RAW_CSV = os.getenv("CIC_UNSW_RAW_CSV", "/content/drive/MyDrive/CIC-UNSW-NB15/CICFlowMeter_out.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "timesnet_baseline")

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

try:
    from google.colab import drive
    drive.mount("/content/drive")
except ModuleNotFoundError:
    pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deep_convert(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_convert(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def build_windows(
    x: np.ndarray,
    y: np.ndarray,
    entity_ids: np.ndarray,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows, family_labels = [], []
    for eid in np.unique(entity_ids):
        mask = entity_ids == eid
        x_e = x[mask]
        y_e = y[mask]
        n = len(x_e)
        for start in range(0, n - window_size + 1, stride):
            windows.append(x_e[start:start + window_size])
            family_labels.append(int(y_e[start:start + window_size].max()))

    if not windows:
        return (
            np.zeros((0, window_size, x.shape[1]), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )

    windows = np.stack(windows).astype(np.float32)
    family_labels = np.array(family_labels, dtype=np.int64)
    attack_labels = (family_labels != 0).astype(np.int64)
    return windows, attack_labels, family_labels


def load_and_preprocess():
    set_seed(SEED)
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

    x_all = data.values.astype(np.float64)
    x_all[~np.isfinite(x_all)] = np.nan
    x_all = SimpleImputer(strategy="median").fit_transform(x_all)
    for j in range(x_all.shape[1]):
        x_all[:, j] = mstats.winsorize(x_all[:, j], limits=[0.001, 0.001])

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

    scaler = StandardScaler()
    scaler.fit(x_all[train_indices])
    x_scaled = scaler.transform(x_all).astype(np.float32)

    print(f"\n  Building windows (W={WINDOW_SIZE}, delta features OFF) ...")
    x_train_w, _, _ = build_windows(
        x_scaled[train_indices], y_all[train_indices],
        entity_ids[train_indices], WINDOW_SIZE, TRAIN_STRIDE)
    x_val_w, _, _ = build_windows(
        x_scaled[val_indices], y_all[val_indices],
        entity_ids[val_indices], WINDOW_SIZE, TRAIN_STRIDE)
    x_test_w, y_test_atk_w, y_test_fam_w = build_windows(
        x_scaled[test_indices], y_all[test_indices],
        entity_ids[test_indices], WINDOW_SIZE, TEST_STRIDE)

    if len(x_train_w) > MAX_TRAIN_WINDOWS:
        idx = np.random.choice(len(x_train_w), MAX_TRAIN_WINDOWS, replace=False)
        x_train_w = x_train_w[idx]
        print(f"  Subsampled train windows to {MAX_TRAIN_WINDOWS}")

    # Val mixed uses known-attack windows sampled from the test-period pool.
    # Remove those sampled windows from test to avoid validation/test overlap.
    holdout_w = np.isin(y_test_fam_w, HOLDOUT_FAMILIES)
    known_atk = (y_test_atk_w == 1) & ~holdout_w
    known_atk_idx = np.where(known_atk)[0]
    val_atk_idx = np.zeros(0, dtype=int)
    if len(known_atk_idx) > 0:
        try:
            val_atk_idx, _ = train_test_split(
                known_atk_idx,
                test_size=0.90,
                random_state=SEED,
                stratify=y_test_fam_w[known_atk_idx],
            )
        except ValueError:
            val_atk_idx = np.random.choice(
                known_atk_idx, max(1, len(known_atk_idx) // 10),
                replace=False)
        x_val_mixed_w = np.concatenate([x_val_w, x_test_w[val_atk_idx]])
        y_val_mixed_w = np.concatenate([
            np.zeros(len(x_val_w), dtype=int),
            np.ones(len(val_atk_idx), dtype=int),
        ])
        test_keep = np.ones(len(x_test_w), dtype=bool)
        test_keep[val_atk_idx] = False
        x_test_w = x_test_w[test_keep]
        y_test_atk_w = y_test_atk_w[test_keep]
        y_test_fam_w = y_test_fam_w[test_keep]
    else:
        x_val_mixed_w = x_val_w.copy()
        y_val_mixed_w = np.zeros(len(x_val_w), dtype=int)

    print("\n  Windows built:")
    print(f"    Train:     {x_train_w.shape}")
    print(f"    Val:       {x_val_w.shape}")
    print(f"    Val mixed: {x_val_mixed_w.shape} (attack rate={y_val_mixed_w.mean():.4f})")
    print(f"      Known-attack val windows removed from test: {len(val_atk_idx)}")
    print(f"    Test:      {x_test_w.shape} (attack rate={y_test_atk_w.mean():.4f})")

    return (
        x_train_w, x_val_w, x_val_mixed_w, y_val_mixed_w,
        x_test_w, y_test_atk_w, y_test_fam_w,
        data.columns.tolist(), scaler,
    )


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

    fpr_arr, tpr_arr, _ = roc_curve(y_attack, scores_test)
    tpr_at_fpr = {}
    for target_fpr in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        idx = np.searchsorted(fpr_arr, target_fpr, side="right") - 1
        idx = max(0, min(idx, len(tpr_arr) - 1))
        tpr_at_fpr[f"TPR@FPR={target_fpr:.1%}"] = float(tpr_arr[idx])
    res["tpr_at_fpr"] = tpr_at_fpr

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

    per_fam = []
    for fam in sorted(np.unique(y_family)):
        mask = y_family == fam
        s = scores_test[mask]
        per_fam.append({
            "name": CIC_LABEL_MAP.get(fam, f"fam_{fam}"),
            "n": int(mask.sum()),
            "mean": float(s.mean()),
            "median": float(np.median(s)),
            "std": float(s.std()),
            "p90": float(np.percentile(s, 90)),
            "p99": float(np.percentile(s, 99)),
        })
    res["per_family"] = per_fam
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

    print("\n  TPR at operating FPR:")
    for k, v in res["tpr_at_fpr"].items():
        print(f"    {k:<14s}: {v:.4f}")

    print("\n  Per-family score summary:")
    print(f"  {'Family':<20s} {'N':>8s} {'Median':>12s} {'P90':>12s} {'P99':>12s}")
    print("  " + "-" * 68)
    for row in res["per_family"]:
        print(f"  {row['name']:<20s} {row['n']:>8d} "
              f"{row['median']:>12.4f} {row['p90']:>12.4f} {row['p99']:>12.4f}")


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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses: List[float] = []
    for (batch_x,) in loader:
        batch_x = batch_x.float().to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.average(losses))


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    for (batch_x,) in loader:
        batch_x = batch_x.float().to(device)
        outputs = model(batch_x)
        losses.append(criterion(outputs, batch_x).item())
    return float(np.average(losses))


@torch.no_grad()
def feature_residuals(model: nn.Module, x: np.ndarray, device: torch.device,
                      batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Per-window per-feature residual E (N, F): mean over the W positions of
    (x − recon)². Note E.mean(axis=1) IS the native TimesNet/TSLib score
    (reconstruction_energy = mean over dims (1, 2)) — raw and the calibrated
    readouts aggregate the SAME residual tensor, so any difference is
    attributable to the readout alone."""
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    out: List[np.ndarray] = []
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        recon = model(batch_x)
        out.append(((recon - batch_x) ** 2).mean(dim=1).cpu().numpy())
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


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int) -> float:
    lr = LR * (0.5 ** ((epoch - 1) // 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def run_experiment(run_seed: int) -> Dict[str, Any]:
    global SEED, OUTPUT_DIR
    SEED = run_seed
    OUTPUT_DIR = os.path.join(BASE_DIR, f"timesnet_baseline_seed_{run_seed}")

    set_seed(run_seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output dir: {OUTPUT_DIR}")

    print("\n" + "=" * 80)
    print("  Shared Data Loading & Windowing (delta features OFF)")
    print("=" * 80)
    (x_train, x_val, x_val_mixed, y_val_mixed,
     x_test, y_test_attack, y_test_family,
     feature_names, _scaler) = load_and_preprocess()

    n_features = x_train.shape[2]
    config = TimesNetConfig(
        seq_len=WINDOW_SIZE,
        pred_len=0,
        enc_in=n_features,
        c_out=n_features,
    )
    model = TimesNet(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "=" * 80)
    print("  TimesNet Baseline Construction")
    print("=" * 80)
    print(f"  Parameters: {n_params:,}")
    print(f"  Window: {WINDOW_SIZE} flows x {n_features} features")
    print(f"  Feature count from shared loader: {len(feature_names)}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint.pt")
    criterion = nn.MSELoss()
    history: List[Dict[str, float]] = []

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float("inf")
    patience_counter = 0

    # Two selection rules tracked side by side (scores captured at the moment
    # of selection; the per-epoch TEST-AP curve is stored for the ORACLE/
    # selection-gap report ONLY):
    #   nat_* — NATIVE selection: the epoch with the lowest benign-val recon
    #           loss (TimesNet/TSLib's own checkpoint rule; never sees labels).
    #           PRIMARY baseline row.
    #   (plain) — val-AP selection: the selection privilege our own method
    #           uses; labeled secondary row ("does epoch-picking explain the
    #           gap?").
    track = {r: {"val_ap": -np.inf, "epoch": None,
                 "scores_test": None, "scores_valb": None,
                 "nat_val_ap": None, "nat_epoch": None,
                 "nat_scores_test": None, "nat_scores_valb": None,
                 "off_val_ap": None, "off_epoch": None,
                 "off_scores_test": None, "off_scores_valb": None,
                 "val_curve": [], "test_curve": []} for r in READOUTS}

    print("\n" + "=" * 80)
    print("  Training (MSE reconstruction) + per-epoch readout tracking")
    print("=" * 80)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        lr = adjust_learning_rate(optimizer, epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        E_b = feature_residuals(model, x_val, device)
        stats = fit_benign_stats(E_b)
        E_vm = feature_residuals(model, x_val_mixed, device)
        sel_ok = epoch >= MIN_SELECT_EPOCH
        vaps = {}
        for r in READOUTS:
            s_vm = apply_readout(E_vm, stats, r)
            vaps[r] = float(average_precision_score(y_val_mixed, s_vm))
            track[r]["val_curve"].append(vaps[r])
        # LAZY test scoring: the (large) test pass runs only on epochs where a
        # capture can happen — val loss improved (native/off selection) or some
        # readout's val AP improved (valAP selection). Skipped epochs record
        # NaN in the test curve (gap/oracle uses the computed points only).
        loss_improved = val_loss < best_val_loss
        need_test = loss_improved or any(
            sel_ok and vaps[r] > track[r]["val_ap"] for r in READOUTS)
        cur = {}
        if need_test:
            E_te = feature_residuals(model, x_test, device)
        for r in READOUTS:
            if need_test:
                s_te = apply_readout(E_te, stats, r).astype(np.float32)
                s_vb = apply_readout(E_b, stats, r).astype(np.float32)
                t_ap = float(average_precision_score(y_test_attack, s_te))
                cur[r] = (vaps[r], s_te, s_vb)
                track[r]["test_curve"].append(t_ap)
                if sel_ok and vaps[r] > track[r]["val_ap"]:
                    track[r].update(val_ap=vaps[r], epoch=epoch,
                                    scores_test=s_te, scores_valb=s_vb)
            else:
                track[r]["test_curve"].append(float("nan"))

        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "time_s": elapsed,
            **{f"val_ap_{r}": track[r]["val_curve"][-1] for r in READOUTS},
        })
        print(f"  Epoch {epoch:3d}/{EPOCHS}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"val_AP raw {track['raw']['val_curve'][-1]:.4f} "
              f"std {track['std']['val_curve'][-1]:.4f} "
              f"maha {track['maha']['val_curve'][-1]:.4f}  "
              f"lr={lr:.2e}  ({elapsed:.1f}s)")

        if loss_improved:
            best_val_loss = val_loss
            patience_counter = 0
            # NATIVE selection: this is exactly the checkpoint TSLib would keep
            for r in READOUTS:
                track[r].update(nat_val_ap=cur[r][0], nat_epoch=epoch,
                                nat_scores_test=cur[r][1],
                                nat_scores_valb=cur[r][2])
                # ... and within the official 3-epoch budget, the faithful row
                if epoch <= OFFICIAL_BUDGET_EPOCHS:
                    track[r].update(off_val_ap=cur[r][0], off_epoch=epoch,
                                    off_scores_test=cur[r][1],
                                    off_scores_valb=cur[r][2])
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "history": history,
                "config": config.__dict__,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val loss={best_val_loss:.6f})")
                break

    # Readout comparison on the shared entity-window task
    print("\n" + "=" * 80)
    print("  TIMESNET READOUT COMPARISON — same residuals, different readout")
    print("  (raw = E.mean(1) IS the native TSLib score, so raw doubles as the")
    print("  paper row. PRIMARY rows use NATIVE selection = lowest benign-val")
    print("  recon loss, TSLib's own checkpoint rule; val-AP-sel rows are the")
    print("  selection-equalized secondary. gap = test-oracle − selected)")
    print("=" * 80)
    print(f"  {'readout':<18}{'sel_ep':>7}{'val_AP':>9}{'test_AP':>9}"
          f"{'AUROC':>8}{'ho_AP':>8}{'gap':>8}{'TPR@1%':>8}{'FPR_act':>9}{'hoTPR':>7}")
    print("  " + "-" * 90)
    rows = {}

    def report_row(key, label, sel_epoch, val_ap, s_test, s_valb, gap_curve):
        res = evaluate_model(f"TimesNet[{label}]", s_test,
                             y_test_attack, y_test_family)
        op = operating_point(s_valb, s_test, y_test_attack, y_test_family)
        finite = [v for v in gap_curve if np.isfinite(v)]
        gap = (max(finite) - res["auprc_overall"]) if finite else float("nan")
        rows[key] = {"readout": label, "sel_epoch": sel_epoch, "val_ap": val_ap,
                     "gap": gap, "operating_point": op, **res}
        print(f"  {label:<18}{sel_epoch:>7}{val_ap:>9.4f}"
              f"{res['auprc_overall']:>9.4f}{res['auroc_overall']:>8.4f}"
              f"{res['auprc_holdout']:>8.4f}{gap:>+8.4f}{op['tpr']:>8.4f}"
              f"{op['fpr_actual']:>9.4f}{op['tpr_holdout']:>7.4f}")

    for r in READOUTS:                       # PRIMARY: native selection
        tr = track[r]
        report_row(f"{r}_natsel", f"{r}[native-sel]", tr["nat_epoch"],
                   tr["nat_val_ap"], tr["nat_scores_test"],
                   tr["nat_scores_valb"], tr["test_curve"])
    for r in READOUTS:                       # fully faithful: official budget
        tr = track[r]
        if tr["off_scores_test"] is not None:
            report_row(f"{r}_official",
                       f"{r}@off{OFFICIAL_BUDGET_EPOCHS}ep",
                       tr["off_epoch"], tr["off_val_ap"],
                       tr["off_scores_test"], tr["off_scores_valb"], [])
    for r in READOUTS:                       # secondary: val-AP selection
        tr = track[r]
        report_row(r, f"{r}[valAP-sel]", tr["epoch"], tr["val_ap"],
                   tr["scores_test"], tr["scores_valb"],
                   tr["test_curve"][MIN_SELECT_EPOCH - 1:])

    for sel, suff in (("native-sel", "_natsel"), ("valAP-sel", "")):
        d_std = (rows[f"std{suff}"]["auprc_overall"]
                 - rows[f"raw{suff}"]["auprc_overall"])
        d_maha = (rows[f"maha{suff}"]["auprc_overall"]
                  - rows[f"raw{suff}"]["auprc_overall"])
        print(f"\n  CALIBRATION DELTA [{sel}] (this seed): std−raw {d_std:+.4f}  "
              f"maha−raw {d_maha:+.4f}   (holdout: "
              f"{rows[f'std{suff}']['auprc_holdout'] - rows[f'raw{suff}']['auprc_holdout']:+.4f} / "
              f"{rows[f'maha{suff}']['auprc_holdout'] - rows[f'raw{suff}']['auprc_holdout']:+.4f})")
    print_result(rows["raw_natsel"])
    d_std_n = rows["std_natsel"]["auprc_overall"] - rows["raw_natsel"]["auprc_overall"]
    d_maha_n = rows["maha_natsel"]["auprc_overall"] - rows["raw_natsel"]["auprc_overall"]
    print_result(rows["maha_natsel"] if d_maha_n >= d_std_n else rows["std_natsel"])

    curves = {r: {"val": track[r]["val_curve"], "test": track[r]["test_curve"]}
              for r in READOUTS}
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(deep_convert(rows), f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(deep_convert({"history": history, "curves": curves}), f,
                  indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    return rows


def main() -> None:
    all_results = []
    for seed in RUN_SEEDS:
        print(f"\n\n{'=' * 80}\n  STARTING TIMESNET RUN WITH SEED {seed}\n{'=' * 80}")
        all_results.append(run_experiment(seed))

    print(f"\n\n{'=' * 80}\n  AGGREGATED TIMESNET RESULTS OVER {len(RUN_SEEDS)} SEEDS\n{'=' * 80}")
    agg_keys = ([f"{r}_natsel" for r in READOUTS]       # PRIMARY (native sel)
                + [f"{r}_official" for r in READOUTS]   # faithful 3-ep budget
                + READOUTS)                             # secondary (val-AP sel)
    for metric in ["auprc_overall", "auprc_holdout", "auroc_overall"]:
        print(f"\n  {metric}:")
        for k in agg_keys:
            vals = np.array([r[k][metric] for r in all_results if k in r],
                            dtype=float)
            if not len(vals):
                continue
            print(f"    {k:<16s}: {vals.mean():.4f} +/- {vals.std():.4f}   "
                  f"per-seed {np.round(vals, 4).tolist()}")
    print("\n  PAIRED CALIBRATION DELTA (the systematic-under-reading test):")
    for suff, sel in (("_natsel", "native-sel"), ("", "valAP-sel")):
        for cal in ("std", "maha"):
            for metric, lbl in (("auprc_overall", "test"), ("auprc_holdout", "ho")):
                d = np.array([r[f"{cal}{suff}"][metric] - r[f"raw{suff}"][metric]
                              for r in all_results], dtype=float)
                print(f"    [{sel}] {cal}−raw [{lbl:<4}]: mean {d.mean():+.4f}  "
                      f"positive on {(d > 0).sum()}/{len(d)} seeds  "
                      f"per-seed {np.round(d, 4).tolist()}")
    print("\n  TPR@1% (val-calibrated budget):")
    for k in agg_keys:
        rs = [r for r in all_results if k in r]
        if not rs:
            continue
        t = np.array([r[k]["operating_point"]["tpr"] for r in rs])
        f_ = np.array([r[k]["operating_point"]["fpr_actual"] for r in rs])
        print(f"    {k:<16s}: TPR {t.mean():.4f}  actual FPR {f_.mean():.4f}")

    out_path = os.path.join(BASE_DIR, "timesnet_baseline_aggregated_results.json")
    with open(out_path, "w") as f:
        json.dump(deep_convert(all_results), f, indent=2)
    print(f"\nSaved aggregated results to {out_path}")


if __name__ == "__main__":
    main()
