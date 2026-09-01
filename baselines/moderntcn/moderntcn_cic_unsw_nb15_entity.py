# Adapted in part from luodhhh/ModernTCN, MIT License.
# Source and full upstream notice: ../../THIRD_PARTY_NOTICES.md

"""
ModernTCN CIC-UNSW-NB15 baseline.

This file uses the entity-grouped CIC-UNSW-NB15 data loading and final metric reporting,
but uses the official ModernTCN anomaly-detection recipe:
  - ModernTCN reconstruction model
  - MSE reconstruction loss
  - Adam optimizer + OneCycleLR
  - early stopping on validation reconstruction loss
  - anomaly score from mean squared reconstruction error

Dataset paths are configurable through environment variables. Outputs are
written to a separate ModernTCN directory.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from scipy.stats import mstats
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from google.colab import drive

    drive.mount("/content/drive")
except Exception:
    pass


# =============================================================================
# Inlined ModernTCN Implementation
# =============================================================================

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1:, :].detach()
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - (self.last if self.subtract_last else self.mean)
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + (self.last if self.subtract_last else self.mean)
        return x


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            return torch.stack(x_out, dim=1)

        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module(
        "conv",
        get_conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        ),
    )
    result.add_module("bn", nn.BatchNorm1d(out_channels))
    return result


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups,
        small_kernel,
        small_kernel_merged=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=False,
            )
            if small_kernel is not None:
                assert small_kernel <= kernel_size
                self.small_conv = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=small_kernel,
                    stride=stride,
                    padding=small_kernel // 2,
                    groups=groups,
                    dilation=1,
                    bias=False,
                )

    def forward(self, inputs):
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, "small_conv"):
                out = out + self.small_conv(inputs)
        return out

    def padding_two_edge_1d(self, x, pad_length_left, pad_length_right, pad_values=0):
        d_out, d_in, _ = x.shape
        device = x.device
        dtype = x.dtype
        pad_left = torch.full((d_out, d_in, pad_length_left), pad_values, device=device, dtype=dtype)
        pad_right = torch.full((d_out, d_in, pad_length_right), pad_values, device=device, dtype=dtype)
        return torch.cat([pad_left, x, pad_right], dim=-1)

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.padding_two_edge_1d(
                small_k,
                (self.kernel_size - self.small_kernel) // 2,
                (self.kernel_size - self.small_kernel) // 2,
                0,
            )
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(
            in_channels=self.lkb_origin.conv.in_channels,
            out_channels=self.lkb_origin.conv.out_channels,
            kernel_size=self.lkb_origin.conv.kernel_size,
            stride=self.lkb_origin.conv.stride,
            padding=self.lkb_origin.conv.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.lkb_origin.conv.groups,
            bias=True,
        )
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        del self.lkb_origin
        if hasattr(self, "small_conv"):
            del self.small_conv


class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):
        super().__init__()
        self.dw = ReparamLargeKernelConv(
            in_channels=nvars * dmodel,
            out_channels=nvars * dmodel,
            kernel_size=large_size,
            stride=1,
            groups=nvars * dmodel,
            small_kernel=small_size,
            small_kernel_merged=small_kernel_merged,
        )
        self.norm = nn.BatchNorm1d(dmodel)

        self.ffn1pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, kernel_size=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, kernel_size=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        self.ffn2pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, kernel_size=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, kernel_size=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

    def forward(self, x):
        residual = x
        bsz, nvars, dmodel, patch_num = x.shape
        x = x.reshape(bsz, nvars * dmodel, patch_num)
        x = self.dw(x)
        x = x.reshape(bsz, nvars, dmodel, patch_num)
        x = x.reshape(bsz * nvars, dmodel, patch_num)
        x = self.norm(x)
        x = x.reshape(bsz, nvars, dmodel, patch_num)
        x = x.reshape(bsz, nvars * dmodel, patch_num)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(bsz, nvars, dmodel, patch_num)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, dmodel * nvars, patch_num)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(bsz, dmodel, nvars, patch_num)
        x = x.permute(0, 2, 1, 3)

        return residual + x


class Stage(nn.Module):
    def __init__(
        self,
        ffn_ratio,
        num_blocks,
        large_size,
        small_size,
        dmodel,
        nvars,
        small_kernel_merged=False,
        drop=0.1,
    ):
        super().__init__()
        d_ffn = dmodel * ffn_ratio
        self.blocks = nn.ModuleList(
            [
                Block(
                    large_size=large_size,
                    small_size=small_size,
                    dmodel=dmodel,
                    dff=d_ffn,
                    nvars=nvars,
                    small_kernel_merged=small_kernel_merged,
                    drop=drop,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ModernTCN(nn.Module):
    def __init__(
        self,
        task_name,
        patch_size,
        patch_stride,
        downsample_ratio,
        ffn_ratio,
        num_blocks,
        large_size,
        small_size,
        dims,
        nvars,
        small_kernel_merged=False,
        backbone_dropout=0.1,
        head_dropout=0.1,
        use_multi_scale=True,
        revin=True,
        affine=True,
        subtract_last=False,
        seq_len=512,
        c_in=7,
        individual=False,
        target_window=96,
    ):
        super().__init__()
        self.task_name = task_name
        self.seq_len = seq_len
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Linear(patch_size, dims[0]))
        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                self.downsample_layers.append(
                    nn.Sequential(
                        nn.BatchNorm1d(dims[i]),
                        nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
                    )
                )

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            self.stages.append(
                Stage(
                    ffn_ratio,
                    num_blocks[stage_idx],
                    large_size[stage_idx],
                    small_size[stage_idx],
                    dmodel=dims[stage_idx],
                    nvars=nvars,
                    small_kernel_merged=small_kernel_merged,
                    drop=backbone_dropout,
                )
            )

        patch_num = seq_len // patch_stride
        self.n_vars = c_in
        self.individual = individual
        d_model = dims[self.num_stage - 1]
        if use_multi_scale:
            self.head_nf = d_model * patch_num
        elif patch_num % pow(downsample_ratio, self.num_stage - 1) == 0:
            self.head_nf = d_model * patch_num // pow(downsample_ratio, self.num_stage - 1)
        else:
            self.head_nf = d_model * (patch_num // pow(downsample_ratio, self.num_stage - 1) + 1)
        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout)

        if self.task_name == "anomaly_detection":
            self.head_dection1 = nn.Linear(d_model, self.patch_size)

    def forward_feature(self, x):
        bsz, nvars, _ = x.shape
        x = x.unsqueeze(-2)

        for i in range(self.num_stage):
            bsz, nvars, dmodel, patch_num = x.shape
            x = x.reshape(bsz * nvars, dmodel, patch_num)

            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
                x = x.reshape(bsz, nvars, 1, -1).squeeze(-2)
                x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
                x = self.downsample_layers[i](x)
                x = x.permute(0, 1, 3, 2)
            else:
                if patch_num % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (patch_num % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
                x = self.downsample_layers[i](x)
                _, dmodel_next, patch_num_next = x.shape
                x = x.reshape(bsz, nvars, dmodel_next, patch_num_next)

            x = self.stages[i](x)
        return x

    def detection(self, x):
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, "norm")
            x = x.permute(0, 2, 1)

        x = self.forward_feature(x)
        x = x.permute(0, 1, 3, 2)
        x = self.head_dection1(x)
        bsz, nvars, _, _ = x.shape
        x = x.reshape(bsz, nvars, -1)
        x = x[:, :, : self.seq_len]
        x = x.permute(0, 2, 1)

        if self.revin:
            x = self.revin_layer(x, "denorm")
        return x

    def forward(self, x):
        if self.task_name == "anomaly_detection":
            return self.detection(x)
        return x

    def structural_reparam(self):
        for module in self.modules():
            if hasattr(module, "merge_kernel"):
                module.merge_kernel()


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = ModernTCN(
            task_name=configs.task_name,
            patch_size=configs.patch_size,
            patch_stride=configs.patch_stride,
            downsample_ratio=configs.downsample_ratio,
            ffn_ratio=configs.ffn_ratio,
            num_blocks=configs.num_blocks,
            large_size=configs.large_size,
            small_size=configs.small_size,
            dims=configs.dims,
            nvars=configs.enc_in,
            small_kernel_merged=configs.small_kernel_merged,
            backbone_dropout=configs.dropout,
            head_dropout=configs.head_dropout,
            use_multi_scale=configs.use_multi_scale,
            revin=configs.revin,
            affine=configs.affine,
            subtract_last=configs.subtract_last,
            seq_len=configs.seq_len,
            c_in=configs.enc_in,
            individual=configs.individual,
            target_window=configs.pred_len,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.model(x)


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

BASE_DIR = os.getenv("CIC_BASE_DIR", "/content/drive/MyDrive/15SOTA")
RAW_CSV = os.getenv("CIC_UNSW_RAW_CSV", "/content/drive/MyDrive/CIC-UNSW-NB15/CICFlowMeter_out.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "moderntcn")

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
WINDOW_SIZE = 50
TRAIN_STRIDE = 25
TEST_STRIDE = 10
MAX_TRAIN_WINDOWS = 150_000

# Official MSL-style ModernTCN configuration: patch 8/4, kernels 51 and 5,
# one block of width 8, batch 128, and Adam at 5e-4 with OneCycleLR. Training
# runs for eight epochs and also reports native selection within the official
# two-epoch budget.
EPOCHS = 8
OFFICIAL_BUDGET_EPOCHS = 2
BATCH_SIZE = 128
LR = 5e-4
PATIENCE = 10
PCT_START = 0.3
ANOMALY_RATIO = 0.5   # SMD value (MSL uses 1); affects the PA exhibit only

# Shared entity-window evaluation settings.
MIN_SELECT_EPOCH = 1
TPR_BUDGET = 0.01         # TPR @ 1% budget, threshold from benign val only
READOUTS = ["raw", "std", "maha"]  # raw = E.mean(1) IS the native window score

PATCH_SIZE = 8
PATCH_STRIDE = 4
DOWNSAMPLE_RATIO = 2
FFN_RATIO = 1
NUM_BLOCKS = [1]
LARGE_SIZE = [51]
SMALL_SIZE = [5]
DIMS = [8]
DROPOUT = 0.1
HEAD_DROPOUT = 0.0
USE_MULTI_SCALE = False
SMALL_KERNEL_MERGED = False
REVIN = True
AFFINE = True
SUBTRACT_LAST = False
INDIVIDUAL = False


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


# =============================================================================
# Shared Data Loading
# =============================================================================

def build_windows(X, y, entity_ids, window_size, stride):
    windows, family_labels = [], []
    for eid in np.unique(entity_ids):
        mask = entity_ids == eid
        X_e = X[mask]
        y_e = y[mask]
        n = len(X_e)
        for start in range(0, n - window_size + 1, stride):
            windows.append(X_e[start:start + window_size])
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
    entity_map = dict(enumerate(pd.Categorical(entity_str).categories))
    print(f"  Entities ({', '.join(ENTITY_COLS)}): {len(entity_map)} unique")

    data = raw.drop(columns=[c for c in DROP_COLS if c in raw.columns])
    for c in ENTITY_COLS:
        if c in data.columns:
            data = data.drop(columns=[c])

    var = data.var(numeric_only=True)
    low_var = var[var < 1e-10].index.tolist()
    data = data.drop(columns=low_var)
    print(f"  Removed {len(low_var)} near-constant features -> {data.shape[1]} remain")

    X_all = data.values.astype(np.float64)
    X_all[~np.isfinite(X_all)] = np.nan
    imputer = SimpleImputer(strategy="median")
    X_all = imputer.fit_transform(X_all)
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

    scaler = StandardScaler()
    scaler.fit(X_all[train_indices])
    X_scaled = scaler.transform(X_all).astype(np.float32)

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
    X_test_w, y_test_atk_w, y_test_fam_w = build_windows(
        X_scaled[test_indices],
        y_all[test_indices],
        entity_ids[test_indices],
        WINDOW_SIZE,
        TEST_STRIDE,
    )

    if len(X_train_w) > MAX_TRAIN_WINDOWS:
        idx = np.random.choice(len(X_train_w), MAX_TRAIN_WINDOWS, replace=False)
        X_train_w = X_train_w[idx]
        print(f"  Subsampled train windows: {len(idx)}")

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
                known_atk_idx, max(1, len(known_atk_idx) // 10), replace=False
            )
        X_val_mixed_w = np.concatenate([X_val_w, X_test_w[val_atk_idx]])
        y_val_mixed_w = np.concatenate(
            [np.zeros(len(X_val_w), dtype=int), np.ones(len(val_atk_idx), dtype=int)]
        )
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

    return (
        X_train_w,
        X_val_w,
        X_val_mixed_w,
        y_val_mixed_w,
        X_test_w,
        y_test_atk_w,
        y_test_fam_w,
        data.columns.tolist(),
        scaler,
    )


# =============================================================================
# ModernTCN Baseline
# =============================================================================

def make_moderntcn_config(n_features):
    return SimpleNamespace(
        task_name="anomaly_detection",
        downsample_ratio=DOWNSAMPLE_RATIO,
        ffn_ratio=FFN_RATIO,
        num_blocks=NUM_BLOCKS,
        large_size=LARGE_SIZE,
        small_size=SMALL_SIZE,
        dims=DIMS,
        enc_in=n_features,
        c_out=n_features,
        small_kernel_merged=SMALL_KERNEL_MERGED,
        dropout=DROPOUT,
        head_dropout=HEAD_DROPOUT,
        use_multi_scale=USE_MULTI_SCALE,
        revin=REVIN,
        affine=AFFINE,
        subtract_last=SUBTRACT_LAST,
        seq_len=WINDOW_SIZE,
        pred_len=0,
        individual=INDIVIDUAL,
        kernel_size=PATCH_SIZE,
        patch_size=PATCH_SIZE,
        patch_stride=PATCH_STRIDE,
        decomposition=False,
    )


def make_loader(X, batch_size, shuffle=False, drop_last=False):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
    }
    if shuffle and FIX_TRAIN_SHUFFLE_ORDER:
        generator = torch.Generator()
        generator.manual_seed(SEED)
        kwargs["generator"] = generator
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), **kwargs)


def run_epoch(model, loader, optimizer, criterion, scheduler, device):
    model.train()
    losses = []
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_x)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
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


@torch.no_grad()
def compute_window_scores(model, X, device, batch_size=BATCH_SIZE):
    """Official ModernTCN energy, aggregated from timestep energy to windows."""
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    loader = make_loader(X, batch_size=batch_size, shuffle=False)
    scores = []
    timestep_scores = []
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        energy = torch.mean(criterion(batch_x, outputs), dim=-1)
        timestep_scores.append(energy.detach().cpu().numpy().reshape(-1))
        scores.append(torch.mean(energy, dim=1).detach().cpu().numpy())
    return np.concatenate(scores), np.concatenate(timestep_scores)


def official_threshold(train_scores, test_scores):
    combined = np.concatenate([train_scores, test_scores], axis=0)
    return float(np.percentile(combined, 100 - ANOMALY_RATIO))


@torch.no_grad()
def feature_residuals(model, X, device, batch_size=BATCH_SIZE):
    """Per-window per-feature residual E (N, F): mean over the W positions of
    (x − recon)². Note E.mean(axis=1) IS the native ModernTCN window score
    (timestep energy meaned over time) — raw and the calibrated readouts
    aggregate the SAME residual tensor, so any difference is attributable to
    the readout alone."""
    model.eval()
    loader = make_loader(X, batch_size=batch_size, shuffle=False)
    out = []
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        out.append(((outputs - batch_x) ** 2).mean(dim=1).cpu().numpy())
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

    benign_mask = y_family == 0
    res["median_benign"] = float(np.median(scores_test[benign_mask])) if benign_mask.any() else 0.0
    res["median_attack"] = float(np.median(scores_test[~benign_mask])) if (~benign_mask).any() else 0.0
    res["direction_correct"] = bool(res["median_attack"] >= res["median_benign"])

    per_fam = []
    for fam in sorted(np.unique(y_family)):
        m = y_family == fam
        s = scores_test[m]
        per_fam.append(
            {
                "name": CIC_LABEL_MAP.get(fam, f"fam_{fam}"),
                "n": int(m.sum()),
                "mean": float(s.mean()),
                "median": float(np.median(s)),
                "std": float(s.std()),
                "p90": float(np.percentile(s, 90)),
                "p99": float(np.percentile(s, 99)),
            }
        )
    res["per_family"] = per_fam
    return res


def add_threshold_metrics(result, scores_test, y_attack, y_family, threshold):
    y_pred = (scores_test > threshold).astype(int)
    accuracy = accuracy_score(y_attack, y_pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        y_attack, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_attack, y_pred).ravel()
    result["official_threshold"] = {
        "anomaly_ratio": ANOMALY_RATIO,
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f_score),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    per_family_detection = []
    for fam in sorted(np.unique(y_family)):
        mask = y_family == fam
        n_fam = int(mask.sum())
        detected = int(y_pred[mask].sum())
        per_family_detection.append(
            {
                "name": CIC_LABEL_MAP.get(fam, f"fam_{fam}"),
                "n": n_fam,
                "detected": detected,
                "rate": float(detected / n_fam) if n_fam else 0.0,
                "holdout": bool(fam in HOLDOUT_FAMILIES),
            }
        )
    result["official_threshold"]["per_family_detection"] = per_family_detection
    return result


def print_result(res):
    print(f"\n{'=' * 80}")
    print(f"  {res['model']}")
    print(f"{'=' * 80}")
    print(f"  AUROC (overall)     : {res['auroc_overall']:.6f}")
    print(f"  AUPRC (overall)     : {res['auprc_overall']:.6f}")
    print(
        f"  AUPRC (holdout 8+9) : {res['auprc_holdout']:.6f}  "
        f"95% CI [{res['holdout_ci_low']:.6f}, {res['holdout_ci_high']:.6f}]"
    )
    direction = "CORRECT" if res["direction_correct"] else "INVERTED"
    print(
        f"  Score direction     : {direction}  "
        f"(benign={res['median_benign']:.2e}, attack={res['median_attack']:.2e})"
    )

    print("\n  TPR @ FPR operating points (overall):")
    for k, v in res.get("tpr_at_fpr", {}).items():
        print(f"    {k:<20s} : {v:.4f}")

    if res.get("holdout_tpr_at_fpr"):
        print("\n  TPR @ FPR operating points (holdout 8+9):")
        for k, v in res["holdout_tpr_at_fpr"].items():
            print(f"    {k:<20s} : {v:.4f}")

    if "official_threshold" in res:
        th = res["official_threshold"]
        print(f"\n  Official percentile threshold (anomaly_ratio={th['anomaly_ratio']}):")
        print(f"    threshold={th['threshold']:.6f}")
        print(
            f"    accuracy={th['accuracy']:.4f} precision={th['precision']:.4f} "
            f"recall={th['recall']:.4f} f1={th['f1']:.4f} fpr={th['fpr']:.4f}"
        )
        print(f"    TP={th['tp']}, FP={th['fp']}, TN={th['tn']}, FN={th['fn']}")

    print(f"\n  {'family':<20s} {'n':>7s} {'mean':>10s} {'median':>10s} {'std':>10s} {'p90':>10s} {'p99':>10s}")
    print("  " + "-" * 79)
    for row in res["per_family"]:
        print(
            f"  {row['name']:<20s} {row['n']:>7d} {row['mean']:>10.2e} "
            f"{row['median']:>10.2e} {row['std']:>10.2e} "
            f"{row['p90']:>10.2e} {row['p99']:>10.2e}"
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


def run_experiment(run_seed):
    global SEED, OUTPUT_DIR
    SEED = run_seed
    OUTPUT_DIR = os.path.join(BASE_DIR, f"moderntcn_seed_{SEED}")
    set_seed(SEED)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n" + "=" * 80)
    print("  Shared Data Loading & Windowing")
    print("=" * 80)
    (
        X_train_w,
        X_val_w,
        X_val_mixed_w,
        y_val_mixed_w,
        X_test_w,
        y_test_attack_w,
        y_test_family_w,
        feature_names,
        _scaler,
    ) = load_and_preprocess()

    n_features = X_train_w.shape[2]

    print("\n" + "=" * 80)
    print("  ModernTCN Model Construction")
    print("=" * 80)
    configs = make_moderntcn_config(n_features)
    model = Model(configs).float().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Window: {WINDOW_SIZE} flows x {n_features} features")
    print("  Delta features: disabled")

    train_loader = make_loader(X_train_w, BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = make_loader(X_val_w, BATCH_SIZE, shuffle=False)
    criterion = nn.MSELoss()
    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint.pth")

    print("\n" + "=" * 80)
    print("  ModernTCN Training + per-epoch readout tracking")
    print("=" * 80)
    history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        steps_per_epoch=max(len(train_loader), 1),
        pct_start=PCT_START,
        epochs=EPOCHS,
        max_lr=LR,
    )

    best_val = float("inf")
    patience_ctr = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Three selection rules tracked side by side (scores captured at the moment
    # of selection; the TEST-AP curve is stored for the ORACLE/gap report ONLY):
    #   nat_* — NATIVE selection: lowest benign-val recon loss over the full
    #           run (ModernTCN's own checkpoint rule; never sees labels).
    #           PRIMARY baseline row.
    #   off_* — NATIVE selection within the OFFICIAL budget (epoch <=
    #           OFFICIAL_BUDGET_EPOCHS): the fully faithful paper row.
    #   (plain) — val-AP selection: the selection privilege our own method
    #           uses; labeled secondary row.
    track = {r: {"val_ap": -np.inf, "epoch": None,
                 "scores_test": None, "scores_valb": None,
                 "nat_val_ap": None, "nat_epoch": None,
                 "nat_scores_test": None, "nat_scores_valb": None,
                 "off_val_ap": None, "off_epoch": None,
                 "off_scores_test": None, "off_scores_valb": None,
                 "val_curve": [], "test_curve": []} for r in READOUTS}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        val_loss = validate(model, val_loader, criterion, device)

        E_b = feature_residuals(model, X_val_w, device)
        stats = fit_benign_stats(E_b)
        E_vm = feature_residuals(model, X_val_mixed_w, device)
        sel_ok = epoch >= MIN_SELECT_EPOCH
        vaps = {}
        for r in READOUTS:
            s_vm = apply_readout(E_vm, stats, r)
            vaps[r] = float(average_precision_score(y_val_mixed_w, s_vm))
            track[r]["val_curve"].append(vaps[r])
        # LAZY test scoring: the (large) test pass runs only on epochs where a
        # capture can happen — val loss improved (native/off selection) or some
        # readout's val AP improved (valAP selection). Skipped epochs record
        # NaN in the test curve (gap/oracle uses the computed points only).
        loss_improved = val_loss < best_val
        need_test = loss_improved or any(
            sel_ok and vaps[r] > track[r]["val_ap"] for r in READOUTS)
        cur = {}
        test_loss = float("nan")
        if need_test:
            E_te = feature_residuals(model, X_test_w, device)
            test_loss = float(E_te.mean())  # == full-tensor MSE over test
        for r in READOUTS:
            if need_test:
                s_te = apply_readout(E_te, stats, r).astype(np.float32)
                s_vb = apply_readout(E_b, stats, r).astype(np.float32)
                t_ap = float(average_precision_score(y_test_attack_w, s_te))
                cur[r] = (vaps[r], s_te, s_vb)
                track[r]["test_curve"].append(t_ap)
                if sel_ok and vaps[r] > track[r]["val_ap"]:
                    track[r].update(val_ap=vaps[r], epoch=epoch,
                                    scores_test=s_te, scores_valb=s_vb)
            else:
                track[r]["test_curve"].append(float("nan"))

        elapsed = time.time() - t0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "time_s": elapsed,
                **{f"val_ap_{r}": track[r]["val_curve"][-1] for r in READOUTS},
            }
        )

        if epoch <= 5 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS} "
                f"train={train_loss:.7f} val={val_loss:.7f} "
                f"test={test_loss:.7f} "
                f"val_AP raw {track['raw']['val_curve'][-1]:.4f} "
                f"std {track['std']['val_curve'][-1]:.4f} "
                f"maha {track['maha']['val_curve'][-1]:.4f} ({elapsed:.1f}s)"
            )

        if loss_improved:
            best_val = val_loss
            patience_ctr = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
            # NATIVE selection: exactly the checkpoint their pipeline would keep
            for r in READOUTS:
                track[r].update(nat_val_ap=cur[r][0], nat_epoch=epoch,
                                nat_scores_test=cur[r][1],
                                nat_scores_valb=cur[r][2])
                # ... and within the official 1-2 epoch budget, the faithful row
                if epoch <= OFFICIAL_BUDGET_EPOCHS:
                    track[r].update(off_val_ap=cur[r][0], off_epoch=epoch,
                                    off_scores_test=cur[r][1],
                                    off_scores_valb=cur[r][2])
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (best val_loss={best_val:.7f})")
                break

    model.load_state_dict(best_state)
    print(f"\n  Best val_loss = {best_val:.7f}")

    # Readout comparison on the shared entity-window task
    print("\n" + "=" * 80)
    print("  MODERNTCN READOUT COMPARISON — same residuals, different readout")
    print("  (raw = E.mean(1) IS the native window score, so raw doubles as the")
    print("  paper row. PRIMARY rows use NATIVE selection = lowest benign-val")
    print(f"  recon loss (their checkpoint rule); @off{OFFICIAL_BUDGET_EPOCHS}ep "
          f"= same rule within the official 1-2 epoch budget (fully faithful);")
    print("  valAP-sel rows are the selection-equalized secondary.)")
    print("=" * 80)
    print(f"  {'readout':<20}{'sel_ep':>7}{'val_AP':>9}{'test_AP':>9}"
          f"{'AUROC':>8}{'ho_AP':>8}{'gap':>8}{'TPR@1%':>8}{'FPR_act':>9}{'hoTPR':>7}")
    print("  " + "-" * 92)
    rows = {}

    def report_row(key, label, sel_epoch, val_ap, s_test, s_valb, gap_curve):
        res = evaluate_model(f"ModernTCN[{label}]", s_test,
                             y_test_attack_w, y_test_family_w)
        op = operating_point(s_valb, s_test, y_test_attack_w, y_test_family_w)
        finite = [v for v in gap_curve if np.isfinite(v)]
        gap = (max(finite) - res["auprc_overall"]) if finite else float("nan")
        rows[key] = {"readout": label, "sel_epoch": sel_epoch, "val_ap": val_ap,
                     "gap": gap, "operating_point": op, **res}
        print(f"  {label:<20}{sel_epoch:>7}{val_ap:>9.4f}"
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
    d_std = rows["std_natsel"]["auprc_overall"] - rows["raw_natsel"]["auprc_overall"]
    d_maha = rows["maha_natsel"]["auprc_overall"] - rows["raw_natsel"]["auprc_overall"]

    # Appendix-only native selection: best-val-loss state with the official
    # train+test percentile threshold. This result is not used in the main table.
    print("\n" + "=" * 80)
    print("  APPENDIX EXHIBIT: native selection (best val loss) + official "
          f"percentile threshold (anomaly_ratio={ANOMALY_RATIO})")
    print("=" * 80)
    scores_train, _train_timestep_scores = compute_window_scores(model, X_train_w, device)
    scores_test, _test_timestep_scores = compute_window_scores(model, X_test_w, device)
    window_threshold = official_threshold(scores_train, scores_test)
    print(f"  Train score mean={scores_train.mean():.6f}, std={scores_train.std():.6f}")
    print(f"  Test score mean={scores_test.mean():.6f}, std={scores_test.std():.6f}")
    print(f"  Window threshold={window_threshold:.6f}")
    result = evaluate_model("ModernTCN[native-exhibit]", scores_test,
                            y_test_attack_w, y_test_family_w)
    result = add_threshold_metrics(
        result, scores_test, y_test_attack_w, y_test_family_w, window_threshold
    )
    print_result(result)
    rows["native_exhibit"] = result
    print_result(rows["raw_natsel"])
    print_result(rows["maha_natsel"] if d_maha >= d_std else rows["std_natsel"])

    rows["config"] = {
        "seed": SEED,
        "window_size": WINDOW_SIZE,
        "train_stride": TRAIN_STRIDE,
        "test_stride": TEST_STRIDE,
        "features": feature_names,
        "delta_features": False,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "patience": PATIENCE,
        "anomaly_ratio": ANOMALY_RATIO,
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "num_blocks": NUM_BLOCKS,
        "large_size": LARGE_SIZE,
        "small_size": SMALL_SIZE,
        "dims": DIMS,
        "dropout": DROPOUT,
        "revin": REVIN,
    }

    curves = {r: {"val": track[r]["val_curve"], "test": track[r]["test_curve"]}
              for r in READOUTS}
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(deep_convert(rows), f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(deep_convert({"history": history, "curves": curves}), f, indent=2)
    np.save(os.path.join(OUTPUT_DIR, "scores_test_native_exhibit.npy"), scores_test)

    print(f"\n  Results saved to {OUTPUT_DIR}/")
    return rows


def main():
    all_results = []
    for s in RUN_SEEDS:
        print(f"\n\n{'=' * 80}\n  STARTING MODERNTCN RUN WITH SEED {s}\n{'=' * 80}")
        all_results.append(run_experiment(s))

    print(f"\n\n{'=' * 80}\n  AGGREGATED MODERNTCN RESULTS OVER {len(RUN_SEEDS)} SEEDS\n{'=' * 80}")
    agg_keys = ([f"{r}_natsel" for r in READOUTS]       # PRIMARY (native sel)
                + [f"{r}_official" for r in READOUTS]   # faithful 2-ep budget
                + READOUTS                              # secondary (val-AP sel)
                + ["native_exhibit"])
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
        rs = [r for r in all_results if k in r and "operating_point" in r[k]]
        if not rs:
            continue
        t = np.array([r[k]["operating_point"]["tpr"] for r in rs])
        f_ = np.array([r[k]["operating_point"]["fpr_actual"] for r in rs])
        print(f"    {k:<16s}: TPR {t.mean():.4f}  actual FPR {f_.mean():.4f}")

    out_path = os.path.join(BASE_DIR, "moderntcn_aggregated_results.json")
    with open(out_path, "w") as f:
        json.dump(deep_convert(all_results), f, indent=2)
    print(f"\nSaved aggregated results to {out_path}")


if __name__ == "__main__":
    main()
