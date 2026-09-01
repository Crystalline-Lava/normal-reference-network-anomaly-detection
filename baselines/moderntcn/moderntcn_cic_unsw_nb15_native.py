# Adapted in part from luodhhh/ModernTCN, MIT License.
# Source and full upstream notice: ../../THIRD_PARTY_NOTICES.md

"""
ModernTCN CIC-UNSW-NB15 baseline — proposed split + ModernTCN's native processing.

Companion to `moderntcn_cic_unsw_nb15_entity.py` (the entity-grouped variant). This
file keeps the proposed split so both methods evaluate on the same flows, but runs
the ModernTCN anomaly-detection recipe with **ModernTCN's own native data processing**
(verified against the official luodhhh/ModernTCN `ModernTCN-detection` code, which
follows Time-Series-Library / Anomaly-Transformer):

  Split (= proposed pipeline, time-split only):
    - cutoff 2015-01-23
    - train = benign pre-cutoff (per-entity first 85%, identical index construction)
    - val   = benign pre-cutoff (last 15%)
    - test  = ALL post-cutoff flows
    - the `val_mixed` known-attack injection/removal is DROPPED (it is the proposed
      threshold-tuning trick; ModernTCN uses its own percentile threshold)

  Processing (= ModernTCN native):
    - NO entity grouping: windows slide across each split's flows as one continuous,
      time-ordered stream (windows may span Src IP boundaries — faithful to the
      SegLoaders)
    - cleaning/scaling = np.nan_to_num + StandardScaler (fit on train); NO winsorize,
      NO median-impute, keep ALL feature columns
    - per-TIMESTEP reconstruction energy = mean(MSE(x, recon)) over feature dim
    - threshold = percentile(concat(train_energy, test_energy), 100 - anomaly_ratio)
    - point-adjustment, then accuracy / precision / recall / f1

  Metrics: BOTH the native point-adjusted P/R/F1 AND raw per-flow AUROC/AUPRC
  (no point-adjust) so there is a closer-to-comparable number alongside the proposed method.

Faithfulness notes / deviations:
  - CIC flows contain `inf`; we map non-finite -> nan before np.nan_to_num (vanilla
    ModernTCN datasets have no infs).
  - val_mixed removed -> the test set here is all post-cutoff flows, a slightly larger
    flow population than the proposed method's post-removal test set (by design choice).
  - WIN_SIZE defaults to ModernTCN's AD default (100); set it to 50 for tighter parity
    with the proposed method (one-line change). The trailing < WIN_SIZE test remainder is
    truncated, matching the official non-overlapping `thre` loader.
  - Point-adjustment inflates F1 (well documented); that is why raw per-flow
    AUROC/AUPRC are reported alongside.

This script is fully self-contained (standalone for publication): the ModernTCN
model definition and all shared helpers/constants are inlined below, so it does
NOT import any sibling project file. The only import-time side effect is a
try/except-wrapped Colab `drive.mount`, which is a no-op off Colab.
"""

from __future__ import annotations

import copy
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
# Inlined ModernTCN implementation (standalone — verified against
# luodhhh/ModernTCN `ModernTCN-detection`)
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

# Official MSL-style ModernTCN training configuration. ``EPOCHS`` also defines
# the OneCycleLR horizon, so the schedule completes within the two-epoch budget.
EPOCHS = 2
BATCH_SIZE = 128
LR = 5e-4
PATIENCE = 10
PCT_START = 0.3
ANOMALY_RATIO = 0.5

# Official MSL-style ModernTCN architecture configuration.
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
        seq_len=WIN_SIZE,
        pred_len=0,
        individual=INDIVIDUAL,
        kernel_size=PATCH_SIZE,
        patch_size=PATCH_SIZE,
        patch_stride=PATCH_STRIDE,
        decomposition=False,
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
RESULTS_DIR = os.path.join(BASE_DIR, "results_moderntcn_native")
OUTPUT_DIR = os.path.join(RESULTS_DIR, f"seed_{SEED}")

WIN_SIZE = 100      # ModernTCN AD default; set 50 for tighter parity with proposed method
TRAIN_STEP = 1      # faithful PSM dense windows; raise to speed up training
SKIP_TRAINING = False


# =============================================================================
# Proposed split + native ModernTCN preprocessing
# =============================================================================

def load_and_preprocess_native():
    """Proposed flow split with ModernTCN-native cleaning/scaling, returned as
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
    """Anomaly-Transformer / ModernTCN point-adjustment.

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
      - per-flow (top-level keys): ModernTCN's native per-timestep unit
      - window-level (res["window_level"]): aggregated to the proposed per-window unit
    Threshold-based numbers are per-flow only (a threshold isn't comparable across
    methods anyway)."""
    res = {"model": "ModernTCN (native pipeline)"}

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

    # ── Threshold-free ranking metrics, PER-FLOW (ModernTCN native granularity) ──
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
    print(f"  Point-adjusted (ModernTCN native, anomaly_ratio={pa['anomaly_ratio']}):")
    print(f"    threshold={pa['threshold']:.6f}")
    print(f"    accuracy={pa['accuracy']:.4f} precision={pa['precision']:.4f} "
          f"recall={pa['recall']:.4f} f1={pa['f1']:.4f}")

    rt = res["raw_threshold"]
    print("\n  Raw at-threshold (no point adjustment):")
    print(f"    precision={rt['precision']:.4f} recall={rt['recall']:.4f} "
          f"fpr={rt['fpr']:.4f}")
    print(f"    TP={rt['tp']}, FP={rt['fp']}, TN={rt['tn']}, FN={rt['fn']}")

    print("\n  Per-FLOW ranking (ModernTCN native per-timestep unit):")
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
    print("  Proposed split + ModernTCN-native preprocessing")
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
    print("  ModernTCN model construction")
    print("=" * 80)
    configs = make_moderntcn_config(n_features)
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

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
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
        "processing": "ModernTCN-native (no entity grouping, nan_to_num+StandardScaler)",
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
        print(f"\n\n{'=' * 80}\n  MODERNTCN (NATIVE PIPELINE) — SEED {s}\n{'=' * 80}")
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
# Local smoke test (no dataset / no Drive needed):  python moderntcn_cic_unsw_nb15_native.py --smoke
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
    configs = make_moderntcn_config(nvars)
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
