from __future__ import annotations
import hashlib
import json, math, time, os, random
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
# Required by PyTorch for deterministic cublas kernels on CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
)
from scipy.stats import gaussian_kde
try:
    from google.colab import drive
except ImportError:
    drive = None


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    return value in {"1", "true", "yes", "on"}


def env_text(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def load_torch_checkpoint(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        print("  Falling back to trusted checkpoint load (weights_only=False) for local .pt file ...")
        return torch.load(path, map_location=map_location, weights_only=False)


if drive is not None:
    drive.mount("/content/drive")

# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Fixed five-seed set reported in the manuscript.
RUN_SEEDS = (42, 456, 7, 789, 1024)
SEED = RUN_SEEDS[0]  # Active model seed; reassigned by run_experiment().
# False matches the stochastic settings used for the reported runs.
# Set True to request deterministic PyTorch algorithms where available.
STRICT_DETERMINISM = False
# False uses the reported DataLoader shuffle behavior.
FIX_TRAIN_SHUFFLE_ORDER = False

# Dynamic label encoding is built at preprocessing time for BCCC-CSE-CIC-IDS2018.
LABEL_TO_ID = {"Benign": 0}
CIC_LABEL_MAP = {0: "Benign"}

# Paths (adjust for your environment: Colab / local)
try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd()
COLAB_BASE_DIR = "/content/drive/MyDrive/cic_only"
COLAB_BCCC_ROOT = (
    "/content/drive/MyDrive/Kaggle_Datasets/kagglehub_cache/datasets/"
    "bcccdatasets/large-scale-ids-dataset-bccc-cse-cic-ids2018/versions/2"
)
LOCAL_BCCC_ROOT = PROJECT_DIR / "BCCC-CSE-CIC-IDS2018"
BASE_DIR = os.getenv(
    "CIC_BASE_DIR",
    COLAB_BASE_DIR if os.path.exists(COLAB_BASE_DIR) else str(PROJECT_DIR),
)
DATA_ROOT = os.getenv(
    "BCCC_CSE_CIC_IDS2018_ROOT",
    COLAB_BCCC_ROOT if os.path.exists(COLAB_BCCC_ROOT) else str(LOCAL_BCCC_ROOT),
)
OUTPUT_DIR = os.path.join(BASE_DIR, "transformer_ae_bccc_cse_cic_ids2018")
TARGET_FILE_GLOB = "*.csv"
CSV_CHUNK_SIZE = 100_000
EVAL_WINDOW_BATCH_SIZE = 4096
CACHE_VERSION = "bccc_cse_cic_ids2018_preproc_v8_family_aware_val"
ENABLE_PREPROCESS_CACHE = True
CACHE_DIR = os.path.join(BASE_DIR, "bccc_cse_cic_ids2018_cache")
DATA_SEED = RUN_SEEDS[0]  # Fixed split/cache seed shared by all model runs.
ENABLE_FULL_VAL_WINDOW_CACHE = True
FULL_VAL_WINDOW_CACHE_DTYPE = np.float16

TARGET_METADATA_CANDIDATES = {
    "timestamp": ["timestamp"],
    "label": ["label"],
    "src_ip": ["src_ip"],
    "dst_ip": ["dst_ip", "destination_ip", "dst ip"],
    "src_port": ["src_port", "source_port", "src port"],
    "dst_port": ["dst_port", "destination_port", "dst port"],
    "protocol": ["protocol"],
}
REQUIRED_METADATA_KEYS = {"timestamp", "label", "src_ip"}
DROP_COLS = {
    "flowid", "timestamp", "srcip", "srcport", "dstip", "dstport",
    "protocol", "label",
}

# Entity grouping & windowing
WINDOW_SIZE = 50                # flows per window
TRAIN_STRIDE = 25               # stride for training windows (50% overlap)
TEST_STRIDE = 10                # stride for test windows (dense overlap)
USE_DELTA_FEATURES = True       # temporal deltas often help window models on flow logs
ENTITY_GROUP_MODE = "service_flow"  # one of: src_ip, service_flow, five_tuple
WINDOW_STORAGE_DTYPE = np.float16
PREPROCESS_SAMPLE_ROWS = 500_000
MAX_KDE_TRAIN_FLOWS = 100_000
MIXED_VAL_GROUPS = 2
MIXED_TEST_GROUPS = 1
VAL_MIXED_LOOKBACK = 4
ATTACK_VAL_GROUP_RATIO = 0.33  # retained in cache/config metadata
ATTACK_TEST_POLICY = "family_coverage_temporal_singletons"  # one of: family_coverage, family_coverage_temporal_singletons, latest_mixed_days
MAX_TRAIN_WINDOWS = 150_000
MAX_VAL_BENIGN_WINDOWS = 10_000
MAX_VAL_ATTACK_WINDOWS = 3_000
MIN_VAL_ATTACK_WINDOWS_PER_FAMILY = 128
VAL_SELECTION_MODE = "full_stream"  # one of: sampled_mixed, full_stream

# Reported evaluation protocol
# Label census: set True to scan the per-family label distribution across all
# files and EXIT (no training). Run this first; then record the holdout choice
# before training and fill HOLDOUT_FAMILY_NAMES below.
RUN_LABEL_CENSUS = False
# Unseen-family holdout, pre-registered from the
# label census (34 files, 48.18M rows): the web-application-attack cluster —
# semantically coherent, disjoint from the dominant volumetric DoS/bruteforce/bot
# traffic, and rare (874 flows total). This mirrors the CIC-UNSW-NB15
# Shellcode+Worms holdout rationale
# (rare + semantically distinct). NOTE: Infiltration has only 5 flows in this
# BCCC build — degenerate, NOT in the holdout, excluded from all claims.
HOLDOUT_FAMILY_NAMES: List[str] = [
    "Brute_Force_Web",
    "Brute_Force_XSS",
    "SQL_Injection",
]
MIN_SELECT_EPOCH = 20          # selection warmup (match main pipeline)
TPR_BUDGET = 0.01              # TPR @ 1% benign-val budget
UNION_SPLIT = (0.50, 0.50)     # reported budget split (prediction, sphere)
CENTER_RECOMPUTE_EVERY = 5     # match main pipeline's center schedule
SINGLETON_ATTACK_VAL_FRACTION = 0.4

# Architecture
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 4
D_FF = 256
DROPOUT = 0.1
BOTTLENECK_DIM = 64
EXPERIMENT_GROUP = "proposed_s1_bccc_cse_cic_ids2018"

# Training
# Default to retraining; set BCCC_CSE_CIC_IDS2018_SKIP_TRAINING=1 to load a checkpoint.
SKIP_TRAINING = env_flag("BCCC_CSE_CIC_IDS2018_SKIP_TRAINING", False)
OUTPUT_TAG = env_text("BCCC_CSE_CIC_IDS2018_OUTPUT_TAG", "")
EPOCHS = 150
BATCH_SIZE = 128
LR = 5e-5
WEIGHT_DECAY = 1e-5
FULL_VAL_PATIENCE = 15
LAMBDA_VAR = 0.0                # S1 does not use variance regularization
MIN_VARIANCE = 0.5              # minimum z variance to maintain
LAMBDA_COV = 0.0                # S1 does not use covariance regularization
COV_WARMUP = 0                  # ramp λ_cov linearly over this many epochs (0=disabled)
MASK_RATIO = 0.15               # fraction of flows masked during training
# S1 retained detector: train-time SVDD with dimension-summed distance,
# sum((z-c.detach())**2, dim=1).mean().
TRAIN_SVDD = True
LAMBDA_SVDD = 3.0

# Optional contrastive pseudo-anomaly augmentation (disabled in reported runs).
LAMBDA_CONTRAST = 0.0           # set >0 to enable; tested at 0.05–0.5, destabilised training
CONTRAST_MARGIN = 50.0          # min sphere distance for pseudo-anomalies
CONTRAST_WARMUP = 30            # delay until model stabilises

# Random seed control
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


def architecture_tag() -> str:
    return (
        f"dm{D_MODEL}_h{N_HEADS}_l{N_LAYERS}_ff{D_FF}_"
        f"bn{BOTTLENECK_DIM}_w{WINDOW_SIZE}_d{int(USE_DELTA_FEATURES)}"
    )


def experiment_root_dir() -> str:
    return os.path.join(BASE_DIR, EXPERIMENT_GROUP, architecture_tag())


def output_dir_for_seed(seed: int) -> str:
    base_name = f"seed_{seed}"
    tag = OUTPUT_TAG or "retrain_splitfix"
    safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("._-")
    if not safe_tag:
        safe_tag = "retrain_splitfix"
    # SKIP_TRAINING re-eval must resolve to the SAME directory the training run
    # wrote its best_*.pt checkpoints to (previously it dropped the tag and
    # looked in seed_<s>/, which has no checkpoints under this group).
    return os.path.join(experiment_root_dir(), f"{base_name}_{safe_tag}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading & Preprocessing — Sequence Windows
# ═══════════════════════════════════════════════════════════════════════════════

def canonicalize_column_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name).strip() if ch.isalnum())


def resolve_column_name(column_lookup: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        actual = column_lookup.get(canonicalize_column_name(candidate))
        if actual is not None:
            return actual
    return None


def entity_metadata_keys() -> List[str]:
    if ENTITY_GROUP_MODE == "src_ip":
        return ["src_ip"]
    if ENTITY_GROUP_MODE == "service_flow":
        return ["src_ip", "dst_ip", "dst_port", "protocol"]
    if ENTITY_GROUP_MODE == "five_tuple":
        return ["src_ip", "src_port", "dst_ip", "dst_port", "protocol"]
    raise ValueError(f"Unknown ENTITY_GROUP_MODE={ENTITY_GROUP_MODE!r}")


def canonicalize_label(label: Any) -> str:
    text = str(label).strip()
    if text.lower() == "benign":
        return "Benign"
    return text.replace(" ", "_")


def holdout_family_ids(label_to_id: Dict[str, int]) -> List[int]:
    """Map the pre-registered HOLDOUT_FAMILY_NAMES to label ids.

    Hard-fails on unknown names so a typo can never silently evaluate with an
    empty holdout."""
    canon = {canonicalize_label(k): v for k, v in label_to_id.items()}
    ids: List[int] = []
    missing: List[str] = []
    for name in HOLDOUT_FAMILY_NAMES:
        key = canonicalize_label(name)
        if key in canon:
            ids.append(int(canon[key]))
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"HOLDOUT_FAMILY_NAMES not found in dataset labels: {missing}. "
            f"Known labels: {sorted(label_to_id)}")
    return ids


def clip_to_bounds(X: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> np.ndarray:
    return np.clip(X, lower_bounds, upper_bounds)


def file_signature(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    stat = p.stat()
    return {
        "path": str(p.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
    }


def file_spec_signature(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file": file_signature(spec["path"]),
        "row_slice": list(spec.get("row_slice", (0.0, 1.0))),
    }


def stable_cache_key(prefix: str, payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()[:16]
    return f"{prefix}_{digest}"


def cache_path(cache_key: str) -> Path:
    return Path(CACHE_DIR) / cache_key


def stage_cache_key(base_key: str, stage: str, extra: Optional[Dict[str, Any]] = None) -> str:
    payload = {"base_key": base_key, "stage": stage}
    if extra:
        payload.update(extra)
    return stable_cache_key(stage, payload)


def save_cache_payload(cache_key: str, payload: Dict[str, Any]) -> None:
    cache_dir = cache_path(cache_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {"arrays": {}, "objects": {}}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            filename = f"{key}.npy"
            target_path = cache_dir / filename
            tmp_path = cache_dir / f"{filename}.tmp"
            with open(tmp_path, "wb") as f:
                np.save(f, value, allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target_path)
            meta["arrays"][key] = filename
        else:
            meta["objects"][key] = value
    meta_path = cache_dir / "meta.pkl"
    tmp_meta_path = cache_dir / "meta.pkl.tmp"
    with open(tmp_meta_path, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_meta_path, meta_path)


def load_cache_payload(cache_key: str) -> Optional[Dict[str, Any]]:
    cache_dir = cache_path(cache_key)
    meta_path = cache_dir / "meta.pkl"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        payload = dict(meta["objects"])
        for key, filename in meta["arrays"].items():
            payload[key] = np.load(cache_dir / filename, mmap_mode="r")
        return payload
    except Exception as exc:
        print(f"\nIgnoring corrupted cache at {cache_dir}: {exc}")
        return None


def save_pickle_atomic(path: Path, payload: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def full_val_file_cache_key(preproc_cache_key: str, file_spec: Dict[str, Any]) -> str:
    return stable_cache_key(
        "full_val_file_windows",
        {
            "preproc_cache_key": preproc_cache_key,
            "file_spec": file_spec_signature(file_spec),
            "window_size": WINDOW_SIZE,
            "stride": TEST_STRIDE,
            "use_delta": USE_DELTA_FEATURES,
            "entity_group_mode": ENTITY_GROUP_MODE,
            "storage_dtype": np.dtype(FULL_VAL_WINDOW_CACHE_DTYPE).str,
        },
    )


def build_full_val_file_window_cache(
    file_spec: Dict[str, Any],
    preproc_ctx: Dict[str, Any],
    cache_key: str,
) -> None:
    cache_dir = cache_path(cache_key)
    meta_path = cache_dir / "meta.pkl"
    if meta_path.exists():
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            if meta.get("complete"):
                return
        except Exception:
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_names: List[Dict[str, Any]] = []
    total_windows = 0
    print(f"    Caching full-val windows for {file_spec['path'].name} ...")
    batch_idx = 0
    for X_w, y_attack, y_family, _ in iter_window_batches_for_files(
        file_specs=[file_spec],
        canonical_feature_order=preproc_ctx["canonical_feature_order"],
        imputer=preproc_ctx["imputer"],
        keep_mask=preproc_ctx["keep_mask"],
        winsor_low=preproc_ctx["winsor_low"],
        winsor_high=preproc_ctx["winsor_high"],
        scaler=preproc_ctx["scaler"],
        label_to_id=preproc_ctx["label_to_id"],
        stride=TEST_STRIDE,
        batch_windows=EVAL_WINDOW_BATCH_SIZE,
    ):
        x_name = f"X_{batch_idx:05d}.npy"
        y_name = f"y_{batch_idx:05d}.npy"
        fam_name = f"fam_{batch_idx:05d}.npy"
        save_npy_atomic(cache_dir / x_name, X_w.astype(FULL_VAL_WINDOW_CACHE_DTYPE, copy=False))
        save_npy_atomic(cache_dir / y_name, y_attack.astype(np.int8, copy=False))
        save_npy_atomic(cache_dir / fam_name, y_family.astype(np.int64, copy=False))
        shard_names.append({
            "x": x_name,
            "y": y_name,
            "family": fam_name,
            "n": int(len(X_w)),
        })
        total_windows += int(len(X_w))
        save_pickle_atomic(
            meta_path,
            {
                "complete": False,
                "shards": shard_names,
                "total_windows": total_windows,
                "storage_dtype": np.dtype(FULL_VAL_WINDOW_CACHE_DTYPE).str,
                "file_name": file_spec["path"].name,
            },
        )
        batch_idx += 1

    save_pickle_atomic(
        meta_path,
        {
            "complete": True,
            "shards": shard_names,
            "total_windows": total_windows,
            "storage_dtype": np.dtype(FULL_VAL_WINDOW_CACHE_DTYPE).str,
            "file_name": file_spec["path"].name,
        },
    )


def iter_cached_full_val_window_batches(
    split_ctx: Dict[str, Any],
    preproc_ctx: Dict[str, Any],
):
    preproc_cache_key = preproc_ctx["preproc_cache_key"]
    all_specs = list(split_ctx["benign_specs"]) + list(split_ctx["attack_specs"])
    for file_spec in all_specs:
        file_cache_key = full_val_file_cache_key(preproc_cache_key, file_spec)
        build_full_val_file_window_cache(file_spec, preproc_ctx, file_cache_key)
        cache_dir = cache_path(file_cache_key)
        with open(cache_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        for shard in meta["shards"]:
            X_w = np.load(cache_dir / shard["x"], mmap_mode="r").astype(np.float32, copy=False)
            y_attack = np.load(cache_dir / shard["y"], mmap_mode="r").astype(np.int64, copy=False)
            y_family = np.load(cache_dir / shard["family"], mmap_mode="r").astype(np.int64, copy=False)
            yield X_w, y_attack, y_family


def stage_cache_complete(payload: Optional[Dict[str, Any]]) -> bool:
    return payload is not None and bool(payload.get("complete", True))


def restore_np_random_state(payload: Optional[Dict[str, Any]]) -> None:
    if payload is not None and "np_random_state" in payload:
        np.random.set_state(payload["np_random_state"])


class ArrayWindowDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X

    def __len__(self) -> int:
        return int(len(self.X))

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.X[idx], dtype=torch.float32)


class ReservoirRows:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.count_seen = 0
        self.data: Optional[np.ndarray] = None

    def add_many(self, rows: np.ndarray) -> None:
        if self.capacity <= 0 or rows.size == 0:
            return
        rows = np.asarray(rows, dtype=np.float32)
        if self.data is None:
            self.data = np.empty((self.capacity, rows.shape[1]), dtype=np.float32)
        for row in rows:
            self.count_seen += 1
            if self.count_seen <= self.capacity:
                self.data[self.count_seen - 1] = row
            else:
                slot = np.random.randint(0, self.count_seen)
                if slot < self.capacity:
                    self.data[slot] = row

    def finalize(self) -> np.ndarray:
        if self.data is None:
            return np.zeros((0, 0), dtype=np.float32)
        size = min(self.count_seen, self.capacity)
        return self.data[:size].copy()

    def export_state(self, prefix: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            f"{prefix}_capacity": int(self.capacity),
            f"{prefix}_count_seen": int(self.count_seen),
        }
        if self.data is not None:
            size = min(self.count_seen, self.capacity)
            payload[f"{prefix}_data"] = self.data[:size].copy()
        return payload

    @classmethod
    def from_state(cls, payload: Dict[str, Any], prefix: str) -> "ReservoirRows":
        reservoir = cls(int(payload[f"{prefix}_capacity"]))
        reservoir.count_seen = int(payload[f"{prefix}_count_seen"])
        data = payload.get(f"{prefix}_data")
        if data is not None:
            data_arr = np.asarray(data, dtype=np.float32)
            reservoir.data = np.empty((reservoir.capacity, data_arr.shape[1]), dtype=np.float32)
            reservoir.data[:len(data_arr)] = data_arr
        return reservoir


class ReservoirWindows:
    def __init__(self, capacity: int, storage_dtype: np.dtype = WINDOW_STORAGE_DTYPE):
        self.capacity = int(capacity)
        self.storage_dtype = np.dtype(storage_dtype)
        self.count_seen = 0
        self.windows: Optional[np.ndarray] = None
        self.family_labels: Optional[np.ndarray] = None

    def add(self, window: np.ndarray, family_label: int) -> None:
        if self.capacity <= 0:
            return
        if self.windows is None:
            shape = (self.capacity,) + tuple(window.shape)
            self.windows = np.empty(shape, dtype=self.storage_dtype)
            self.family_labels = np.empty(self.capacity, dtype=np.int64)
        self.count_seen += 1
        if self.count_seen <= self.capacity:
            slot = self.count_seen - 1
        else:
            slot = np.random.randint(0, self.count_seen)
            if slot >= self.capacity:
                return
        self.windows[slot] = window.astype(self.storage_dtype, copy=False)
        self.family_labels[slot] = int(family_label)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.windows is None or self.family_labels is None:
            feat_dim = 0
            return (
                np.zeros((0, WINDOW_SIZE, feat_dim), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
            )
        size = min(self.count_seen, self.capacity)
        X = self.windows[:size].astype(np.float32)
        y_family = self.family_labels[:size].copy()
        y_attack = (y_family != 0).astype(np.int64)
        return X, y_attack, y_family

    def export_state(self, prefix: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            f"{prefix}_capacity": int(self.capacity),
            f"{prefix}_count_seen": int(self.count_seen),
            f"{prefix}_storage_dtype": self.storage_dtype.str,
        }
        if self.windows is not None and self.family_labels is not None:
            size = min(self.count_seen, self.capacity)
            payload[f"{prefix}_windows"] = self.windows[:size].copy()
            payload[f"{prefix}_family_labels"] = self.family_labels[:size].copy()
        return payload

    @classmethod
    def from_state(cls, payload: Dict[str, Any], prefix: str) -> "ReservoirWindows":
        storage_dtype = np.dtype(payload.get(f"{prefix}_storage_dtype", WINDOW_STORAGE_DTYPE))
        reservoir = cls(int(payload[f"{prefix}_capacity"]), storage_dtype=storage_dtype)
        reservoir.count_seen = int(payload[f"{prefix}_count_seen"])
        windows = payload.get(f"{prefix}_windows")
        family_labels = payload.get(f"{prefix}_family_labels")
        if windows is not None and family_labels is not None:
            windows_arr = np.asarray(windows, dtype=storage_dtype)
            labels_arr = np.asarray(family_labels, dtype=np.int64)
            reservoir.windows = np.empty((reservoir.capacity,) + tuple(windows_arr.shape[1:]), dtype=storage_dtype)
            reservoir.family_labels = np.empty(reservoir.capacity, dtype=np.int64)
            reservoir.windows[:len(windows_arr)] = windows_arr
            reservoir.family_labels[:len(labels_arr)] = labels_arr
        return reservoir


class FamilyAwareReservoirWindows:
    def __init__(self, capacities: Dict[int, int], storage_dtype: np.dtype = WINDOW_STORAGE_DTYPE):
        self.storage_dtype = np.dtype(storage_dtype)
        self.capacities = {
            int(fam): int(cap)
            for fam, cap in capacities.items()
            if int(cap) > 0
        }
        self.reservoirs: Dict[int, ReservoirWindows] = {
            fam: ReservoirWindows(cap, storage_dtype=self.storage_dtype)
            for fam, cap in self.capacities.items()
        }

    def add(self, window: np.ndarray, family_label: int) -> None:
        fam = int(family_label)
        if fam == 0:
            return
        reservoir = self.reservoirs.get(fam)
        if reservoir is None:
            return
        reservoir.add(window, fam)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_parts: List[np.ndarray] = []
        y_attack_parts: List[np.ndarray] = []
        y_family_parts: List[np.ndarray] = []
        for fam in sorted(self.reservoirs):
            X_fam, y_attack_fam, y_family_fam = self.reservoirs[fam].finalize()
            if len(X_fam) == 0:
                continue
            X_parts.append(X_fam)
            y_attack_parts.append(y_attack_fam)
            y_family_parts.append(y_family_fam)
        if not X_parts:
            feat_dim = 0
            return (
                np.zeros((0, WINDOW_SIZE, feat_dim), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
            )
        X = np.concatenate(X_parts, axis=0)
        y_attack = np.concatenate(y_attack_parts, axis=0)
        y_family = np.concatenate(y_family_parts, axis=0)
        if len(X) > 1:
            order = np.random.permutation(len(X))
            X = X[order]
            y_attack = y_attack[order]
            y_family = y_family[order]
        return X, y_attack, y_family

    def export_state(self, prefix: str) -> Dict[str, Any]:
        return {
            f"{prefix}_capacities": dict(self.capacities),
            f"{prefix}_storage_dtype": self.storage_dtype.str,
            f"{prefix}_reservoirs": {
                int(fam): reservoir.export_state("reservoir")
                for fam, reservoir in self.reservoirs.items()
            },
        }

    @classmethod
    def from_state(cls, payload: Dict[str, Any], prefix: str) -> "FamilyAwareReservoirWindows":
        storage_dtype = np.dtype(payload.get(f"{prefix}_storage_dtype", WINDOW_STORAGE_DTYPE))
        capacities = {
            int(fam): int(cap)
            for fam, cap in dict(payload.get(f"{prefix}_capacities", {})).items()
        }
        wrapper = cls(capacities, storage_dtype=storage_dtype)
        reservoir_payloads = dict(payload.get(f"{prefix}_reservoirs", {}))
        wrapper.reservoirs = {
            int(fam): ReservoirWindows.from_state(state, "reservoir")
            for fam, state in reservoir_payloads.items()
        }
        return wrapper


class ReservoirLabeledRows:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.count_seen = 0
        self.rows: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None

    def add(self, row: np.ndarray, label: int) -> None:
        if self.capacity <= 0:
            return
        row = np.asarray(row, dtype=np.float32)
        if self.rows is None:
            self.rows = np.empty((self.capacity, row.shape[0]), dtype=np.float32)
            self.labels = np.empty(self.capacity, dtype=np.int64)
        self.count_seen += 1
        if self.count_seen <= self.capacity:
            slot = self.count_seen - 1
        else:
            slot = np.random.randint(0, self.count_seen)
            if slot >= self.capacity:
                return
        self.rows[slot] = row
        self.labels[slot] = int(label)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.rows is None or self.labels is None:
            return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int64)
        size = min(self.count_seen, self.capacity)
        return self.rows[:size].copy(), self.labels[:size].copy()


def discover_dataset_files() -> List[Path]:
    root = Path(DATA_ROOT)
    if root.is_file():
        files = [root]
    else:
        files = sorted(root.rglob(TARGET_FILE_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No BCCC-CSE-CIC-IDS2018 CSV files found under {DATA_ROOT!r}. "
            "Update DATA_ROOT to your dataset folder."
        )
    return files


def resolve_dataset_specs(csv_files: List[Path]) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, str]]:
    specs: List[Dict[str, Any]] = []
    canonical_feature_order: Optional[List[str]] = None
    canonical_feature_labels: Dict[str, str] = {}

    for path in csv_files:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        column_lookup = {canonicalize_column_name(c): c for c in header}
        metadata_cols = {}
        for key, candidates in TARGET_METADATA_CANDIDATES.items():
            actual = resolve_column_name(column_lookup, candidates)
            if actual is None and key in REQUIRED_METADATA_KEYS:
                raise KeyError(f"Missing required column {key!r} in {path}")
            if actual is not None:
                metadata_cols[key] = actual

        file_feature_canon = [
            canonicalize_column_name(col)
            for col in header
            if canonicalize_column_name(col) not in DROP_COLS
        ]
        if canonical_feature_order is None:
            canonical_feature_order = file_feature_canon
            canonical_feature_labels = {
                canonicalize_column_name(col): col
                for col in header
                if canonicalize_column_name(col) in canonical_feature_order
            }
        else:
            missing = [c for c in canonical_feature_order if c not in column_lookup]
            if missing:
                raise KeyError(f"File {path} is missing feature columns: {missing[:10]}")

        specs.append({
            "path": path,
            "header": header,
            "column_lookup": column_lookup,
            "metadata_cols": metadata_cols,
        })

    assert canonical_feature_order is not None
    return specs, canonical_feature_order, canonical_feature_labels


DAY_KEY_RE = re.compile(
    r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)_(\d{2})_(\d{2})_(\d{4})"
)


def extract_day_key(path: Path) -> Tuple[str, pd.Timestamp]:
    match = DAY_KEY_RE.match(path.stem.lower())
    if match is None:
        raise ValueError(f"Could not parse BCCC-CSE-CIC-IDS2018 day key from {path.name}")
    weekday, day, month, year = match.groups()
    key = f"{weekday}_{day}_{month}_{year}"
    date = pd.Timestamp(f"{year}-{month}-{day}")
    return key, date


def flatten_grouped_specs(grouped_specs: List[Tuple[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for _, specs in grouped_specs:
        flat.extend(specs)
    return flat


def plan_fair_file_splits(
    file_specs: List[Dict[str, Any]],
    day_attack_rows: Optional[Dict[str, int]] = None,
    file_label_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    def clone_with_row_slice(spec: Dict[str, Any], row_slice: Tuple[float, float]) -> Dict[str, Any]:
        cloned = dict(spec)
        cloned["row_slice"] = (float(row_slice[0]), float(row_slice[1]))
        return cloned

    benign_groups: Dict[str, List[Dict[str, Any]]] = {}
    attack_groups: Dict[str, List[Dict[str, Any]]] = {}
    day_order: Dict[str, pd.Timestamp] = {}

    for spec in file_specs:
        day_key, day_ts = extract_day_key(spec["path"])
        spec["day_key"] = day_key
        spec["day_ts"] = day_ts
        day_order[day_key] = day_ts
        is_benign = "benign" in spec["path"].name.lower()
        (benign_groups if is_benign else attack_groups).setdefault(day_key, []).append(spec)

    benign_day_keys = set(benign_groups)
    attack_day_keys = set(attack_groups)
    mixed_day_keys = sorted(benign_day_keys & attack_day_keys, key=lambda key: day_order[key])
    if len(mixed_day_keys) < MIXED_VAL_GROUPS + MIXED_TEST_GROUPS:
        raise RuntimeError(
            "Not enough day-aligned mixed groups for validation/test splitting."
        )

    candidate_val_day_keys = mixed_day_keys[:-MIXED_TEST_GROUPS]
    if len(candidate_val_day_keys) < MIXED_VAL_GROUPS:
        raise RuntimeError("Not enough mixed day groups left for validation.")
    lookback = max(MIXED_VAL_GROUPS, VAL_MIXED_LOOKBACK)
    recent_val_candidates = candidate_val_day_keys[-lookback:]
    day_attack_rows = day_attack_rows or {}
    ranked_val_candidates = sorted(
        recent_val_candidates,
        key=lambda key: (int(day_attack_rows.get(key, 0)), day_order[key]),
        reverse=True,
    )
    val_day_keys = sorted(
        ranked_val_candidates[:MIXED_VAL_GROUPS],
        key=lambda key: day_order[key],
    )
    test_day_keys = mixed_day_keys[-MIXED_TEST_GROUPS:]
    eval_day_keys = set(val_day_keys) | set(test_day_keys)
    earliest_eval_ts = min(day_order[key] for key in eval_day_keys)

    train_day_keys = sorted(
        [
            key for key in benign_day_keys
            if key not in eval_day_keys and day_order[key] < earliest_eval_ts
        ],
        key=lambda key: day_order[key],
    )
    unused_benign_day_keys = sorted(
        [
            key for key in benign_day_keys
            if key not in eval_day_keys and key not in train_day_keys
        ],
        key=lambda key: day_order[key],
    )

    if not train_day_keys:
        raise RuntimeError("Day-aligned split leaves no earlier benign days for training.")

    train_benign = flatten_grouped_specs([(key, benign_groups[key]) for key in train_day_keys])
    val_benign = flatten_grouped_specs([(key, benign_groups[key]) for key in val_day_keys])
    test_benign = flatten_grouped_specs([(key, benign_groups[key]) for key in test_day_keys])

    if ATTACK_TEST_POLICY in {"family_coverage", "family_coverage_temporal_singletons"} and file_label_counts:
        attack_specs_sorted = sorted(
            [spec for specs in attack_groups.values() for spec in specs],
            key=lambda spec: (spec["day_ts"], str(spec["path"].name)),
        )
        attack_labels = sorted({
            label
            for counts in file_label_counts.values()
            for label, count in counts.items()
            if label != "Benign" and int(count) > 0
        })
        use_temporal_singletons = ATTACK_TEST_POLICY == "family_coverage_temporal_singletons"
        family_candidate_paths: Dict[str, set[str]] = {}
        for label in attack_labels:
            family_candidate_paths[label] = {
                str(spec["path"])
                for spec in attack_specs_sorted
                if int(file_label_counts.get(str(spec["path"]), {}).get(label, 0)) > 0
            }

        # Evaluation rule: holdout families are assigned entirely to test.
        # They are masked
        # out of the selection metric anyway, so any val presence is pure waste
        # AND halves the (already small) test holdout sample. Forcing them to
        # test makes the exclusion physical, not just a mask.
        holdout_forced = set(HOLDOUT_FAMILY_NAMES)
        missing_holdout = sorted(holdout_forced - set(attack_labels))
        if holdout_forced and missing_holdout:
            raise RuntimeError(
                f"HOLDOUT_FAMILY_NAMES not found in the label inventory: "
                f"{missing_holdout}. Known attack labels: {attack_labels}")
        holdout_forced_path_keys: set[str] = set()
        for label in sorted(holdout_forced):
            for path_key in sorted(family_candidate_paths.get(label, set())):
                holdout_forced_path_keys.add(path_key)
        for path_key in sorted(holdout_forced_path_keys):
            rider_labels = sorted(
                label
                for label, count in file_label_counts.get(path_key, {}).items()
                if label != "Benign" and label not in holdout_forced and int(count) > 0
            )
            if rider_labels:
                print(
                    f"  WARNING: holdout-forced file {Path(path_key).name} also "
                    f"contains non-holdout families {rider_labels}; those rows "
                    f"ride along fully into test (no val presence)."
                )

        test_attack_specs_by_key: Dict[Tuple[str, Tuple[float, float]], Dict[str, Any]] = {}
        singleton_split_path_keys: set[str] = set()
        singleton_split_families: List[str] = []
        test_attack_labels: List[str] = []
        for label in attack_labels:
            candidates = [
                spec for spec in attack_specs_sorted
                if int(file_label_counts.get(str(spec["path"]), {}).get(label, 0)) > 0
            ]
            if not candidates:
                continue
            candidate_paths = family_candidate_paths.get(label, set())
            if label in holdout_forced or candidate_paths <= holdout_forced_path_keys:
                # Holdout family (or a family living only inside holdout-forced
                # files): every candidate file goes to test in full.
                for spec in candidates:
                    test_attack_specs_by_key[(str(spec["path"]), (0.0, 1.0))] = dict(spec)
                test_attack_labels.append(label)
                continue
            if use_temporal_singletons and len(candidate_paths) == 1:
                chosen = candidates[0]
                path_key = str(chosen["path"])
                singleton_split_path_keys.add(path_key)
                singleton_split_families.append(label)
                test_attack_specs_by_key[(path_key, (SINGLETON_ATTACK_VAL_FRACTION, 1.0))] = clone_with_row_slice(
                    chosen,
                    (SINGLETON_ATTACK_VAL_FRACTION, 1.0),
                )
            else:
                chosen = max(candidates, key=lambda spec: (spec["day_ts"], str(spec["path"].name)))
                test_attack_specs_by_key[(str(chosen["path"]), (0.0, 1.0))] = dict(chosen)
            test_attack_labels.append(label)
        test_attack_path_keys = {path_key for path_key, _ in test_attack_specs_by_key.keys()}
        test_attack = list(test_attack_specs_by_key.values())
        val_attack = []
        for spec in attack_specs_sorted:
            path_key = str(spec["path"])
            if path_key in singleton_split_path_keys:
                val_attack.append(clone_with_row_slice(spec, (0.0, SINGLETON_ATTACK_VAL_FRACTION)))
            elif path_key not in test_attack_path_keys:
                val_attack.append(dict(spec))
        val_attack_day_keys = sorted({spec["day_key"] for spec in val_attack}, key=lambda key: day_order[key])
        test_attack_day_keys = sorted({spec["day_key"] for spec in test_attack}, key=lambda key: day_order[key])

        plan_name = (
            "temporal benign + family-covering attack test + temporal singleton attack split"
            if use_temporal_singletons else
            "temporal benign + family-covering attack test"
        )
        print(f"\nFair split plan ({plan_name}):")
        print("  Train benign days:", ", ".join(train_day_keys))
        print("  Val benign days  :", ", ".join(val_day_keys))
        print("  Test benign days :", ", ".join(test_day_keys))
        print("  Val attack days  :", ", ".join(val_attack_day_keys))
        print("  Test attack days :", ", ".join(test_attack_day_keys))
        print("  Test families    :", ", ".join(test_attack_labels))
        if holdout_forced_path_keys:
            print(
                "  Holdout families assigned entirely to test "
                f"({', '.join(sorted(holdout_forced))}): "
                + ", ".join(sorted(Path(p).name for p in holdout_forced_path_keys))
            )
        if singleton_split_families:
            print(
                "  Singleton families temporally split:",
                ", ".join(sorted(set(singleton_split_families))),
            )
            print(
                f"  Singleton val fraction: first {SINGLETON_ATTACK_VAL_FRACTION:.0%} for validation, "
                f"last {1.0 - SINGLETON_ATTACK_VAL_FRACTION:.0%} for test"
            )
    else:
        val_attack = flatten_grouped_specs([(key, attack_groups[key]) for key in val_day_keys])
        test_attack = flatten_grouped_specs([(key, attack_groups[key]) for key in test_day_keys])

        print("\nFair split plan (day-aligned before windowing):")
        print("  Train benign days:", ", ".join(train_day_keys))
        print("  Val mixed days   :", ", ".join(val_day_keys))
        print("  Test mixed days  :", ", ".join(test_day_keys))
        if val_day_keys:
            print(
                "  Val attack rows  :",
                ", ".join(f"{key}={int(day_attack_rows.get(key, 0)):,}" for key in val_day_keys),
            )
        if test_day_keys:
            print(
                "  Test attack rows :",
                ", ".join(f"{key}={int(day_attack_rows.get(key, 0)):,}" for key in test_day_keys),
            )
    if unused_benign_day_keys:
        print("  Unused benign days:", ", ".join(unused_benign_day_keys))

    return {
        "train_benign": train_benign,
        "val_benign": val_benign,
        "test_benign": test_benign,
        "val_attack": val_attack,
        "test_attack": test_attack,
    }


def iter_csv_chunks(file_spec: Dict[str, Any], canonical_feature_order: List[str]):
    feature_actual_cols = [file_spec["column_lookup"][c] for c in canonical_feature_order]
    usecols = list(file_spec["metadata_cols"].values()) + feature_actual_cols
    dedup_usecols = list(dict.fromkeys(usecols))
    row_slice = tuple(file_spec.get("row_slice", (0.0, 1.0)))
    total_rows = int(file_spec.get("total_rows", 0))
    slice_start = max(0.0, min(1.0, float(row_slice[0])))
    slice_end = max(slice_start, min(1.0, float(row_slice[1])))
    start_row = int(math.floor(slice_start * total_rows))
    end_row = int(math.floor(slice_end * total_rows)) if slice_end < 1.0 else total_rows
    if row_slice != (0.0, 1.0) and total_rows <= 0:
        raise ValueError(f"row_slice requires total_rows metadata for {file_spec['path']}")

    rows_seen = 0
    for chunk in pd.read_csv(
        file_spec["path"],
        usecols=dedup_usecols,
        chunksize=CSV_CHUNK_SIZE,
        low_memory=False,
    ):
        chunk_start = rows_seen
        chunk_end = rows_seen + len(chunk)
        rows_seen = chunk_end
        if row_slice != (0.0, 1.0):
            keep_start = max(0, start_row - chunk_start)
            keep_end = min(len(chunk), end_row - chunk_start)
            if keep_start >= keep_end:
                if chunk_start >= end_row:
                    break
                continue
            chunk = chunk.iloc[keep_start:keep_end].copy()
        if len(chunk):
            yield chunk
        if row_slice != (0.0, 1.0) and chunk_end >= end_row:
            break


def chunk_to_feature_matrix(
    chunk: pd.DataFrame,
    file_spec: Dict[str, Any],
    canonical_feature_order: List[str],
) -> np.ndarray:
    n_rows = len(chunk)
    X = np.full((n_rows, len(canonical_feature_order)), np.nan, dtype=np.float64)
    for j, canon_name in enumerate(canonical_feature_order):
        actual = file_spec["column_lookup"][canon_name]
        X[:, j] = pd.to_numeric(chunk[actual], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    X[~np.isfinite(X)] = np.nan
    return X


def transform_features(
    X_raw: np.ndarray,
    imputer: SimpleImputer,
    keep_mask: np.ndarray,
    winsor_low: np.ndarray,
    winsor_high: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    X_imp = imputer.transform(X_raw)
    X_imp = X_imp[:, keep_mask]
    X_imp = clip_to_bounds(X_imp, winsor_low, winsor_high)
    return scaler.transform(X_imp).astype(np.float32, copy=False)


def build_entity_keys(chunk: pd.DataFrame, metadata_cols: Dict[str, str]) -> np.ndarray:
    if ENTITY_GROUP_MODE == "src_ip":
        return chunk[metadata_cols["src_ip"]].astype(str).to_numpy()

    key_parts: List[np.ndarray] = [chunk[metadata_cols["src_ip"]].astype(str).to_numpy()]
    for key in entity_metadata_keys()[1:]:
        actual = metadata_cols.get(key)
        if actual is None:
            continue
        key_parts.append(chunk[actual].astype(str).to_numpy())

    if len(key_parts) == 1:
        return key_parts[0]

    entity_chunk = key_parts[0]
    for part in key_parts[1:]:
        entity_chunk = np.char.add(np.char.add(entity_chunk, "|"), part)
    return entity_chunk


def materialize_window(base_window: np.ndarray, use_deltas: bool) -> np.ndarray:
    if not use_deltas:
        return base_window.astype(np.float32, copy=False)
    deltas = np.diff(base_window, axis=0, prepend=base_window[:1])
    return np.concatenate([base_window, deltas], axis=1).astype(np.float32, copy=False)


def scan_label_distribution(
    file_specs: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, Dict[str, int]], Dict[str, int]]:
    label_to_id: Dict[str, int] = {"Benign": 0}
    label_counts: Dict[str, int] = {}
    file_label_counts: Dict[str, Dict[str, int]] = {}
    day_attack_rows: Dict[str, int] = {}

    for file_spec in file_specs:
        label_iter = pd.read_csv(
            file_spec["path"],
            usecols=[file_spec["metadata_cols"]["label"]],
            chunksize=CSV_CHUNK_SIZE,
            low_memory=False,
        )
        path_key = str(file_spec["path"])
        day_key = str(file_spec.get("day_key") or extract_day_key(file_spec["path"])[0])
        file_counts: Dict[str, int] = file_label_counts.setdefault(path_key, {})
        for label_chunk in label_iter:
            labels = label_chunk.iloc[:, 0].map(canonicalize_label)
            for label in labels.unique():
                label_count = int((labels == label).sum())
                if label not in label_to_id:
                    label_to_id[label] = len(label_to_id)
                label_counts[label] = label_counts.get(label, 0) + label_count
                file_counts[label] = file_counts.get(label, 0) + label_count
                if label != "Benign":
                    day_attack_rows[day_key] = day_attack_rows.get(day_key, 0) + label_count
    return label_to_id, label_counts, file_label_counts, day_attack_rows


def sample_windows_from_files(
    file_specs: List[Dict[str, Any]],
    canonical_feature_order: List[str],
    imputer: SimpleImputer,
    keep_mask: np.ndarray,
    winsor_low: np.ndarray,
    winsor_high: np.ndarray,
    scaler: StandardScaler,
    label_to_id: Dict[str, int],
    reservoir: ReservoirWindows,
    stride: int,
    kde_flows: Optional[ReservoirRows] = None,
) -> int:
    n_windows_seen = 0
    for file_spec in file_specs:
        file_tag = file_spec["path"].name
        state_by_entity: Dict[str, Dict[str, Any]] = {}
        print(f"  Sampling windows from {file_tag} ...")
        for chunk in iter_csv_chunks(file_spec, canonical_feature_order):
            metadata_cols = file_spec["metadata_cols"]
            labels = chunk[metadata_cols["label"]].map(canonicalize_label)
            y_chunk = np.array([label_to_id[label] for label in labels], dtype=np.int64)
            entity_chunk = build_entity_keys(chunk, metadata_cols)
            X_raw = chunk_to_feature_matrix(chunk, file_spec, canonical_feature_order)
            X_scaled = transform_features(
                X_raw, imputer, keep_mask, winsor_low, winsor_high, scaler,
            )

            if kde_flows is not None:
                benign_mask = y_chunk == 0
                if benign_mask.any():
                    kde_flows.add_many(X_scaled[benign_mask])

            for i in range(len(X_scaled)):
                entity_key = entity_chunk[i]
                st = state_by_entity.setdefault(entity_key, {
                    "features": [],
                    "labels": [],
                    "buffer_start": 0,
                    "rows_seen": 0,
                    "next_start": 0,
                })
                st["features"].append(X_scaled[i])
                st["labels"].append(int(y_chunk[i]))
                st["rows_seen"] += 1

                while st["next_start"] + WINDOW_SIZE <= st["rows_seen"]:
                    rel = st["next_start"] - st["buffer_start"]
                    base_window = np.stack(st["features"][rel:rel + WINDOW_SIZE]).astype(
                        np.float32, copy=False
                    )
                    family_label = int(np.max(st["labels"][rel:rel + WINDOW_SIZE]))
                    reservoir.add(materialize_window(base_window, USE_DELTA_FEATURES), family_label)
                    n_windows_seen += 1
                    st["next_start"] += stride

                trim_upto = st["next_start"] - st["buffer_start"]
                if trim_upto > 0 and trim_upto >= WINDOW_SIZE:
                    st["features"] = st["features"][trim_upto:]
                    st["labels"] = st["labels"][trim_upto:]
                    st["buffer_start"] += trim_upto
    return n_windows_seen


def attack_family_row_counts_for_specs(
    attack_specs: List[Dict[str, Any]],
    file_label_counts: Dict[str, Dict[str, int]],
    label_to_id: Dict[str, int],
) -> Dict[int, int]:
    family_counts: Dict[int, int] = {}
    for spec in attack_specs:
        path = str(spec["path"])
        row_slice = tuple(spec.get("row_slice", (0.0, 1.0)))
        slice_fraction = max(0.0, min(1.0, float(row_slice[1]) - float(row_slice[0])))
        for label, count in file_label_counts.get(path, {}).items():
            if label == "Benign" or int(count) <= 0:
                continue
            fam = int(label_to_id[label])
            family_counts[fam] = family_counts.get(fam, 0) + max(1, int(round(float(count) * slice_fraction)))
    return family_counts


def allocate_validation_attack_family_capacities(
    family_row_counts: Dict[int, int],
    total_capacity: int,
    min_per_family: int = MIN_VAL_ATTACK_WINDOWS_PER_FAMILY,
) -> Dict[int, int]:
    family_ids = sorted(int(fam) for fam, count in family_row_counts.items() if int(count) > 0)
    if total_capacity <= 0 or not family_ids:
        return {}

    n_families = len(family_ids)
    if total_capacity < n_families:
        ranked = sorted(
            family_ids,
            key=lambda fam: (int(family_row_counts.get(fam, 0)), fam),
            reverse=True,
        )
        return {fam: 1 for fam in ranked[:total_capacity]}

    base = min(int(min_per_family), total_capacity // n_families)
    capacities = {fam: base for fam in family_ids}
    remaining = total_capacity - base * n_families
    if remaining <= 0:
        return capacities

    weights = np.sqrt(np.array([max(int(family_row_counts[fam]), 1) for fam in family_ids], dtype=np.float64))
    weights_sum = float(weights.sum())
    if weights_sum <= 0.0:
        for idx in range(remaining):
            capacities[family_ids[idx % n_families]] += 1
        return capacities

    raw_extra = remaining * weights / weights_sum
    extra_floor = np.floor(raw_extra).astype(int)
    for fam, extra in zip(family_ids, extra_floor):
        capacities[fam] += int(extra)
    assigned = int(extra_floor.sum())
    leftover = remaining - assigned
    if leftover > 0:
        fractional = raw_extra - extra_floor
        order = np.argsort(-fractional)
        for idx in order[:leftover]:
            capacities[family_ids[int(idx)]] += 1
    return capacities


def iter_window_batches_for_files(
    file_specs: List[Dict[str, Any]],
    canonical_feature_order: List[str],
    imputer: SimpleImputer,
    keep_mask: np.ndarray,
    winsor_low: np.ndarray,
    winsor_high: np.ndarray,
    scaler: StandardScaler,
    label_to_id: Dict[str, int],
    stride: int,
    batch_windows: int = EVAL_WINDOW_BATCH_SIZE,
):
    windows: List[np.ndarray] = []
    attack_labels: List[int] = []
    family_labels: List[int] = []
    last_flows: List[np.ndarray] = []

    for file_spec in file_specs:
        file_tag = file_spec["path"].name
        state_by_entity: Dict[str, Dict[str, Any]] = {}
        print(f"    Streaming {file_tag} ...")
        for chunk in iter_csv_chunks(file_spec, canonical_feature_order):
            metadata_cols = file_spec["metadata_cols"]
            labels = chunk[metadata_cols["label"]].map(canonicalize_label)
            y_chunk = np.array([label_to_id[label] for label in labels], dtype=np.int64)
            entity_chunk = build_entity_keys(chunk, metadata_cols)
            X_raw = chunk_to_feature_matrix(chunk, file_spec, canonical_feature_order)
            X_scaled = transform_features(
                X_raw, imputer, keep_mask, winsor_low, winsor_high, scaler,
            )

            for i in range(len(X_scaled)):
                entity_key = entity_chunk[i]
                st = state_by_entity.setdefault(entity_key, {
                    "features": [],
                    "labels": [],
                    "buffer_start": 0,
                    "rows_seen": 0,
                    "next_start": 0,
                })
                st["features"].append(X_scaled[i])
                st["labels"].append(int(y_chunk[i]))
                st["rows_seen"] += 1

                while st["next_start"] + WINDOW_SIZE <= st["rows_seen"]:
                    rel = st["next_start"] - st["buffer_start"]
                    base_window = np.stack(st["features"][rel:rel + WINDOW_SIZE]).astype(
                        np.float32, copy=False
                    )
                    family_label = int(np.max(st["labels"][rel:rel + WINDOW_SIZE]))
                    windows.append(materialize_window(base_window, USE_DELTA_FEATURES))
                    attack_labels.append(int(family_label != 0))
                    family_labels.append(family_label)
                    last_flows.append(base_window[-1].astype(np.float32, copy=False))
                    st["next_start"] += stride

                    if len(windows) >= batch_windows:
                        yield (
                            np.stack(windows).astype(np.float32, copy=False),
                            np.array(attack_labels, dtype=np.int64),
                            np.array(family_labels, dtype=np.int64),
                            np.stack(last_flows).astype(np.float32, copy=False),
                        )
                        windows, attack_labels, family_labels, last_flows = [], [], [], []

                trim_upto = st["next_start"] - st["buffer_start"]
                if trim_upto > 0 and trim_upto >= WINDOW_SIZE:
                    st["features"] = st["features"][trim_upto:]
                    st["labels"] = st["labels"][trim_upto:]
                    st["buffer_start"] += trim_upto

    if windows:
        yield (
            np.stack(windows).astype(np.float32, copy=False),
            np.array(attack_labels, dtype=np.int64),
            np.array(family_labels, dtype=np.int64),
            np.stack(last_flows).astype(np.float32, copy=False),
        )


def load_and_preprocess():
    """Fit preprocessing on train-benign only and keep validation/test fully disjoint."""
    global LABEL_TO_ID, CIC_LABEL_MAP

    set_seed(DATA_SEED)
    csv_files = discover_dataset_files()
    file_specs, canonical_feature_order, canonical_feature_labels = resolve_dataset_specs(csv_files)
    for spec in file_specs:
        day_key, day_ts = extract_day_key(spec["path"])
        spec["day_key"] = day_key
        spec["day_ts"] = day_ts

    cache_key = stable_cache_key(
        "preproc",
        {
            "version": CACHE_VERSION,
            "data_seed": DATA_SEED,
            "files": [file_signature(spec["path"]) for spec in file_specs],
            "window_size": WINDOW_SIZE,
            "train_stride": TRAIN_STRIDE,
            "test_stride": TEST_STRIDE,
            "use_delta": USE_DELTA_FEATURES,
            "entity_group_mode": ENTITY_GROUP_MODE,
            "preprocess_sample_rows": PREPROCESS_SAMPLE_ROWS,
            "max_kde_train_flows": MAX_KDE_TRAIN_FLOWS,
            "max_train_windows": MAX_TRAIN_WINDOWS,
            "max_val_benign_windows": MAX_VAL_BENIGN_WINDOWS,
            "max_val_attack_windows": MAX_VAL_ATTACK_WINDOWS,
            "min_val_attack_windows_per_family": MIN_VAL_ATTACK_WINDOWS_PER_FAMILY,
            "mixed_val_groups": MIXED_VAL_GROUPS,
            "mixed_test_groups": MIXED_TEST_GROUPS,
            "val_mixed_lookback": VAL_MIXED_LOOKBACK,
            "attack_test_policy": ATTACK_TEST_POLICY,
            "attack_val_group_ratio": ATTACK_VAL_GROUP_RATIO,
            "singleton_attack_val_fraction": SINGLETON_ATTACK_VAL_FRACTION,
            # The split plan depends on the holdout set, so changing it must
            # invalidate the preprocessing cache.
            "holdout_family_names": sorted(HOLDOUT_FAMILY_NAMES),
        },
    )
    if ENABLE_PREPROCESS_CACHE:
        cached = load_cache_payload(cache_key)
        if cached is not None:
            print(f"\nLoading preprocessing cache from {cache_path(cache_key)}")
            LABEL_TO_ID = dict(cached["label_to_id"])
            CIC_LABEL_MAP = dict(cached["cic_label_map"])
            preproc_ctx = dict(cached["preproc_ctx"])
            preproc_ctx["preproc_cache_key"] = cache_key
            val_stream_ctx = dict(cached["val_stream_ctx"])
            val_stream_ctx["split_name"] = "val"
            val_stream_ctx["enable_full_window_cache"] = ENABLE_FULL_VAL_WINDOW_CACHE
            test_stream_ctx = dict(cached["test_stream_ctx"])
            test_stream_ctx["split_name"] = "test"
            test_stream_ctx["enable_full_window_cache"] = False
            return {
                "X_train_w": cached["X_train_w"],
                "X_val_w": cached["X_val_w"],
                "X_val_mixed_w": cached["X_val_mixed_w"],
                "y_val_mixed_w": cached["y_val_mixed_w"],
                "X_train_flows": cached["X_train_flows"],
                "feature_names": cached["feature_names"],
                "feature_names_full": cached["feature_names_full"],
                "preproc_ctx": preproc_ctx,
                "val_stream_ctx": val_stream_ctx,
                "test_stream_ctx": test_stream_ctx,
            }

    label_stage_key = stage_cache_key(cache_key, "label_scan")
    cached_labels = load_cache_payload(label_stage_key) if ENABLE_PREPROCESS_CACHE else None
    if stage_cache_complete(cached_labels):
        print(f"\nLoading label scan cache from {cache_path(label_stage_key)}")
        label_to_id = dict(cached_labels["label_to_id"])
        label_counts = dict(cached_labels["label_counts"])
        file_label_counts = {
            str(path): dict(counts)
            for path, counts in dict(cached_labels["file_label_counts"]).items()
        }
        day_attack_rows = {
            str(day): int(count)
            for day, count in dict(cached_labels["day_attack_rows"]).items()
        }
    else:
        if cached_labels is not None:
            print(f"\nResuming label scan cache from {cache_path(label_stage_key)}")
            label_to_id = dict(cached_labels["label_to_id"])
            label_counts = dict(cached_labels["label_counts"])
            file_label_counts = {
                str(path): dict(counts)
                for path, counts in dict(cached_labels.get("file_label_counts", {})).items()
            }
            day_attack_rows = {
                str(day): int(count)
                for day, count in dict(cached_labels.get("day_attack_rows", {})).items()
            }
            processed_label_paths = set(cached_labels.get("processed_paths", []))
        else:
            label_to_id = {"Benign": 0}
            label_counts = {}
            file_label_counts = {}
            day_attack_rows = {}
            processed_label_paths = set()

        for file_spec in file_specs:
            path_key = str(file_spec["path"])
            if path_key in processed_label_paths:
                continue
            label_iter = pd.read_csv(
                file_spec["path"],
                usecols=[file_spec["metadata_cols"]["label"]],
                chunksize=CSV_CHUNK_SIZE,
                low_memory=False,
            )
            for label_chunk in label_iter:
                labels = label_chunk.iloc[:, 0].map(canonicalize_label)
                for label in labels.unique():
                    label_count = int((labels == label).sum())
                    if label not in label_to_id:
                        label_to_id[label] = len(label_to_id)
                    label_counts[label] = label_counts.get(label, 0) + label_count
                    file_counts = file_label_counts.setdefault(path_key, {})
                    file_counts[label] = file_counts.get(label, 0) + label_count
                    if label != "Benign":
                        day_key = str(file_spec["day_key"])
                        day_attack_rows[day_key] = day_attack_rows.get(day_key, 0) + label_count
            processed_label_paths.add(path_key)
            if ENABLE_PREPROCESS_CACHE:
                save_cache_payload(
                    label_stage_key,
                    {
                        "complete": len(processed_label_paths) == len(file_specs),
                        "processed_paths": sorted(processed_label_paths),
                        "label_to_id": label_to_id,
                        "label_counts": label_counts,
                        "file_label_counts": file_label_counts,
                        "day_attack_rows": day_attack_rows,
                    },
                )

    split_plan = plan_fair_file_splits(
        file_specs,
        day_attack_rows=day_attack_rows,
        file_label_counts=file_label_counts,
    )
    total_rows_by_path = {
        str(path): int(sum(int(count) for count in counts.values()))
        for path, counts in file_label_counts.items()
    }
    for split_name in ("train_benign", "val_benign", "test_benign", "val_attack", "test_attack"):
        for spec in split_plan[split_name]:
            spec["total_rows"] = total_rows_by_path.get(str(spec["path"]), 0)

    val_attack_specs = list(split_plan["val_attack"])
    test_attack_specs = list(split_plan["test_attack"])
    val_attack_labels = {
        label
        for spec in val_attack_specs
        for label, count in file_label_counts.get(str(spec["path"]), {}).items()
        if label != "Benign" and int(count) > 0
    }
    test_attack_labels = {
        label
        for spec in test_attack_specs
        for label, count in file_label_counts.get(str(spec["path"]), {}).items()
        if label != "Benign" and int(count) > 0
    }
    label_name_by_id = {idx: name for name, idx in label_to_id.items()}
    val_attack_family_row_counts = attack_family_row_counts_for_specs(
        val_attack_specs, file_label_counts, label_to_id
    )
    val_attack_family_capacities = allocate_validation_attack_family_capacities(
        val_attack_family_row_counts,
        MAX_VAL_ATTACK_WINDOWS,
        min_per_family=MIN_VAL_ATTACK_WINDOWS_PER_FAMILY,
    )

    sample_stage_key = stage_cache_key(cache_key, "sample_stats")
    cached_sample = load_cache_payload(sample_stage_key) if ENABLE_PREPROCESS_CACHE else None
    if stage_cache_complete(cached_sample):
        print(f"\nLoading preprocessing-sample cache from {cache_path(sample_stage_key)}")
        feature_names = list(cached_sample["feature_names"])
        imputer = cached_sample["imputer"]
        keep_mask = np.asarray(cached_sample["keep_mask"], dtype=bool)
        winsor_low = np.asarray(cached_sample["winsor_low"], dtype=np.float64)
        winsor_high = np.asarray(cached_sample["winsor_high"], dtype=np.float64)
        sample_rows = int(cached_sample["sample_rows"])
        restore_np_random_state(cached_sample)
    else:
        if cached_sample is not None:
            print(f"\nResuming preprocessing-sample cache from {cache_path(sample_stage_key)}")
            benign_sample = ReservoirRows.from_state(cached_sample, "sample_reservoir")
            processed_sample_paths = set(cached_sample.get("processed_paths", []))
            restore_np_random_state(cached_sample)
        else:
            benign_sample = ReservoirRows(PREPROCESS_SAMPLE_ROWS)
            processed_sample_paths = set()
        print("\nFitting preprocessing sample from train-benign traffic only ...")
        for file_spec in split_plan["train_benign"]:
            path_key = str(file_spec["path"])
            if path_key in processed_sample_paths:
                continue
            print(f"  Sampling {file_spec['path'].name} ...")
            for chunk in iter_csv_chunks(file_spec, canonical_feature_order):
                labels = chunk[file_spec["metadata_cols"]["label"]].map(canonicalize_label)
                benign_mask = labels.to_numpy() == "Benign"
                if not benign_mask.any():
                    continue
                X_raw = chunk_to_feature_matrix(chunk.loc[benign_mask], file_spec, canonical_feature_order)
                benign_sample.add_many(X_raw.astype(np.float32))
            processed_sample_paths.add(path_key)
            if ENABLE_PREPROCESS_CACHE:
                save_cache_payload(
                    sample_stage_key,
                    {
                        "complete": False,
                        "processed_paths": sorted(processed_sample_paths),
                        "np_random_state": np.random.get_state(),
                        **benign_sample.export_state("sample_reservoir"),
                    },
                )

        sample_matrix = benign_sample.finalize().astype(np.float64, copy=False)
        if sample_matrix.size == 0:
            raise RuntimeError("No benign train rows were sampled from BCCC-CSE-CIC-IDS2018.")

        sample_rows = int(sample_matrix.shape[0])
        sample_matrix[~np.isfinite(sample_matrix)] = np.nan
        sample_imputer = SimpleImputer(strategy="median")
        sample_imputed = sample_imputer.fit_transform(sample_matrix)
        keep_mask = np.var(sample_imputed, axis=0) >= 1e-10
        feature_names = [
            canonical_feature_labels[canon]
            for canon, keep in zip(canonical_feature_order, keep_mask)
            if keep
        ]
        if not feature_names:
            raise RuntimeError("No usable numeric features remained after preprocessing.")

        imputer = SimpleImputer(strategy="median")
        imputer.fit(sample_matrix)
        sample_kept = imputer.transform(sample_matrix)[:, keep_mask]
        winsor_low = np.quantile(sample_kept, 0.001, axis=0).astype(np.float64)
        winsor_high = np.quantile(sample_kept, 0.999, axis=0).astype(np.float64)

        if ENABLE_PREPROCESS_CACHE:
            print(f"\nSaving preprocessing-sample cache to {cache_path(sample_stage_key)}")
            save_cache_payload(
                sample_stage_key,
                {
                    "complete": True,
                    "feature_names": feature_names,
                    "imputer": imputer,
                    "keep_mask": keep_mask.astype(bool),
                    "winsor_low": winsor_low,
                    "winsor_high": winsor_high,
                    "sample_rows": sample_rows,
                    "np_random_state": np.random.get_state(),
                },
            )

    scaler_stage_key = stage_cache_key(cache_key, "scaler")
    cached_scaler = load_cache_payload(scaler_stage_key) if ENABLE_PREPROCESS_CACHE else None
    if stage_cache_complete(cached_scaler):
        print(f"\nLoading scaler cache from {cache_path(scaler_stage_key)}")
        scaler = cached_scaler["scaler"]
    else:
        if cached_scaler is not None:
            print(f"\nResuming scaler cache from {cache_path(scaler_stage_key)}")
            scaler = cached_scaler["scaler"]
            processed_scaler_paths = set(cached_scaler.get("processed_paths", []))
        else:
            scaler = StandardScaler()
            processed_scaler_paths = set()
        print("\nStreaming train-benign traffic to fit StandardScaler ...")
        for file_spec in split_plan["train_benign"]:
            path_key = str(file_spec["path"])
            if path_key in processed_scaler_paths:
                continue
            print(f"  Scaling stats from {file_spec['path'].name} ...")
            for chunk in iter_csv_chunks(file_spec, canonical_feature_order):
                labels = chunk[file_spec["metadata_cols"]["label"]].map(canonicalize_label)
                benign_mask = labels.to_numpy() == "Benign"
                if not benign_mask.any():
                    continue
                X_raw = chunk_to_feature_matrix(chunk.loc[benign_mask], file_spec, canonical_feature_order)
                X_imp = imputer.transform(X_raw)[:, keep_mask]
                scaler.partial_fit(clip_to_bounds(X_imp, winsor_low, winsor_high))
            processed_scaler_paths.add(path_key)
            if ENABLE_PREPROCESS_CACHE:
                save_cache_payload(
                    scaler_stage_key,
                    {
                        "complete": len(processed_scaler_paths) == len(split_plan["train_benign"]),
                        "processed_paths": sorted(processed_scaler_paths),
                        "scaler": scaler,
                    },
                )
        if ENABLE_PREPROCESS_CACHE:
            print(f"\nSaving scaler cache to {cache_path(scaler_stage_key)}")
            save_cache_payload(
                scaler_stage_key,
                {
                    "complete": True,
                    "processed_paths": [str(spec["path"]) for spec in split_plan["train_benign"]],
                    "scaler": scaler,
                },
            )

    train_stage_key = stage_cache_key(cache_key, "train_windows")
    cached_train = load_cache_payload(train_stage_key) if ENABLE_PREPROCESS_CACHE else None
    if stage_cache_complete(cached_train):
        print(f"\nLoading train-window cache from {cache_path(train_stage_key)}")
        X_train_w = cached_train["X_train_w"]
        X_train_flows_sample = cached_train["X_train_flows"]
        train_seen = int(cached_train["train_seen"])
        restore_np_random_state(cached_train)
    else:
        if cached_train is not None:
            print(f"\nResuming train-window cache from {cache_path(train_stage_key)}")
            train_reservoir = ReservoirWindows.from_state(cached_train, "train_reservoir")
            kde_flows = ReservoirRows.from_state(cached_train, "kde_flows")
            train_seen = int(cached_train["train_seen"])
            processed_train_paths = set(cached_train.get("processed_paths", []))
            restore_np_random_state(cached_train)
        else:
            train_reservoir = ReservoirWindows(MAX_TRAIN_WINDOWS)
            kde_flows = ReservoirRows(MAX_KDE_TRAIN_FLOWS)
            train_seen = 0
            processed_train_paths = set()
        print("\nSampling train windows for optimization ...")
        for file_spec in split_plan["train_benign"]:
            path_key = str(file_spec["path"])
            if path_key in processed_train_paths:
                continue
            train_seen += sample_windows_from_files(
                [file_spec], canonical_feature_order, imputer, keep_mask,
                winsor_low, winsor_high, scaler, label_to_id, train_reservoir,
                stride=TRAIN_STRIDE, kde_flows=kde_flows,
            )
            processed_train_paths.add(path_key)
            if ENABLE_PREPROCESS_CACHE:
                save_cache_payload(
                    train_stage_key,
                    {
                        "complete": False,
                        "processed_paths": sorted(processed_train_paths),
                        "train_seen": train_seen,
                        "np_random_state": np.random.get_state(),
                        **train_reservoir.export_state("train_reservoir"),
                        **kde_flows.export_state("kde_flows"),
                    },
                )
        X_train_w, _, _ = train_reservoir.finalize()
        X_train_flows_sample = kde_flows.finalize().astype(np.float32, copy=False)
        if ENABLE_PREPROCESS_CACHE:
            print(f"\nSaving train-window cache to {cache_path(train_stage_key)}")
            save_cache_payload(
                train_stage_key,
                {
                    "complete": True,
                    "X_train_w": X_train_w,
                    "X_train_flows": X_train_flows_sample,
                    "train_seen": train_seen,
                    "np_random_state": np.random.get_state(),
                },
            )

    val_b_stage_key = stage_cache_key(cache_key, "val_benign_windows")
    cached_val_b = load_cache_payload(val_b_stage_key) if ENABLE_PREPROCESS_CACHE else None
    if stage_cache_complete(cached_val_b):
        print(f"\nLoading val-benign cache from {cache_path(val_b_stage_key)}")
        X_val_w = cached_val_b["X_val_w"]
        val_benign_seen = int(cached_val_b["val_benign_seen"])
        restore_np_random_state(cached_val_b)
    else:
        if cached_val_b is not None:
            print(f"\nResuming val-benign cache from {cache_path(val_b_stage_key)}")
            val_benign_reservoir = ReservoirWindows.from_state(cached_val_b, "val_benign_reservoir")
            val_benign_seen = int(cached_val_b["val_benign_seen"])
            processed_val_b_paths = set(cached_val_b.get("processed_paths", []))
            restore_np_random_state(cached_val_b)
        else:
            val_benign_reservoir = ReservoirWindows(MAX_VAL_BENIGN_WINDOWS)
            val_benign_seen = 0
            processed_val_b_paths = set()
        print("\nSampling val-benign windows for optimization ...")
        for file_spec in split_plan["val_benign"]:
            path_key = str(file_spec["path"])
            if path_key in processed_val_b_paths:
                continue
            val_benign_seen += sample_windows_from_files(
                [file_spec], canonical_feature_order, imputer, keep_mask,
                winsor_low, winsor_high, scaler, label_to_id, val_benign_reservoir,
                stride=TEST_STRIDE,
            )
            processed_val_b_paths.add(path_key)
            if ENABLE_PREPROCESS_CACHE:
                save_cache_payload(
                    val_b_stage_key,
                    {
                        "complete": False,
                        "processed_paths": sorted(processed_val_b_paths),
                        "val_benign_seen": val_benign_seen,
                        "np_random_state": np.random.get_state(),
                        **val_benign_reservoir.export_state("val_benign_reservoir"),
                    },
                )
        X_val_w, _, _ = val_benign_reservoir.finalize()
        if ENABLE_PREPROCESS_CACHE:
            print(f"\nSaving val-benign cache to {cache_path(val_b_stage_key)}")
            save_cache_payload(
                val_b_stage_key,
                {
                    "complete": True,
                    "X_val_w": X_val_w,
                    "val_benign_seen": val_benign_seen,
                    "np_random_state": np.random.get_state(),
                },
            )

    val_a_stage_key = stage_cache_key(cache_key, "val_attack_windows")
    cached_val_a = load_cache_payload(val_a_stage_key) if ENABLE_PREPROCESS_CACHE else None
    if stage_cache_complete(cached_val_a):
        print(f"\nLoading val-attack cache from {cache_path(val_a_stage_key)}")
        X_val_attack_w = cached_val_a["X_val_attack_w"]
        val_attack_seen = int(cached_val_a["val_attack_seen"])
        val_attack_sampled_family_counts = dict(cached_val_a.get("val_attack_sampled_family_counts", {}))
        restore_np_random_state(cached_val_a)
    else:
        if cached_val_a is not None:
            print(f"\nResuming val-attack cache from {cache_path(val_a_stage_key)}")
            val_attack_reservoir = FamilyAwareReservoirWindows.from_state(
                cached_val_a,
                "val_attack_family_reservoir",
            )
            val_attack_seen = int(cached_val_a["val_attack_seen"])
            processed_val_a_paths = set(cached_val_a.get("processed_paths", []))
            restore_np_random_state(cached_val_a)
        else:
            val_attack_reservoir = FamilyAwareReservoirWindows(val_attack_family_capacities)
            val_attack_seen = 0
            processed_val_a_paths = set()
        print("\nSampling val-attack windows for optimization ...")
        if val_attack_family_capacities:
            cap_desc = ", ".join(
                f"{label_name_by_id.get(fam, f'fam_{fam}')}={cap}"
                for fam, cap in sorted(val_attack_family_capacities.items())
            )
            print(f"    Val attack family caps: {cap_desc}")
        for file_spec in split_plan["val_attack"]:
            path_key = str(file_spec["path"])
            if path_key in processed_val_a_paths:
                continue
            val_attack_seen += sample_windows_from_files(
                [file_spec], canonical_feature_order, imputer, keep_mask,
                winsor_low, winsor_high, scaler, label_to_id, val_attack_reservoir,
                stride=TEST_STRIDE,
            )
            processed_val_a_paths.add(path_key)
            if ENABLE_PREPROCESS_CACHE:
                save_cache_payload(
                    val_a_stage_key,
                    {
                        "complete": False,
                        "processed_paths": sorted(processed_val_a_paths),
                        "val_attack_seen": val_attack_seen,
                        "np_random_state": np.random.get_state(),
                        **val_attack_reservoir.export_state("val_attack_family_reservoir"),
                    },
                )
        X_val_attack_w, _, y_val_attack_family = val_attack_reservoir.finalize()
        val_attack_sampled_family_counts = {
            label_name_by_id.get(int(fam), f"fam_{int(fam)}"): int((y_val_attack_family == fam).sum())
            for fam in sorted(np.unique(y_val_attack_family))
        }
        if ENABLE_PREPROCESS_CACHE:
            print(f"\nSaving val-attack cache to {cache_path(val_a_stage_key)}")
            save_cache_payload(
                val_a_stage_key,
                {
                    "complete": True,
                    "X_val_attack_w": X_val_attack_w,
                    "val_attack_seen": val_attack_seen,
                    "val_attack_sampled_family_counts": val_attack_sampled_family_counts,
                    "np_random_state": np.random.get_state(),
                },
            )
    if len(X_val_attack_w) > 0:
        X_val_mixed_w = np.concatenate([X_val_w, X_val_attack_w], axis=0)
        y_val_mixed_w = np.concatenate([
            np.zeros(len(X_val_w), dtype=np.int64),
            np.ones(len(X_val_attack_w), dtype=np.int64),
        ])
    else:
        X_val_mixed_w = X_val_w.copy()
        y_val_mixed_w = np.zeros(len(X_val_w), dtype=np.int64)

    LABEL_TO_ID = label_to_id
    CIC_LABEL_MAP = {idx: name for name, idx in LABEL_TO_ID.items()}

    print(f"\n  Sample rows from train benign: {sample_rows:,}")
    print(f"  Feature count after low-variance drop: {len(feature_names)}")
    print(f"  Entity grouping mode: {ENTITY_GROUP_MODE}")
    print("\n  Label distribution discovered:")
    for name, count in sorted(label_counts.items(), key=lambda kv: (LABEL_TO_ID[kv[0]], kv[0])):
        print(f"    {name:<20s} {count:>12,d}")
    print("\n  Optimization window counts:")
    print(f"    Train benign seen : {train_seen:,}")
    print(f"    Val benign seen   : {val_benign_seen:,}")
    print(f"    Val attack seen   : {val_attack_seen:,}")
    print("\n  Final sampled tensors for optimization:")
    print(f"    Train:     {X_train_w.shape}")
    print(f"    Val:       {X_val_w.shape}")
    print(f"    Val mixed: {X_val_mixed_w.shape} (attack rate={y_val_mixed_w.mean() if len(y_val_mixed_w) else 0.0:.4f})")
    print(f"    KDE flows: {X_train_flows_sample.shape}")
    print(f"    Val attack labels : {sorted(val_attack_labels) if val_attack_labels else 'none'}")
    print(f"    Test attack labels: {sorted(test_attack_labels) if test_attack_labels else 'none'}")
    if val_attack_sampled_family_counts:
        sampled_desc = ", ".join(
            f"{name}={count}"
            for name, count in sorted(val_attack_sampled_family_counts.items())
        )
        print(f"    Val attack sampled counts: {sampled_desc}")

    if USE_DELTA_FEATURES:
        feature_names_full = feature_names + [f"Δ_{name}" for name in feature_names]
    else:
        feature_names_full = feature_names

    preproc_ctx = {
        "canonical_feature_order": canonical_feature_order,
        "imputer": imputer,
        "keep_mask": keep_mask,
        "winsor_low": winsor_low,
        "winsor_high": winsor_high,
        "scaler": scaler,
        "label_to_id": label_to_id,
        "preproc_cache_key": cache_key,
    }
    val_stream_ctx = {
        "benign_specs": split_plan["val_benign"],
        "attack_specs": split_plan["val_attack"],
        "split_name": "val",
        "enable_full_window_cache": ENABLE_FULL_VAL_WINDOW_CACHE,
    }
    test_stream_ctx = {
        "benign_specs": split_plan["test_benign"],
        "attack_specs": split_plan["test_attack"],
        "split_name": "test",
        "enable_full_window_cache": False,
    }

    data_ctx = {
        "X_train_w": X_train_w,
        "X_val_w": X_val_w,
        "X_val_mixed_w": X_val_mixed_w,
        "y_val_mixed_w": y_val_mixed_w,
        "X_train_flows": X_train_flows_sample,
        "feature_names": feature_names,
        "feature_names_full": feature_names_full,
        "scaler": scaler,
        "preproc_ctx": preproc_ctx,
        "val_stream_ctx": val_stream_ctx,
        "test_stream_ctx": test_stream_ctx,
    }
    if ENABLE_PREPROCESS_CACHE:
        print(f"\nSaving preprocessing cache to {cache_path(cache_key)}")
        save_cache_payload(
            cache_key,
            {
                "X_train_w": data_ctx["X_train_w"],
                "X_val_w": data_ctx["X_val_w"],
                "X_val_mixed_w": data_ctx["X_val_mixed_w"],
                "y_val_mixed_w": data_ctx["y_val_mixed_w"],
                "X_train_flows": data_ctx["X_train_flows"],
                "feature_names": data_ctx["feature_names"],
                "feature_names_full": data_ctx["feature_names_full"],
                "preproc_ctx": data_ctx["preproc_ctx"],
                "val_stream_ctx": data_ctx["val_stream_ctx"],
                "test_stream_ctx": data_ctx["test_stream_ctx"],
                "label_to_id": LABEL_TO_ID,
                "cic_label_map": CIC_LABEL_MAP,
            },
        )
    return data_ctx


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

        # Decoder: position-aware heavier MLP for better reconstruction
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
    """Next-flow prediction loss + variance + covariance regularization.

    L = pred_err + λ_var · max(0, min_var - var(z)) + λ_cov · cov_loss(p)

    Variance term prevents embedding collapse (all dims → constant).
    Covariance term (VICReg-style) decorrelates projected representation p,
    where p = proj_head(z). Using a projection head decouples the
    decorrelation objective from the scoring representation z.

    When enabled, train-time SVDD adds a dimension-summed sphere loss exactly
    as specified by the reported training objective:
        svdd = sum((z - center.detach())**2, dim=1).mean()
    """

    def __init__(self, lambda_var=1.0, min_variance=0.5, lambda_cov=0.04,
                 feature_weights=None, train_svdd=False, lambda_svdd=1.0):
        super().__init__()
        self.lambda_var = lambda_var
        self.min_variance = min_variance
        self.lambda_cov = lambda_cov
        self.train_svdd = train_svdd
        self.lambda_svdd = lambda_svdd
        if feature_weights is not None:
            self.register_buffer("feature_weights", feature_weights)
        else:
            self.feature_weights = None

    def forward(self, x, x_hat, z, p, center=None):
        # Next-flow prediction: decoder[t] should predict input[t+1]
        x_target = x[:, 1:, :]       # (B, W-1, F)
        x_pred = x_hat[:, :-1, :]    # (B, W-1, F)
        if self.feature_weights is not None:
            pred_err = torch.mean(self.feature_weights * (x_pred - x_target) ** 2)
        else:
            pred_err = F.mse_loss(x_pred, x_target)

        # Variance regularization on z: prevent embedding collapse
        z_var = z.var(dim=0).mean()   # mean variance across bottleneck dims
        var_loss = F.relu(self.min_variance - z_var)

        # Covariance regularization on p (projected): decorrelate dimensions
        # Using p instead of z decouples decorrelation from scoring geometry
        p_centered = p - p.mean(dim=0)
        cov_matrix = (p_centered.T @ p_centered) / max(p.shape[0] - 1, 1)
        d = p.shape[1]
        off_diag_mask = ~torch.eye(d, device=p.device, dtype=torch.bool)
        cov_loss = cov_matrix[off_diag_mask].pow(2).sum() / d

        svdd_loss = torch.tensor(0.0, device=x.device)
        if self.train_svdd and center is not None:
            svdd_loss = torch.sum((z - center.detach())**2, dim=1).mean()

        total = (pred_err + self.lambda_var * var_loss
                 + self.lambda_cov * cov_loss
                 + self.lambda_svdd * svdd_loss)
        return total, pred_err, var_loss, cov_loss, svdd_loss


# ═══════════════════════════════════════════════════════════════════════════════
#  Training Utilities
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def init_center(model, dataloader, device, eps=0.1):
    """Initialize Deep SVDD center as mean of z over benign training windows."""
    model.eval()
    z_list = []
    for batch_x in dataloader:
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
    for bx in loader:
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
    for bx in loader:
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


@torch.no_grad()
def compute_resid_on_batch(model, bx, center):
    """Per-window per-feature prediction residual + euclid sphere.

    Returns:
        E:      (B, F) mean over the W-1 predicted positions of
                (x[t+1] - x_hat[t])² — the SAME residual tensor for both the
                raw and standardized readouts (readout is the only variable).
        sphere: (B,) Euclidean ‖z − c‖², the native SVDD readout.
    """
    x_hat, z, _ = model(bx)
    E = torch.mean((bx[:, 1:, :] - x_hat[:, :-1, :])**2, dim=1)   # (B, F)
    sphere = torch.sum((z - center)**2, dim=1)
    return E, sphere


@torch.no_grad()
def fit_pred_stats(model, X_val_benign, center, device, batch_size=256):
    """Benign per-feature residual variance from the PURE benign val reservoir
    (X_val_w) — the calibration statistics for the standardized readout.
    No labels, no attack data, no leakage."""
    model.eval()
    loader = DataLoader(ArrayWindowDataset(X_val_benign),
                        batch_size=batch_size, shuffle=False)
    E_parts = []
    for bx in loader:
        E, _ = compute_resid_on_batch(model, bx.to(device), center)
        E_parts.append(E.cpu().numpy())
    E_b = np.concatenate(E_parts)
    return {"var": E_b.var(0) + 1e-8, "mu": E_b.mean(0)}


def apply_pred_readouts(E: np.ndarray, pred_stats: Dict[str, np.ndarray]):
    """E (N,F) → (raw, standardized) scalar scores, as in the main pipeline."""
    raw = E.mean(1)
    std = (E / pred_stats["var"]).mean(1)
    return raw.astype(np.float32), std.astype(np.float32)




@torch.no_grad()
def compute_raw_signals_stream(
    model,
    split_ctx: Dict[str, Any],
    preproc_ctx: Dict[str, Any],
    center,
    device,
    pred_stats: Dict[str, np.ndarray],
    sample_last_flows: int = 0,
):
    """Streaming scorer: per-chunk residuals read as raw+std, euclid sphere."""
    model.eval()
    raw_parts: List[np.ndarray] = []
    std_parts: List[np.ndarray] = []
    sphere_parts: List[np.ndarray] = []
    attack_parts: List[np.ndarray] = []
    family_parts: List[np.ndarray] = []
    flow_reservoir = ReservoirLabeledRows(sample_last_flows)
    use_cached_val_windows = (
        split_ctx.get("split_name") == "val"
        and bool(split_ctx.get("enable_full_window_cache"))
        and sample_last_flows == 0
        and "preproc_cache_key" in preproc_ctx
    )

    def _score_batch(X_w):
        bx = torch.from_numpy(X_w).to(device)
        E, sphere = compute_resid_on_batch(model, bx, center)
        raw, std = apply_pred_readouts(E.cpu().numpy(), pred_stats)
        raw_parts.append(raw)
        std_parts.append(std)
        sphere_parts.append(sphere.cpu().numpy().astype(np.float32, copy=False))

    if use_cached_val_windows:
        for X_w, y_attack, y_family in iter_cached_full_val_window_batches(split_ctx, preproc_ctx):
            _score_batch(X_w)
            attack_parts.append(y_attack)
            family_parts.append(y_family)
    else:
        all_specs = list(split_ctx["benign_specs"]) + list(split_ctx["attack_specs"])
        for X_w, y_attack, y_family, last_flows in iter_window_batches_for_files(
            file_specs=all_specs,
            canonical_feature_order=preproc_ctx["canonical_feature_order"],
            imputer=preproc_ctx["imputer"],
            keep_mask=preproc_ctx["keep_mask"],
            winsor_low=preproc_ctx["winsor_low"],
            winsor_high=preproc_ctx["winsor_high"],
            scaler=preproc_ctx["scaler"],
            label_to_id=preproc_ctx["label_to_id"],
            stride=TEST_STRIDE,
            batch_windows=EVAL_WINDOW_BATCH_SIZE,
        ):
            _score_batch(X_w)
            attack_parts.append(y_attack)
            family_parts.append(y_family)
            if sample_last_flows > 0:
                for flow, fam in zip(last_flows, y_family):
                    flow_reservoir.add(flow, int(fam))

    z0 = np.zeros(0, dtype=np.float32)
    y_attack = np.concatenate(attack_parts) if attack_parts else np.zeros(0, dtype=np.int64)
    y_family = np.concatenate(family_parts) if family_parts else np.zeros(0, dtype=np.int64)
    sampled_flows, sampled_family = flow_reservoir.finalize()
    return {
        "pred_raw": np.concatenate(raw_parts) if raw_parts else z0,
        "pred_std": np.concatenate(std_parts) if std_parts else z0.copy(),
        "sphere": np.concatenate(sphere_parts) if sphere_parts else z0.copy(),
        "y_attack": y_attack,
        "y_family": y_family,
        "sampled_last_flows": sampled_flows,
        "sampled_last_flow_family": sampled_family,
    }




# ═══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(run_seed):
    global SEED, OUTPUT_DIR
    SEED = run_seed
    OUTPUT_DIR = output_dir_for_seed(SEED)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {architecture_tag()}")
    print(f"Output dir: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump({
            "seed": SEED,
            "data_seed": DATA_SEED,
            "architecture_tag": architecture_tag(),
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "bottleneck_dim": BOTTLENECK_DIM,
            "window_size": WINDOW_SIZE,
            "train_stride": TRAIN_STRIDE,
            "test_stride": TEST_STRIDE,
            "use_delta_features": USE_DELTA_FEATURES,
            "attack_test_policy": ATTACK_TEST_POLICY,
            "val_selection_mode": VAL_SELECTION_MODE,
            "max_val_benign_windows": MAX_VAL_BENIGN_WINDOWS,
            "max_val_attack_windows": MAX_VAL_ATTACK_WINDOWS,
            "min_val_attack_windows_per_family": MIN_VAL_ATTACK_WINDOWS_PER_FAMILY,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "output_tag": OUTPUT_TAG or ("retrain_splitfix" if not SKIP_TRAINING else ""),
            "skip_training": bool(SKIP_TRAINING),
        }, f, indent=2)

    # Data loading and preprocessing
    print("\n" + "="*80)
    print("  Data Loading & Windowing")
    print("="*80)
    data_ctx = load_and_preprocess()
    # load_and_preprocess intentionally uses DATA_SEED for stable data splits/cache.
    # Restore the run seed so model init, DataLoader shuffle, and training RNG vary per run.
    set_seed(SEED)
    X_train_w = data_ctx["X_train_w"]
    X_val_w = data_ctx["X_val_w"]
    X_val_mixed_w = data_ctx["X_val_mixed_w"]
    X_train_flows = data_ctx["X_train_flows"]
    feature_names = data_ctx["feature_names"]
    preproc_ctx = data_ctx["preproc_ctx"]
    val_stream_ctx = data_ctx["val_stream_ctx"]
    test_stream_ctx = data_ctx["test_stream_ctx"]

    if len(X_train_w) < max(2, BATCH_SIZE):
        raise RuntimeError(
            f"Too few training windows ({len(X_train_w)}) for BATCH_SIZE={BATCH_SIZE}. "
            "Increase MAX_TRAIN_WINDOWS or reduce BATCH_SIZE."
        )
    if len(X_val_w) == 0 or len(X_val_mixed_w) == 0:
        raise RuntimeError(
            "Validation window reservoirs are empty. "
            "Increase the val caps or adjust the fair file split."
        )

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
    train_loader = DataLoader(
        ArrayWindowDataset(X_train_w), **train_loader_kwargs)
    val_loader = DataLoader(
        ArrayWindowDataset(X_val_w), batch_size=BATCH_SIZE, shuffle=False)

    # Center initialization, objective construction, and training
    # Three tracked signals, each with its own val-AP-best epoch + checkpoint
    # (protocol: selection on val only; holdout families excluded from the
    # selection metric). "pred" = standardized readout (the method), "pred_raw"
    # = raw readout on the same residuals (the calibration-delta witness),
    # "sphere" = euclid ‖z−c‖² (the union's second detector).
    SIGNALS = {"pred": "pred_std", "pred_raw": "pred_raw", "sphere": "sphere"}
    ckpt_paths = {sig: os.path.join(OUTPUT_DIR, f"best_{sig}.pt")
                  for sig in SIGNALS}

    label_to_id = preproc_ctx["label_to_id"]
    id_to_label = {v: k for k, v in label_to_id.items()}
    holdout_ids = holdout_family_ids(label_to_id)
    print(f"  Holdout families (pre-registered): "
          f"{[(i, id_to_label.get(i, '?')) for i in holdout_ids]}")

    if SKIP_TRAINING:
        print("\n" + "="*80)
        print("  Loading per-signal checkpoints (SKIP_TRAINING=True)")
        print("="*80)
        sel = {}
        for sig in SIGNALS:
            sel[sig] = load_torch_checkpoint(ckpt_paths[sig], map_location="cpu")
            print(f"  {sig}: epoch {sel[sig]['epoch']}  "
                  f"val_AP={sel[sig]['val_ap']:.6f}")
        history = []
    else:
        loss_fn = PredictionLoss(lambda_var=LAMBDA_VAR, min_variance=MIN_VARIANCE,
                                 lambda_cov=LAMBDA_COV,
                                 feature_weights=kde_weights,
                                 train_svdd=TRAIN_SVDD,
                                 lambda_svdd=LAMBDA_SVDD).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                      weight_decay=WEIGHT_DECAY)
        warmup_epochs = 5
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1 + math.cos(
                math.pi * (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        print("\n" + "="*80)
        print("  Training S1 (prediction + KDE weighting + train-time SVDD)")
        print(f"  Masking: {MASK_RATIO:.0%}, λ_var={LAMBDA_VAR}, λ_cov={LAMBDA_COV}, "
              f"λ_svdd={LAMBDA_SVDD} (dim-summed), center every "
              f"{CENTER_RECOMPUTE_EVERY} ep")
        print(f"  Selection: per-signal val AP, epochs >= {MIN_SELECT_EPOCH}, "
              f"holdout families EXCLUDED from the selection metric")
        print("="*80)
        history = []
        center = init_center(model, train_loader, device)
        track = {sig: {"val_ap": -np.inf, "epoch": None} for sig in SIGNALS}
        traj = {"pred_raw": [], "pred_std": [], "sphere": []}
        traj_y = None
        patience_ctr = 0

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            if COV_WARMUP > 0 and epoch <= COV_WARMUP:
                loss_fn.lambda_cov = LAMBDA_COV * (epoch / COV_WARMUP)
            else:
                loss_fn.lambda_cov = LAMBDA_COV
            lc = LAMBDA_CONTRAST if epoch > CONTRAST_WARMUP else 0.0
            tm = train_one_epoch(model, train_loader, optimizer, loss_fn,
                                 device, mask_ratio=MASK_RATIO, center=center,
                                 lambda_contrast=lc,
                                 contrast_margin=CONTRAST_MARGIN)
            vm = validate(model, val_loader, loss_fn, device, center=center)
            scheduler.step()

            # Match the main pipeline's center schedule exactly.
            if epoch % CENTER_RECOMPUTE_EVERY == 0:
                center = init_center(model, train_loader, device)

            # Per-epoch selection signals on the FULL val stream. Benign
            # calibration stats come from the PURE benign reservoir (X_val_w).
            pred_stats = fit_pred_stats(model, X_val_w, center, device)
            vs = compute_raw_signals_stream(
                model, val_stream_ctx, preproc_ctx, center, device,
                pred_stats, sample_last_flows=0)
            y_va, fam_va = vs["y_attack"], vs["y_family"]
            if traj_y is None:
                traj_y = (y_va, fam_va)
            sel_mask = ~np.isin(fam_va, holdout_ids)
            aps = {sig: float(average_precision_score(
                y_va[sel_mask], vs[key][sel_mask]))
                for sig, key in SIGNALS.items()}
            for key in traj:
                traj[key].append(vs[key])

            sel_ok = epoch >= MIN_SELECT_EPOCH
            improved_pred = False
            for sig, key in SIGNALS.items():
                if sel_ok and aps[sig] > track[sig]["val_ap"]:
                    track[sig].update(val_ap=aps[sig], epoch=epoch)
                    if sig == "pred":
                        improved_pred = True
                    torch.save({
                        "model": {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()},
                        "center": center.detach().cpu(),
                        "epoch": epoch, "val_ap": aps[sig],
                        "val_pred_raw": vs["pred_raw"],
                        "val_pred_std": vs["pred_std"],
                        "val_sphere": vs["sphere"],
                        "y_val_attack": y_va, "y_val_family": fam_va,
                    }, ckpt_paths[sig])

            elapsed = time.time() - t0
            history.append({"epoch": epoch,
                            **{f"train_{k}": v for k, v in tm.items()},
                            **{f"val_{k}": v for k, v in vm.items()},
                            **{f"val_ap_{sig}": aps[sig] for sig in SIGNALS},
                            "lr": optimizer.param_groups[0]["lr"],
                            "time_s": elapsed})
            print(f"  Epoch {epoch:3d}/{EPOCHS}  train={tm['total']:.6f}  "
                  f"svdd={tm['svdd']:.4f}  val_AP std {aps['pred']:.4f} "
                  f"raw {aps['pred_raw']:.4f} sphere {aps['sphere']:.4f}  "
                  f"({elapsed:.1f}s)")

            if sel_ok:
                if improved_pred:
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= FULL_VAL_PATIENCE:
                        print(f"\n  Early stopping at epoch {epoch} "
                              f"(best pred val_AP={track['pred']['val_ap']:.6f})")
                        break

        if track["pred"]["epoch"] is None:
            raise RuntimeError(
                f"No epoch reached MIN_SELECT_EPOCH={MIN_SELECT_EPOCH}; "
                "increase EPOCHS or lower the warmup.")

        traj_path = os.path.join(OUTPUT_DIR, "val_signal_trajectory.npz")
        np.savez_compressed(
            traj_path,
            epochs=np.arange(1, len(traj["pred_std"]) + 1, dtype=np.int32),
            val_pred_raw=np.stack(traj["pred_raw"]),
            val_pred=np.stack(traj["pred_std"]),
            val_sphere_eu=np.stack(traj["sphere"]),
            y_val=traj_y[0], y_val_family=traj_y[1],
            holdout_ids=np.asarray(holdout_ids, dtype=np.int64))
        print(f"  Dumped val signal trajectory -> {traj_path}")
        sel = {sig: load_torch_checkpoint(ckpt_paths[sig], map_location="cpu")
               for sig in SIGNALS}

    # Final evaluation
    print("\n" + "="*80)
    print("  Final evaluation — reported protocol")
    print("="*80)
    ei = {sig: int(sel[sig]["epoch"]) for sig in SIGNALS}
    print("  Selected epochs: " + "  ".join(
        f"{sig}@{ei[sig]} (val_AP={sel[sig]['val_ap']:.4f})" for sig in SIGNALS))

    def _restore(sig):
        state = {k: torch.as_tensor(v).to(device)
                 for k, v in sel[sig]["model"].items()}
        model.load_state_dict(state)
        return torch.as_tensor(sel[sig]["center"]).to(device)

    # One test streaming pass per DISTINCT selected epoch.
    ts_by_epoch = {}
    for sig in SIGNALS:
        e = ei[sig]
        if e not in ts_by_epoch:
            center_e = _restore(sig)
            stats_e = fit_pred_stats(model, X_val_w, center_e, device)
            print(f"  Streaming test at epoch {e} ...")
            ts_by_epoch[e] = compute_raw_signals_stream(
                model, test_stream_ctx, preproc_ctx, center_e, device,
                stats_e, sample_last_flows=0)
    ts = {sig: ts_by_epoch[ei[sig]] for sig in SIGNALS}

    y_te = ts["pred"]["y_attack"].astype(int)
    fam_te = ts["pred"]["y_family"].astype(int)
    atk_te = y_te == 1
    hold_te = np.isin(fam_te, holdout_ids)
    print(f"  Test windows: {len(y_te)} (attack rate={atk_te.mean():.4f}, "
          f"holdout windows={int(hold_te.sum())})")

    def protocol_row(name, s_test, valb_scores):
        row = {"name": name}
        row["auprc_test"] = float(average_precision_score(y_te, s_test))
        row["auroc"] = float(roc_auc_score(y_te, s_test))
        hmask = (fam_te == 0) | hold_te
        row["auprc_holdout"] = (
            float(average_precision_score((fam_te[hmask] != 0).astype(int),
                                          s_test[hmask]))
            if hold_te.any() else float("nan"))
        thr = float(np.quantile(valb_scores, 1.0 - TPR_BUDGET))
        flag = s_test >= thr
        row["threshold"] = thr
        row["tpr_at_budget"] = float(flag[atk_te].mean())
        row["fpr_actual"] = float(flag[~atk_te].mean())
        row["tpr_holdout"] = (float(flag[atk_te & hold_te].mean())
                              if (atk_te & hold_te).any() else float("nan"))
        row["per_family_tpr"] = {
            id_to_label.get(int(f), str(int(f))): float(flag[fam_te == f].mean())
            for f in np.unique(fam_te) if int(f) != 0}
        return row

    def _valb(sig, key):
        benign = sel[sig]["y_val_attack"] == 0
        return sel[sig][key][benign]

    rows = {}
    rows["pred_std"] = protocol_row("pred[std]", ts["pred"]["pred_std"],
                                    _valb("pred", "val_pred_std"))
    rows["pred_raw"] = protocol_row("pred[raw]", ts["pred_raw"]["pred_raw"],
                                    _valb("pred_raw", "val_pred_raw"))
    rows["sphere"] = protocol_row("sphere[euclid]", ts["sphere"]["sphere"],
                                  _valb("sphere", "val_sphere"))

    print(f"\n  {'signal':<16}{'sel_ep':>7}{'test_AP':>9}{'AUROC':>8}"
          f"{'ho_AP':>8}{'TPR@1%':>8}{'FPR_act':>9}{'hoTPR':>7}")
    print("  " + "-" * 72)
    for key, sig in (("pred_std", "pred"), ("pred_raw", "pred_raw"),
                     ("sphere", "sphere")):
        r = rows[key]
        print(f"  {r['name']:<16}{ei[sig]:>7}{r['auprc_test']:>9.4f}"
              f"{r['auroc']:>8.4f}{r['auprc_holdout']:>8.4f}"
              f"{r['tpr_at_budget']:>8.4f}{r['fpr_actual']:>9.4f}"
              f"{r['tpr_holdout']:>7.4f}")
    d_cal = rows["pred_std"]["auprc_test"] - rows["pred_raw"]["auprc_test"]
    d_cal_ho = rows["pred_std"]["auprc_holdout"] - rows["pred_raw"]["auprc_holdout"]
    print(f"\n  CALIBRATION DELTA (std−raw, this seed): test {d_cal:+.4f}  "
          f"holdout {d_cal_ho:+.4f}")

    # Union at the reported 50/50 budget split, with each signal at its own epoch
    # thresholds from its own benign-val quantiles. Baseline = pred@full budget.
    bp, bs = UNION_SPLIT
    thr_pu = float(np.quantile(_valb("pred", "val_pred_std"),
                               1.0 - TPR_BUDGET * bp))
    thr_su = float(np.quantile(_valb("sphere", "val_sphere"),
                               1.0 - TPR_BUDGET * bs))
    flag_p = ts["pred"]["pred_std"] >= rows["pred_std"]["threshold"]
    flag_u = ((ts["pred"]["pred_std"] >= thr_pu)
              | (ts["sphere"]["sphere"] >= thr_su))
    union = {}
    for nm, flag in (("pred@1%", flag_p), ("union 50/50", flag_u)):
        union[nm] = {
            "tpr": float(flag[atk_te].mean()),
            "fpr": float(flag[~atk_te].mean()),
            "ho_tpr": (float(flag[atk_te & hold_te].mean())
                       if (atk_te & hold_te).any() else float("nan")),
        }
        u = union[nm]
        print(f"  {nm:<12} TPR {u['tpr']:.4f}  FPR {u['fpr']:.4f}  "
              f"hoTPR {u['ho_tpr']:.4f}")
    union["dTPR"] = union["union 50/50"]["tpr"] - union["pred@1%"]["tpr"]
    union["dFPR"] = union["union 50/50"]["fpr"] - union["pred@1%"]["fpr"]
    print(f"  union dTPR {union['dTPR']:+.4f}  dFPR {union['dFPR']:+.4f}")

    print("\n  Per-family TPR@1% (pred[std]):")
    for fam, tpr in sorted(rows["pred_std"]["per_family_tpr"].items()):
        fam_id = label_to_id.get(fam)
        tag = " [HOLDOUT]" if fam_id in holdout_ids else ""
        print(f"    {fam:<28s} {tpr:.4f}{tag}")

    result = {
        "seed": SEED,
        "epochs_selected": ei,
        "rows": rows,
        "union": union,
        "holdout_families": {int(i): id_to_label.get(int(i), "?")
                             for i in holdout_ids},
        "n_test_windows": int(len(y_te)),
        "test_attack_rate": float(atk_te.mean()),
    }
    # Dump the selected-epoch TEST score arrays (plus benign-val scores for
    # thresholding) so offline analyses (union budget sweep, rescue rates) can
    # run without re-streaming test. ~a few MB per seed.
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "test_scores.npz"),
        pred_std=ts["pred"]["pred_std"].astype(np.float32),
        pred_raw=ts["pred_raw"]["pred_raw"].astype(np.float32),
        sphere=ts["sphere"]["sphere"].astype(np.float32),
        y_attack=y_te.astype(np.int8), y_family=fam_te.astype(np.int16),
        valb_pred_std=_valb("pred", "val_pred_std").astype(np.float32),
        valb_pred_raw=_valb("pred_raw", "val_pred_raw").astype(np.float32),
        valb_sphere=_valb("sphere", "val_sphere").astype(np.float32),
        epochs_selected=np.array([ei["pred"], ei["pred_raw"], ei["sphere"]],
                                 dtype=np.int32),
        holdout_ids=np.array(sorted(holdout_ids), dtype=np.int16),
    )
    print(f"  Dumped selected-epoch test scores -> "
          f"{os.path.join(OUTPUT_DIR, 'test_scores.npz')}")
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(deep_convert(result), f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(deep_convert(history), f, indent=2)
    print(f"\n✓ Results saved to {OUTPUT_DIR}/")
    return result


def deep_convert(obj):
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


def run_label_census():
    """Scan the per-family label distribution across all dataset files and
    exit. Record the holdout choice before training, then fill
    HOLDOUT_FAMILY_NAMES and start the runs."""
    print("="*80)
    print("  LABEL CENSUS (no training) — for holdout pre-registration")
    print("="*80)
    csv_files = discover_dataset_files()
    specs, _, _ = resolve_dataset_specs(csv_files)
    _, label_counts, _, day_attack_rows = \
        scan_label_distribution(specs)
    total = sum(label_counts.values())
    print(f"\n  Files: {len(specs)}   Total rows: {total:,}")
    print(f"\n  {'label':<40s}{'rows':>12s}{'share':>9s}")
    print("  " + "-" * 61)
    for label, count in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<40s}{count:>12,}{count / total:>9.4%}")
    print("\n  Attack rows per day-key:")
    for day, count in sorted(day_attack_rows.items()):
        print(f"    {day}: {count:,}")
    print("\n  NEXT: record the holdout choice, fill HOLDOUT_FAMILY_NAMES, "
          "set RUN_LABEL_CENSUS=False, and run.")


def main():
    if RUN_LABEL_CENSUS:
        run_label_census()
        return
    if not HOLDOUT_FAMILY_NAMES:
        raise RuntimeError(
            "HOLDOUT_FAMILY_NAMES is empty. The evaluation protocol requires the label "
            "census (RUN_LABEL_CENSUS=True) and a recorded holdout choice "
            "before any experiment run.")

    all_results = []
    for s in RUN_SEEDS:
        print(f"\n\n{'='*80}\n  STARTING S1 BCCC-CSE-CIC-IDS2018 RUN WITH SEED {s}\n{'='*80}")
        all_results.append(run_experiment(s))

    print(f"\n\n{'='*80}\n  AGGREGATED OVER {len(RUN_SEEDS)} SEEDS "
          f"(reported protocol)\n{'='*80}")
    for key in ("pred_std", "pred_raw", "sphere"):
        for metric in ("auprc_test", "auprc_holdout", "tpr_at_budget"):
            vals = np.array([r["rows"][key][metric] for r in all_results],
                            dtype=float)
            print(f"  {key:<10s} {metric:<14s}: {vals.mean():.4f} +/- "
                  f"{vals.std():.4f}   per-seed {np.round(vals, 4).tolist()}")
    print("\n  PAIRED CALIBRATION DELTA (std−raw):")
    for metric, lbl in (("auprc_test", "test"), ("auprc_holdout", "ho")):
        d = np.array([r["rows"]["pred_std"][metric]
                      - r["rows"]["pred_raw"][metric]
                      for r in all_results], dtype=float)
        print(f"    [{lbl:<4}] mean {d.mean():+.4f}  positive on "
              f"{(d > 0).sum()}/{len(d)} seeds  per-seed "
              f"{np.round(d, 4).tolist()}")
    d_tpr = np.array([r["union"]["dTPR"] for r in all_results], dtype=float)
    d_fpr = np.array([r["union"]["dFPR"] for r in all_results], dtype=float)
    print(f"\n  UNION 50/50 vs pred@1%: mean dTPR {d_tpr.mean():+.4f} "
          f"(nonneg {(d_tpr >= 0).sum()}/{len(d_tpr)})  "
          f"mean dFPR {d_fpr.mean():+.4f}")

    out_root = experiment_root_dir()
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, "aggregated_results.json")
    with open(out_path, "w") as f:
        json.dump(deep_convert(all_results), f, indent=2)
    print(f"\nSaved aggregated results to {out_path}")


if __name__ == "__main__":
    main()
