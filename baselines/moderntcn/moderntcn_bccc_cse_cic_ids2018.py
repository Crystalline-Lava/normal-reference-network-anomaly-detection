# Adapted in part from luodhhh/ModernTCN, MIT License.
# Source and full upstream notice: ../../THIRD_PARTY_NOTICES.md

"""ModernTCN baseline for the BCCC-CSE-CIC-IDS2018 entity-window task.

The script shares the reported data split, window construction, and fixed
``DATA_SEED`` with the other BCCC-CSE-CIC-IDS2018 experiments. Delta features
are disabled. The model uses the official MSL-style ModernTCN configuration
and trains for eight epochs, while retaining the official two-epoch budget
result.

Results include native validation-loss selection, official-budget selection,
and validation-AP selection for each readout. The split fingerprint is checked
on the first run to prevent data drift.
"""
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
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
try:
    from google.colab import drive
except ImportError:
    drive = None


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
HOLDOUT_FAMILY_NAMES: List[str] = [
    "Brute_Force_Web",
    "Brute_Force_XSS",
    "SQL_Injection",
]
TPR_BUDGET = 0.01              # TPR @ 1% benign-val budget
SINGLETON_ATTACK_VAL_FRACTION = 0.4

# ModernTCN-specific configuration

from types import SimpleNamespace  # noqa: E402  (make_moderntcn_config)
from torch.optim import lr_scheduler  # noqa: E402  (OneCycleLR, per official recipe)
from torch.utils.data import TensorDataset  # noqa: E402  (feature_residuals)

USE_DELTA_FEATURES = False       # baselines get raw features (delta OFF)
EPOCHS = 8
OFFICIAL_BUDGET_EPOCHS = 2       # official ModernTCN AD budget row (@off2ep)
BATCH_SIZE = 128
LR = 5e-4                        # Adam + OneCycleLR (official recipe)
PATIENCE = 10                    # official value; never fires within 8 epochs
PCT_START = 0.3
MIN_SELECT_EPOCH = 1
READOUTS = ["raw", "std", "maha"]  # raw = E.mean(1) IS the native window score

# Official MSL-style ModernTCN configuration. MSL is the closest upstream
# anomaly-detection configuration in dimensionality. The native percentile
# exhibit is implemented only by the CIC-UNSW-NB15 native-stream entry point.
PATCH_SIZE = 8
PATCH_STRIDE = 4
DOWNSAMPLE_RATIO = 2
FFN_RATIO = 1
NUM_BLOCKS = [1]
LARGE_SIZE = [51]
SMALL_SIZE = [5]
DIMS = [8]
DROPOUT = 0.1                    # ModernTCN backbone dropout (official)
HEAD_DROPOUT = 0.0
USE_MULTI_SCALE = False
SMALL_KERNEL_MERGED = False
REVIN = True
AFFINE = True
SUBTRACT_LAST = False
INDIVIDUAL = False

EXPERIMENT_GROUP = "moderntcn_bccc_cse_cic_ids2018"

# Filled per run from the dynamic label encoding (holdout_family_ids);
# read by operating_point().
HOLDOUT_FAMILIES: List[int] = []


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
#  ModernTCN model stack + readout helpers — model classes + config + run_epoch
#  generic helpers follow the proposed BCCC-CSE-CIC-IDS2018 data protocol where the
#  signatures run_experiment needs are identical
# ═══════════════════════════════════════════════════════════════════════════════

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
    (x − recon)². Note E.mean(axis=1) IS the native ModernTCN score
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


# ═══════════════════════════════════════════════════════════════════════════════
#  ModernTCN baseline glue — output roots, streaming scorers, selection loop
# ═══════════════════════════════════════════════════════════════════════════════

# Flat output naming inside the baseline group (the transformer arch tag from
# the proposed pipeline is not used for ModernTCN):
# .../moderntcn_bccc_cse_cic_ids2018/seed_<s>/
def architecture_tag() -> str:  # override
    return (f"mtcn_p{PATCH_SIZE}s{PATCH_STRIDE}_lk{LARGE_SIZE[0]}"
            f"_nb{NUM_BLOCKS[0]}_dim{DIMS[0]}"
            f"_w{WINDOW_SIZE}_d{int(USE_DELTA_FEATURES)}")


def experiment_root_dir() -> str:  # override: flat, no arch subdir
    return os.path.join(BASE_DIR, EXPERIMENT_GROUP)


def output_dir_for_seed(seed: int) -> str:
    return os.path.join(experiment_root_dir(), f"seed_{seed}")


class TupleWindowDataset(ArrayWindowDataset):
    """ArrayWindowDataset wrapped to yield 1-tuples so the ModernTCN baseline's
    run_epoch/validate loops (`for (batch_x,) in loader`) work on the
    fp16 window reservoirs without a full float32 copy up front."""

    def __getitem__(self, idx: int):
        return (super().__getitem__(idx),)


EVAL_MODEL_BATCH = 512  # windows per forward pass when scoring stream batches


def _iter_split_window_batches(split_ctx: Dict[str, Any],
                               preproc_ctx: Dict[str, Any]):
    """(X_w, y_attack, y_family) batches for a split — cached shards for the
    full-val stream (the same on-disk cache used by this baseline), direct CSV
    streaming for test. Mirrors the iteration skeleton of the main method's
    streaming scorer."""
    use_cached = (
        split_ctx.get("split_name") == "val"
        and bool(split_ctx.get("enable_full_window_cache"))
        and "preproc_cache_key" in preproc_ctx
    )
    if use_cached:
        yield from iter_cached_full_val_window_batches(split_ctx, preproc_ctx)
        return
    all_specs = list(split_ctx["benign_specs"]) + list(split_ctx["attack_specs"])
    for X_w, y_attack, y_family, _last_flows in iter_window_batches_for_files(
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
        yield X_w, y_attack, y_family


@torch.no_grad()
def _score_split_stream(model: nn.Module, split_ctx: Dict[str, Any],
                        preproc_ctx: Dict[str, Any], device):
    """Stream a split through ModernTCN; return (E, y_attack, y_family) with E
    the per-window per-feature residual (N, F) = ((recon − x)²).mean(dim=1) —
    exactly the tensor feature_residuals() produces for in-memory arrays, so
    E.mean(1) is the native ModernTCN score."""
    model.eval()
    E_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    fam_parts: List[np.ndarray] = []
    for X_w, y_attack, y_family in _iter_split_window_batches(split_ctx, preproc_ctx):
        X_w = np.asarray(X_w, dtype=np.float32)
        for i in range(0, len(X_w), EVAL_MODEL_BATCH):
            bx = torch.from_numpy(
                np.ascontiguousarray(X_w[i:i + EVAL_MODEL_BATCH])).to(device)
            recon = model(bx)
            E_parts.append(((recon - bx) ** 2).mean(dim=1)
                           .cpu().numpy().astype(np.float32, copy=False))
        y_parts.append(np.asarray(y_attack, dtype=np.int64))
        fam_parts.append(np.asarray(y_family, dtype=np.int64))
    E = np.concatenate(E_parts) if E_parts else np.zeros((0, 1), dtype=np.float32)
    y = np.concatenate(y_parts) if y_parts else np.zeros(0, dtype=np.int64)
    fam = np.concatenate(fam_parts) if fam_parts else np.zeros(0, dtype=np.int64)
    return E, y, fam


def score_val_stream(model: nn.Module, data_ctx: Dict[str, Any],
                     preproc_ctx: Dict[str, Any], device):
    """Residuals + labels over the FULL val window stream (cached shards)."""
    return _score_split_stream(model, data_ctx["val_stream_ctx"], preproc_ctx,
                               device)


def score_test_stream(model: nn.Module, test_stream_ctx: Dict[str, Any],
                      preproc_ctx: Dict[str, Any], device):
    """Residuals + labels over the TEST stream (direct CSV streaming)."""
    return _score_split_stream(model, test_stream_ctx, preproc_ctx, device)




def run_experiment(run_seed):
    global SEED, OUTPUT_DIR, HOLDOUT_FAMILIES
    SEED = run_seed
    OUTPUT_DIR = output_dir_for_seed(run_seed)
    set_seed(run_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {architecture_tag()}")
    print(f"Output dir: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump({
            "seed": run_seed,
            "data_seed": DATA_SEED,
            "model": "ModernTCN",
            "architecture_tag": architecture_tag(),
            "patch_size": PATCH_SIZE, "patch_stride": PATCH_STRIDE,
            "num_blocks": NUM_BLOCKS, "large_size": LARGE_SIZE,
            "small_size": SMALL_SIZE, "dims": DIMS,
            "ffn_ratio": FFN_RATIO, "dropout": DROPOUT,
            "head_dropout": HEAD_DROPOUT, "pct_start": PCT_START,
            "window_size": WINDOW_SIZE,
            "train_stride": TRAIN_STRIDE,
            "test_stride": TEST_STRIDE,
            "use_delta_features": USE_DELTA_FEATURES,
            "attack_test_policy": ATTACK_TEST_POLICY,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "official_budget_epochs": OFFICIAL_BUDGET_EPOCHS,
            "lr": LR,
            "patience": PATIENCE,
            "min_select_epoch": MIN_SELECT_EPOCH,
            "readouts": READOUTS,
        }, f, indent=2)

    # Shared data layer (delta features disabled for this baseline)
    print("\n" + "=" * 80)
    print("  Shared Data Loading & Windowing (delta features OFF)")
    print("=" * 80)
    data_ctx = load_and_preprocess()
    # load_and_preprocess uses DATA_SEED for stable splits/cache; restore the
    # run seed so model init and DataLoader shuffle vary per run.
    set_seed(run_seed)
    X_train_w = data_ctx["X_train_w"]
    X_val_w = data_ctx["X_val_w"]
    feature_names_full = data_ctx["feature_names_full"]
    preproc_ctx = data_ctx["preproc_ctx"]
    test_stream_ctx = data_ctx["test_stream_ctx"]
    if len(X_train_w) < max(2, BATCH_SIZE):
        raise RuntimeError(
            f"Too few training windows ({len(X_train_w)}) for "
            f"BATCH_SIZE={BATCH_SIZE}.")
    if len(X_val_w) == 0:
        raise RuntimeError("Benign val window reservoir is empty.")

    # Split-identity assertions available right after the (cached) split plan.
    label_to_id = preproc_ctx["label_to_id"]
    id_to_label = {v: k for k, v in label_to_id.items()}
    holdout_ids = holdout_family_ids(label_to_id)
    if len(holdout_ids) != 3:
        raise RuntimeError(
            f"SPLIT DRIFT vs proposed_bccc_cse_cic_ids2018.py: expected exactly 3 holdout families "
            f"(web-attack cluster), got ids {holdout_ids}")
    HOLDOUT_FAMILIES = [int(i) for i in holdout_ids]
    print(f"  Holdout families (pre-registered): "
          f"{[(i, id_to_label.get(i, '?')) for i in holdout_ids]}")

    n_features = int(X_train_w.shape[2])
    if n_features != len(feature_names_full):
        raise RuntimeError(
            f"Feature-count mismatch: windows have {n_features}, loader "
            f"reports {len(feature_names_full)} (delta override not applied?)")

    # ModernTCN construction (official MSL recipe)
    print("\n" + "=" * 80)
    print("  ModernTCN Baseline Construction (official MSL config)")
    print("=" * 80)
    configs = make_moderntcn_config(n_features)
    model = Model(configs).float().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Window: {WINDOW_SIZE} flows x {n_features} features "
          f"(delta features OFF)")

    train_loader = DataLoader(TupleWindowDataset(X_train_w),
                              batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(TupleWindowDataset(X_val_w),
                            batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        steps_per_epoch=max(len(train_loader), 1),
        pct_start=PCT_START,
        epochs=EPOCHS,
        max_lr=LR,
    )

    # Training and checkpoint selection (native, official-budget, or validation AP)
    print("\n" + "=" * 80)
    print("  Training (MSE reconstruction) + per-epoch readout tracking")
    print(f"  Selection: native = lowest benign-val recon loss (PRIMARY); "
          f"@off{OFFICIAL_BUDGET_EPOCHS}ep = same within official budget; "
          f"valAP per readout, epochs >= {MIN_SELECT_EPOCH}, holdout EXCLUDED")
    print("=" * 80)
    best_val_loss = float("inf")
    best_valap = {r: -np.inf for r in READOUTS}
    captures: Dict[str, Dict[str, Any]] = {}
    history: List[Dict[str, Any]] = []
    patience_ctr = 0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, criterion,
                               scheduler, device)
        lr = optimizer.param_groups[0]["lr"]  # OneCycleLR steps per batch
        val_loss = validate(model, val_loader, criterion, device)

        # Benign calibration stats from the PURE benign reservoir (X_val_w).
        E_b = feature_residuals(model, X_val_w, device)
        stats = fit_benign_stats(E_b)
        valb = {r: apply_readout(E_b, stats, r).astype(np.float32)
                for r in READOUTS}
        # Selection metric on the FULL val stream, holdout families EXCLUDED.
        E_va, y_va, fam_va = score_val_stream(model, data_ctx, preproc_ctx,
                                              device)
        sel_mask = ~np.isin(fam_va, holdout_ids)
        vaps = {}
        for r in READOUTS:
            s_va = apply_readout(E_va, stats, r)
            vaps[r] = float(average_precision_score(y_va[sel_mask],
                                                    s_va[sel_mask]))

        cap_cache: Dict[str, Any] = {}

        def _capture():
            # One shared capture per epoch (cpu-copied weights + stats +
            # benign-val scores + val APs at this epoch).
            if "cap" not in cap_cache:
                cap_cache["cap"] = {
                    "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()},
                    "stats": {k: np.array(v, copy=True)
                              for k, v in stats.items()},
                    "valb": {r: valb[r].copy() for r in READOUTS},
                    "val_ap": dict(vaps),
                }
            return cap_cache["cap"]

        loss_improved = val_loss < best_val_loss
        if loss_improved:
            best_val_loss = val_loss
            patience_ctr = 0
            captures["nat"] = _capture()      # NATIVE: ModernTCN's own rule
            if epoch <= OFFICIAL_BUDGET_EPOCHS:
                captures["off"] = _capture()  # official 3-epoch budget
        else:
            patience_ctr += 1
        for r in READOUTS:
            if epoch >= MIN_SELECT_EPOCH and vaps[r] > best_valap[r]:
                best_valap[r] = vaps[r]
                captures[f"valap_{r}"] = _capture()

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "lr": lr, "time_s": elapsed,
                        **{f"val_ap_{r}": vaps[r] for r in READOUTS}})
        print(f"  Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.6f}  "
              f"val={val_loss:.6f}  val_AP raw {vaps['raw']:.4f} "
              f"std {vaps['std']:.4f} maha {vaps['maha']:.4f}  "
              f"lr={lr:.2e}  ({elapsed:.1f}s)")

        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} "
                  f"(best val loss={best_val_loss:.6f})")
            break

    # Final evaluation: one test streaming pass per distinct selected epoch
    print("\n" + "=" * 80)
    print("  Final evaluation — reported protocol")
    print("=" * 80)
    print("  Selected epochs: " + "  ".join(
        f"{k}@{c['epoch']}" for k, c in sorted(captures.items())))
    epochs_needed = sorted({c["epoch"] for c in captures.values()})
    print(f"  Distinct checkpoint epochs to stream test at: {epochs_needed}")

    y_te = None
    fam_te = None
    scores_test: Dict[str, Dict[str, np.ndarray]] = {}
    for e in epochs_needed:
        cap_e = next(c for c in captures.values() if c["epoch"] == e)
        model.load_state_dict({k: v.to(device)
                               for k, v in cap_e["state"].items()})
        print(f"  Streaming test at epoch {e} ...")
        E_te, y_e, fam_e = score_test_stream(model, test_stream_ctx,
                                             preproc_ctx, device)
        if y_te is None:
            y_te = y_e.astype(int)
            fam_te = fam_e.astype(int)
        for key, c in captures.items():
            if c["epoch"] == e:
                scores_test[key] = {
                    r: apply_readout(E_te, c["stats"], r).astype(np.float32)
                    for r in READOUTS}
        del E_te

    atk_te = y_te == 1
    hold_te = np.isin(fam_te, holdout_ids)
    print(f"  Test windows: {len(y_te)} (attack rate={atk_te.mean():.4f}, "
          f"holdout windows={int(hold_te.sum())})")

    # Confirm that this baseline uses the same reported test split.
    if run_seed == RUN_SEEDS[0]:
        n_hold = int(hold_te.sum())
        rate = float(atk_te.mean())
        if (len(y_te) != 326370 or n_hold != 28
                or abs(rate - 0.4802) >= 0.001):
            raise RuntimeError(
                f"SPLIT DRIFT vs proposed_bccc_cse_cic_ids2018.py: got n_test={len(y_te)}, "
                f"holdout_windows={n_hold}, attack_rate={rate:.4f}; expected "
                f"326370 / 28 / 0.4802±0.001. Baseline is NOT comparable — "
                f"fix the data layer before trusting any number.")
        print("  Split fingerprint OK: 326370 test windows / 28 holdout "
              "windows / attack rate 0.4802")

    def protocol_row(name, s_test, valb_scores):
        row = {"name": name}
        row["auprc_test"] = float(average_precision_score(y_te, s_test))
        row["auroc"] = float(roc_auc_score(y_te, s_test))
        hmask = (fam_te == 0) | hold_te
        row["auprc_holdout"] = (
            float(average_precision_score((fam_te[hmask] != 0).astype(int),
                                          s_test[hmask]))
            if hold_te.any() else float("nan"))
        op = operating_point(valb_scores, s_test, y_te, fam_te)
        thr = float(np.quantile(valb_scores, 1.0 - TPR_BUDGET))
        flag = s_test >= thr
        row["threshold"] = thr
        row["tpr_at_budget"] = op["tpr"]
        row["fpr_actual"] = op["fpr_actual"]
        row["tpr_holdout"] = op["tpr_holdout"]
        row["per_family_tpr"] = {
            id_to_label.get(int(fm), str(int(fm))):
                float(flag[fam_te == fm].mean())
            for fm in np.unique(fam_te) if int(fm) != 0}
        return row

    print("\n" + "=" * 80)
    print("  MODERNTCN READOUT COMPARISON — same residuals, different readout")
    print("  (raw = E.mean(1) IS the native ModernTCN score, so raw doubles as the")
    print("  paper row. PRIMARY rows use NATIVE selection = lowest benign-val")
    print(f"  recon loss; @off{OFFICIAL_BUDGET_EPOCHS}ep = same rule within "
          f"the official budget; valAP-sel = selection-equalized secondary.)")
    print("=" * 80)
    print(f"  {'readout':<18}{'sel_ep':>7}{'val_AP':>9}{'test_AP':>9}"
          f"{'AUROC':>8}{'ho_AP':>8}{'TPR@1%':>8}{'FPR_act':>9}{'hoTPR':>7}")
    print("  " + "-" * 84)
    rows = {}
    rules = [("natsel", "nat", "native-sel"),
             ("official", "off", f"@off{OFFICIAL_BUDGET_EPOCHS}ep"),
             ("valap", None, "valAP-sel")]
    for rule_key, cap_key, rule_lbl in rules:
        for r in READOUTS:
            ck = cap_key if cap_key is not None else f"valap_{r}"
            if ck not in captures:
                continue
            cap = captures[ck]
            row = protocol_row(f"{r}[{rule_lbl}]", scores_test[ck][r],
                               cap["valb"][r])
            row["sel_epoch"] = int(cap["epoch"])
            row["val_ap"] = float(cap["val_ap"][r])
            rows[f"{r}_{rule_key}"] = row
            print(f"  {row['name']:<18}{row['sel_epoch']:>7}"
                  f"{row['val_ap']:>9.4f}{row['auprc_test']:>9.4f}"
                  f"{row['auroc']:>8.4f}{row['auprc_holdout']:>8.4f}"
                  f"{row['tpr_at_budget']:>8.4f}{row['fpr_actual']:>9.4f}"
                  f"{row['tpr_holdout']:>7.4f}")

    for rule_key, sel in (("natsel", "native-sel"), ("valap", "valAP-sel")):
        d_std = (rows[f"std_{rule_key}"]["auprc_test"]
                 - rows[f"raw_{rule_key}"]["auprc_test"])
        d_maha = (rows[f"maha_{rule_key}"]["auprc_test"]
                  - rows[f"raw_{rule_key}"]["auprc_test"])
        d_std_ho = (rows[f"std_{rule_key}"]["auprc_holdout"]
                    - rows[f"raw_{rule_key}"]["auprc_holdout"])
        d_maha_ho = (rows[f"maha_{rule_key}"]["auprc_holdout"]
                     - rows[f"raw_{rule_key}"]["auprc_holdout"])
        print(f"\n  CALIBRATION DELTA [{sel}] (this seed): "
              f"std−raw {d_std:+.4f}  maha−raw {d_maha:+.4f}   "
              f"(holdout: {d_std_ho:+.4f} / {d_maha_ho:+.4f})")

    for key in ("raw_natsel", "std_natsel"):
        print(f"\n  Per-family TPR@1% ({rows[key]['name']}):")
        for fam_name, tpr in sorted(rows[key]["per_family_tpr"].items()):
            fam_id = label_to_id.get(fam_name)
            tag = " [HOLDOUT]" if fam_id in holdout_ids else ""
            print(f"    {fam_name:<28s} {tpr:.4f}{tag}")

    result = {
        "seed": run_seed,
        "epochs_selected": {k: int(c["epoch"]) for k, c in captures.items()},
        "rows": rows,
        "holdout_families": {int(i): id_to_label.get(int(i), "?")
                             for i in holdout_ids},
        "n_test_windows": int(len(y_te)),
        "test_attack_rate": float(atk_te.mean()),
    }
    npz_payload = {
        "y_attack": y_te.astype(np.int8),
        "y_family": fam_te.astype(np.int16),
        "holdout_ids": np.array(sorted(holdout_ids), dtype=np.int16),
        "epoch_natsel": np.int32(captures["nat"]["epoch"]),
    }
    for r in READOUTS:
        npz_payload[f"{r}_natsel"] = scores_test["nat"][r]
        npz_payload[f"{r}_valap"] = scores_test[f"valap_{r}"][r]
        npz_payload[f"valb_{r}_natsel"] = captures["nat"]["valb"][r]
        npz_payload[f"valb_{r}_valap"] = captures[f"valap_{r}"]["valb"][r]
        npz_payload[f"epoch_valap_{r}"] = np.int32(
            captures[f"valap_{r}"]["epoch"])
    np.savez_compressed(os.path.join(OUTPUT_DIR, "test_scores.npz"),
                        **npz_payload)
    print(f"  Dumped selected-epoch test scores -> "
          f"{os.path.join(OUTPUT_DIR, 'test_scores.npz')}")
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(deep_convert(result), f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(deep_convert(history), f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/")
    return result


def main():
    if not HOLDOUT_FAMILY_NAMES:
        raise RuntimeError(
            "HOLDOUT_FAMILY_NAMES is empty. The evaluation protocol requires "
            "pre-registration BEFORE any experiment run.")

    all_results = []
    for s in RUN_SEEDS:
        print(f"\n\n{'=' * 80}\n  STARTING MODERNTCN BCCC-CSE-CIC-IDS2018 RUN WITH SEED {s}"
              f"\n{'=' * 80}")
        all_results.append(run_experiment(s))

    print(f"\n\n{'=' * 80}\n  AGGREGATED MODERNTCN RESULTS OVER "
          f"{len(RUN_SEEDS)} SEEDS (reported protocol)\n{'=' * 80}")
    agg_keys = ([f"{r}_natsel" for r in READOUTS]      # PRIMARY (native sel)
                + [f"{r}_official" for r in READOUTS]  # faithful 2-ep budget
                + [f"{r}_valap" for r in READOUTS])    # secondary (val-AP sel)
    for metric in ("auprc_test", "auprc_holdout", "auroc", "tpr_at_budget"):
        print(f"\n  {metric}:")
        for k in agg_keys:
            vals = np.array([res["rows"][k][metric] for res in all_results
                             if k in res["rows"]], dtype=float)
            if not len(vals):
                continue
            print(f"    {k:<16s}: {vals.mean():.4f} +/- {vals.std():.4f}   "
                  f"per-seed {np.round(vals, 4).tolist()}")

    print("\n  PAIRED CALIBRATION DELTA (the systematic-under-reading test):")
    for rule_key, sel in (("natsel", "native-sel"), ("valap", "valAP-sel")):
        for cal in ("std", "maha"):
            for metric, lbl in (("auprc_test", "test"),
                                ("auprc_holdout", "ho")):
                d = np.array([res["rows"][f"{cal}_{rule_key}"][metric]
                              - res["rows"][f"raw_{rule_key}"][metric]
                              for res in all_results], dtype=float)
                print(f"    [{sel}] {cal}−raw [{lbl:<4}]: mean {d.mean():+.4f}"
                      f"  positive on {(d > 0).sum()}/{len(d)} seeds  "
                      f"per-seed {np.round(d, 4).tolist()}")

    out_root = experiment_root_dir()
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, "aggregated_results.json")
    with open(out_path, "w") as f:
        json.dump(deep_convert(all_results), f, indent=2)
    print(f"\nSaved aggregated results to {out_path}")


if __name__ == "__main__":
    main()
