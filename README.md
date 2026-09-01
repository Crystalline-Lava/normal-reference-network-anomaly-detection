# Network-flow anomaly detection: reproducibility code

This directory is a cleaned, public-release copy of the code used for the
ESWA manuscript.

## Layout

- `proposed/`: the proposed normal-reference detector:
  `proposed_cic_unsw_nb15.py` and `proposed_bccc_cse_cic_ids2018.py`.
- `baselines/timesnet/`: `timesnet_bccc_cse_cic_ids2018.py`, plus
  entity-window and native-stream CIC-UNSW-NB15 variants.
- `baselines/moderntcn/`: `moderntcn_bccc_cse_cic_ids2018.py`, plus
  entity-window and native-stream CIC-UNSW-NB15 variants.
- `baselines/usad/`: `usad_bccc_cse_cic_ids2018.py` and
  `usad_cic_unsw_nb15.py`.

All Python files use the naming pattern `<model>_<dataset>[_<variant>].py`.
Here, `entity` means the entity-grouped window protocol, while `native` means
the baseline's native continuous-stream preprocessing. The names describe
what each file runs; they do not refer to historical source filenames.

The dataset slugs are fixed as follows:

| Formal dataset name | Code slug |
| --- | --- |
| CIC-UNSW-NB15 | `cic_unsw_nb15` |
| BCCC-CSE-CIC-IDS2018 | `bccc_cse_cic_ids2018` |

The abbreviated slug `cic_ids2018` is intentionally not used, because it
omits the BCCC-CSE dataset provider/name prefix.

Each retained entry point has a manuscript role; none is an alternative copy
of the same reported run:

| Entry point | Dataset | Manuscript role |
| --- | --- | --- |
| `proposed/proposed_cic_unsw_nb15.py` | CIC-UNSW-NB15 | Retained S1 detector with P0 and S1_nodelta paired controls |
| `proposed/proposed_bccc_cse_cic_ids2018.py` | BCCC-CSE-CIC-IDS2018 | Retained S1 detector |
| `baselines/timesnet/timesnet_cic_unsw_nb15_entity.py` | CIC-UNSW-NB15 | Shared entity-window comparison |
| `baselines/moderntcn/moderntcn_cic_unsw_nb15_entity.py` | CIC-UNSW-NB15 | Shared entity-window comparison |
| `baselines/usad/usad_cic_unsw_nb15.py` | CIC-UNSW-NB15 | Shared entity-window comparison |
| `baselines/timesnet/timesnet_bccc_cse_cic_ids2018.py` | BCCC-CSE-CIC-IDS2018 | Shared entity-window comparison |
| `baselines/moderntcn/moderntcn_bccc_cse_cic_ids2018.py` | BCCC-CSE-CIC-IDS2018 | Shared entity-window comparison |
| `baselines/usad/usad_bccc_cse_cic_ids2018.py` | BCCC-CSE-CIC-IDS2018 | Shared entity-window comparison |
| `baselines/timesnet/timesnet_cic_unsw_nb15_native.py` | CIC-UNSW-NB15 | Secondary native flow-level experiment |
| `baselines/moderntcn/moderntcn_cic_unsw_nb15_native.py` | CIC-UNSW-NB15 | Secondary native flow-level experiment |

The two `native` scripts are required for the separately reported flow-level
experiment. USAD is absent from that experiment by design, matching the manuscript.

## Data and environment

The CIC-UNSW-NB15 and BCCC-CSE-CIC-IDS2018 datasets must be obtained from
their respective providers. Do not commit raw datasets, trained checkpoints,
generated PDFs, caches, local paths, credentials, or `.git` directories.

The experiments require Python 3.10+ and the packages listed in
`requirements.txt`. GPU training additionally requires a compatible PyTorch
installation.

The scripts retain their reported experiment protocols, but their input and
output roots should be supplied through environment variables before running;
`.env.example` documents the expected locations. BCCC-CSE-CIC-IDS2018 scripts
use `BCCC_CSE_CIC_IDS2018_ROOT` and `CIC_BASE_DIR`; CIC-UNSW-NB15 scripts use
`CIC_UNSW_RAW_CSV` and `CIC_BASE_DIR`. The scripts still contain
Colab-compatible fallback paths for continuity with the original experiments,
but those defaults should be overridden in a public deployment. No claim is
made that this release trains models without dataset-specific configuration.

For example, from this directory:

```powershell
$env:CIC_UNSW_RAW_CSV = "D:\data\CIC-UNSW-NB15\CICFlowMeter_out.csv"
$env:CIC_BASE_DIR = "D:\runs\time-sphere"
python proposed\proposed_cic_unsw_nb15.py
```

The CIC-UNSW-NB15 proposed entry point defaults to `S1`, the full detector in
the paper. `P0` is the no-SVDD paired control, and `S1_nodelta` is the
no-delta-feature paired control. The BCCC proposed
entry point likewise defaults to the S1 loss configuration (KDE-weighted
prediction plus train-time SVDD, with variance/covariance penalties disabled).

All reported runs use the fixed seed tuple `(42, 456, 7, 789, 1024)`. `SEED`
is the active model seed for one run. BCCC-CSE-CIC-IDS2018 scripts additionally
use `DATA_SEED` for the shared split and preprocessing cache so every model run
uses exactly the same data partition.

The BCCC proposed script supports checkpoint-only re-evaluation by setting
`BCCC_CSE_CIC_IDS2018_SKIP_TRAINING=1`; the corresponding `best_*.pt` files
must already exist in the resolved run directory. The baseline entry points do
not advertise a generic checkpoint-skip interface unless the script itself
defines one.

## Reproducibility status


The baseline scripts contain the exact model definitions used by the released
pipelines, so no separate model package is required. Similar preprocessing
blocks across dataset-specific scripts are deliberate: each pipeline remains
standalone and preserves the protocol used for its reported baseline.

## License

Original code in this repository is released under the [MIT License](LICENSE),
Copyright (c) 2026 Yiyang Sun. Adapted TimesNet, ModernTCN, and USAD portions
remain subject to their upstream licenses and copyright notices; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full terms and sources.

The datasets are not distributed with this repository and are not covered by
the repository license. Obtain them from their providers and follow the
providers' respective terms of use.
