# Texture-penalized confidence

Code, trained weights, the frozen configuration and the region lists for the
letter *Texture-Penalized Confidence for Accepting the Outputs of a
Landsat-8-Trained Cloud Mask on Sentinel-2* (Bowen Zhang, Pengcheng Luo,
Hongjun Xie, Shujin Jia, Genke Yang and Jian Chu, 2026).

A per-pixel cloud mask trained on Landsat 8 (SPARCS) is applied without
retraining to Sentinel-2 tiles of CloudSEN12+, and a gate accepts the top 80%
of its tile calls. Texture-penalized confidence evaluates a local binary
pattern texture score only on cloud calls and subtracts an error probability,
fitted on labelled Landsat 8 tiles, from their confidence before that one
global cut. It uses no Sentinel-2 labels at fitting or decision time; the
texture score and the configuration were chosen on labelled development data.

## Contents

| Path | What it holds |
|---|---|
| `models/member{0..4}.pt` | The five ensemble members of the mask (PyTorch state dicts, 4,675 parameters each); `member0` is the single model the gates score. `split.json` and `train-log.json` record the SPARCS split and training. SHA-256 sums are in `tasks/premise-test/evidence/freeze.json` (`model.member_sha256`). |
| `tasks/texture-s2-supplementary/evidence/freeze.json` | The frozen configuration: the k-means texture prototypes per LBP radius (`prototypes`, radii 1, 3 and 9), the cloud-call error mapping (`cloud_call_mapping`), the cloud-fraction baseline's source scales (`scale`) and the constants. `tasks/texture-s2-confirmation/evidence/freeze.json` repeats it for the untouched regions. |
| `tasks/untouched-s2-population/evidence/manifest.csv` | The untouched regions: 330 regions (1,650 patches) of CloudSEN12+'s training split, with the selection record in `selection.json`. |
| `tasks/s2-independent-confirmation/evidence/manifest.csv` | The test split: the 165 official CloudSEN12+ test regions (825 patches). |
| `tasks/*/evidence/results.json` | The retained results each script wrote. The letter's numbers come from `texture-s2-confirmation` (untouched regions), `texture-s2-supplementary` (test split), `texture-permutation-premise` (error mapping and the 14 candidate scoring rules), `texture-conf-paper-readouts` and `texture-veto-midterm`. |
| `src/onboard/` | Shared code: the mask, tiling, gate scores, metrics, SPARCS and HYPSO-1 readers. |
| `experiments/<task>/` | The scripts of each step, below. |

The development regions are CloudSEN12+'s official validation split (107
regions, 535 patches), so they need no list.

## Steps

Each script takes its task directory as the first argument and writes to
`<task>/evidence/`, for example
`uv run python experiments/texture-s2-confirmation/evaluate.py tasks/texture-s2-confirmation`.

| Folder | Step |
|---|---|
| `premise-test` | SPARCS preprocessing (`prep_sparcs.py`) and training of the five members (`train.py`). |
| `texture-permutation-premise` | Source-only texture prototypes and error mappings fitted on SPARCS calibration tiles, and the development comparison of the candidate scoring rules. |
| `texture-s2-supplementary` | Frozen texture-penalized confidence (`frozen.py`) and its scoring on the 165 test regions: `prepare.py`, `runtime.py` (image-only, timed), `evaluate.py`, `check.py`. |
| `untouched-s2-population` | Selection of the untouched regions from metadata (`select.py`), download and checks. |
| `texture-s2-confirmation` | The same frozen scoring on the untouched regions: `prepare.py`, `runtime.py`, `truth.py` (labels read only after the prediction pass), `evaluate.py`, `check.py`. |
| `s2-independent-confirmation` | Test-split download, label conversion (`truth.py`) and the earlier runtime pass. |
| `zero-label-s2-test` | The cloud-fraction baseline on the test split. |
| `texture-conf-paper-readouts` | Post hoc readouts reported in the letter. |
| `texture-veto-midterm` | The development test of the hard texture veto and the class-blind mapping fit. |
| `cloudsen12-room`, `single-forward-correctness-gate`, `texture-full-diagnostic`, `budget-swap-gate` | Earlier steps whose readers, predictor and frozen scales the scripts above import. |

## Setup and data

`uv sync` installs the dependencies (Python 3.12). The data are not included:

- CloudSEN12+ (Aybar et al.), the `train_509_high`, `validation_509_high` and
  `test_509_high` files at
  <https://huggingface.co/datasets/isp-uv-es/CloudSEN12Plus>; the manifests
  give each patch's byte range.
- SPARCS (Hughes and Kennedy, 2019), from the U.S. Geological Survey.
- The labelled HYPSO-1 captures (DataverseNO), used in development only.

The scripts read data under `~/Data/onboard-acceptance`. The frozen records
keep the absolute paths and commit hashes of the authors' run (paths under
`/Users/zhangbowen/Data/onboard-acceptance`, commits of the authors' research
repository, which is not published); to re-run, replace that path prefix with
the local data directory.

Per-tile features, labels and decisions are not included. With them in place,
`check.py` recomputes each population's `results.json` exactly:

- untouched regions: `features.csv` and `labels.csv` in
  `tasks/texture-s2-confirmation/evidence/`, written by that folder's
  `runtime.py` and `truth.py`;
- test split: `features.csv` in `tasks/texture-s2-supplementary/evidence/`,
  written by that folder's `runtime.py`, and `features-single-1.csv`,
  `features-ensemble-1.csv` and `decisions.csv` in
  `tasks/s2-independent-confirmation/evidence/`, written by that folder's
  `runtime.py`, `truth.py` and `evaluate.py`.

The development steps (`premise-test`, `texture-permutation-premise`,
`texture-full-diagnostic` and the earlier folders) also read per-tile files
and raw HYPSO-1 data that are not shipped; their scripts are the code as run,
and their outputs are the frozen records listed above.

## Licence

The code, weights and lists are under the MIT licence ([LICENSE](LICENSE)).
CloudSEN12+, SPARCS and HYPSO-1 keep their own licences.
