# Gradient-enhanced physics-informed LSTM for RUL prediction

Curated research code associated with Chenyang Lai's work. [Associated publication](https://doi.org/10.1016/j.ress.2025.111485)

This is a **runnable, curated recovery** of selected legacy code. The default experiment is deliberately small and checks data loading, training, held-out prediction and checkpoint saving. It does **not** reproduce the paper's numerical results.

## Method

A recurrent RUL model uses input features plus an explicit time coordinate at its output layers. First-derivative constraints encode decreasing remaining life; a second-derivative penalty encourages consistency of that relation.

## Install and run

Use **CPython 3.12** and a CPU environment. The pinned PyTorch wheel requires **macOS 14 or later on Apple Silicon**.
This dependency set has no Intel macOS PyTorch wheel; Intel Macs require a separately selected and tested environment.
The preceding audit passed hosted CPU checks on Linux, macOS and Windows. Check the current PR checks for this revision; passing smoke tests does not validate all platform configurations.
CUDA and MPS execution are not implemented by this runner. MATLAB, ANSYS and TensorFlow are not required for the curated entry point.

Clone the repository using an account with access while it remains private:

```sh
git clone https://github.com/wowSUNBOY/gradient-enhanced-pilstm.git
cd gradient-enhanced-pilstm
```

On macOS or Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==26.2.2
python -m pip install -r requirements-tested.txt
python -m pip check
python run.py --demo --epochs 3 --max-samples 32 --output outputs/demo
```

On Windows, create the environment with `py -3.12 -m venv .venv` and invoke `.venv\Scripts\python.exe` for each Python command.
This avoids any PowerShell activation-policy changes. `requirements.txt` selects the same pinned direct dependencies.
`requirements-lock-macos-py312.txt` records the complete macOS validation environment; it is not a universal platform lock.

The synthetic demonstration generates examples in memory and requires no research dataset. For authorized local research data, point to the original project root described in [DATA.md](DATA.md):

```sh
python run.py --data-root "/absolute/path/to/original/project" --epochs 3 --max-samples 32 --output outputs/real
```

`--epochs`, `--batch-size`, `--lr`, `--seed`, `--max-samples` and `--output` are configurable. Some models have additional fixed caps documented below. Longer training alone does not restore the full published protocol.

Use a new or empty output directory for every run; existing results are protected from accidental overwriting.

The output folder contains metrics/history, predictions and a checkpoint (`model.pt`); preprocessing information is saved separately or inside the joblib bundle. The documented output folder and common data formats are git-ignored. Inspect `git status` before committing any generated files. Regression MSE is in normalized coordinates unless the task says otherwise.

To replay held-out inference from a checkpoint (not resume training):

```sh
python run.py --demo --seed 22 --max-samples 32 --checkpoint outputs/demo/model.pt --output outputs/replay
```

Replay requires the sibling `protocol.json`, identical data, data mode, seed, sample cap, repository and maintained source code. The pipeline rebuilds its deterministic preprocessing and data split; this is not an arbitrary new-data prediction API. Protocol version 3 checks SHA-256 fingerprints of input files and maintained source code before loading weights. Identical data may be relocated to another root directory.
Use the same code revision and dependency versions for checkpoint replay. Input fingerprints do not establish scientific equivalence.
Create new checkpoints with this version; older protocols are rejected. `environment.json` records the runtime versions.
Protocol files no longer store the absolute data-root path. They still contain relative input identifiers and hashes; predictions, selected features and metrics may reveal research information. Keep output bundles local until separately approved for disclosure.
Only load checkpoints from trusted sources.

## Data and execution protocol

Split entire cells deterministically 80/20 (seed 22). Fit feature, time and RUL normalization on training cells. Use nonoverlapping length-32 windows, hidden size 32, one LSTM layer and output 33→16→1. Loss weights: data 0.9, first derivative 0.1, second derivative 0.01. Normalized dRUL/dt = -range(time)/range(RUL).

## Validation

```sh
python -m unittest discover -s tests -v
```

The regression suite trains on synthetic data and compares predictions after checkpoint replay. It also checks invalid arguments, changed input fingerprints, output protection and missing-data errors. [VALIDATION.md](VALIDATION.md) records local-data checks without including the underlying data.

## Scope and limitations

This runner selects one battery-data path, uses a compact model and capped windows, and has no ensemble, hyperparameter search or complete cross-validation. It does not reproduce all electronic-component case studies or published performance.

## Repository layout and provenance

- `run.py`: command-line entry point.
- `experiment.py`: data selection, training objective and evaluation protocol.
- `model.py`: extracted model definitions or explicitly documented port.
- `runtime.py`: shared CPU experiment utilities.
- `legacy/`: selected archival source for traceability; notebooks have outputs and execution metadata removed. These archival files are reference material, not supported executable entry points. They retain historical paths, optional dependencies and some unavailable local-module imports; use `run.py`. Their syntax is checked separately from execution.
- `project.json`: original relative source paths and original SHA-256 hashes. Original hashes are retained; `archived_sha256` records the current reviewed copy, including any documented safety edits or notebook sanitization.
- `CHANGELOG.md`: changes and deviations from the archive.

Original folders were left unchanged. No raw data, manuscript, trained research weights or partner documents are distributed. No open-source license has been assigned; see [RIGHTS.md](RIGHTS.md). Keep this repository private until its scope and rights are reviewed.

## Before public release

See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) and [RELEASE_DECISIONS.md](RELEASE_DECISIONS.md) for the exact unresolved rights, data-access and scientific decisions. Passing an execution check does not establish publication permission or reproduce scientific results.

## Developer checks

Install `requirements-dev.txt` in addition to the application requirements to run the same lint and advisory tools used by CI:

```sh
python -m pip install -r requirements-tested.txt -r requirements-dev.txt
python -m pip check
python -m unittest discover -s tests -v
python -m pip_audit
```

The audit-tool constraints include a patched certifi CA bundle. Advisory results are time-dependent; a clean audit is not a guarantee of security.
