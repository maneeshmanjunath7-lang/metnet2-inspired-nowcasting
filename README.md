# Data-Driven Weather Forecasting with Deep Learning

Repository companion for the TUM theoretical semester thesis **“Data-Driven
Weather Forecasting with Deep Learning: Reproducing the MetNet-2 Model”**
(submitted 10 December 2025).

The work investigates a reduced MetNet-2-style precipitation-nowcasting model
that can be trained and evaluated with limited academic computing resources.
It uses DWD RADOLAN RW radar composites rather than attempting to reproduce the
full operational MetNet-2 system.

## Experiment at a glance

- Data: RADOLAN RW composites at five-minute temporal resolution.
- Input: six frames, representing 30 minutes of history.
- Output: eight frames, representing a 40-minute forecast horizon.
- Spatial resolution: 256 × 256 pixels in the submitted experiment.
- Training set: 1,000 sequences.
- Evaluation set: 1,000 sequences.
- Evaluation: MSE, MAE, probability of detection, false alarm ratio, critical
  success index, and visual case studies.

The reported errors increase steadily with lead time. MSE rises from `0.0028`
at the first forecast step to `0.0081` at the eighth; MAE rises from `0.033` to
`0.060`. Broad stratiform motion remains coherent, while small convective
features and peak intensities become smoother at longer lead times. The full
numeric table is available in [`results/lead-time-errors.csv`](results/lead-time-errors.csv).

## Repository contents

- `archive-prototype/`: recoverable exploratory Python files from the supplied
  `Weather Forecasting.zip`, with text preserved and line endings normalized.
- `results/lead-time-errors.csv`: the submitted per-lead-time MSE and MAE table.
- `ARCHIVE_AUDIT.md`: provenance, exclusions, reproducibility limits, and known
  source issues.
- `MANUSCRIPT_REVIEW.md`: concise technical reading of the submitted thesis.

## Important provenance note

The supplied ZIP contains exploratory scripts, checkpoints, sample RADOLAN
files, a complete local virtual environment, presentations, recordings,
third-party papers, and an unrelated LaTeX template. The final manuscript cites
later scripts such as `dataset_radolan.py`, `model_minimetnet2.py`, and
`evaluate_minimetnet2.py`, but those files are not present as standalone source
files in the archive. The manuscript includes them as a typeset appendix.

For that reason, this repository preserves and documents the recoverable code
without claiming that the archive is a turn-key reproduction of the submitted
experiment.

## Data and manuscript policy

Raw RADOLAN archives, generated caches, model checkpoints, the local `.venv`,
recordings, and third-party PDFs are intentionally excluded. They are large,
generated, licensed separately, or contain information that should not be
published as source code.

The submitted 72-page PDF was reviewed to prepare this package but is not
committed here because its front matter contains personal academic identifiers
and declarations. A deliberately redacted or approved manuscript can be added
later.

## Inspecting the archived prototype

Create a fresh environment and install the declared dependencies manually. Do
not restore the archived `.venv`.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r archive-prototype/requirements.txt pillow
```

Read [`ARCHIVE_AUDIT.md`](ARCHIVE_AUDIT.md) before running the scripts. Several
files are exploratory and contain hard-coded local paths, missing imports, or
filename mismatches. They are preserved as evidence of the development process,
not advertised as a maintained package.
