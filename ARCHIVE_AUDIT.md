# Archive and reproducibility audit

## Sources reviewed

- `Weather Forecasting.zip`: 32,137 file entries, 3.409 GB uncompressed and
  1.313 GB compressed.
- `Weather Forecasting/`: 32,137 files and approximately 3.66 GB on disk.
- Submitted thesis PDF: 72 A4 pages, SHA-256
  `065D911A3470EBB2E27EAE3BE94B988AF5A61EC9663E8D9801BE261013875A70`.

The PDF title, abstract, methodology, results, discussion, conclusion,
references, and complete source-code appendix were visually reviewed. The
document renders cleanly overall; the content notes below are retained for
future revision.

## What was preserved

The top-level exploratory Python scripts and `requirements.txt` from
`Weather Forecasting/Project Code/` were preserved under `archive-prototype/`.
Text content is unchanged except for normalization to LF line endings.

## What was excluded

- `Project Code/.venv/` (more than 31,000 archive entries including PyTorch
  binaries and installed third-party packages).
- Raw and nested RADOLAN data archives.
- Generated `.npy` caches and model checkpoints (`.pt` / `.pth`).
- Audio recordings, presentations, handbooks, and third-party papers.
- The unrelated automotive-structure LaTeX template found under
  `Vorlage LateX/`.
- Tiny placeholder files under `metnet2-thesis/`; several are only 27–35 bytes
  and do not contain an implementation.

## Gap between archive and submitted manuscript

The final appendix cites the following standalone sources, none of which were
found in the supplied folder or ZIP:

- `dataset_radolan.py`
- `evaluate_minimetnet2.py`
- `model_base.py`
- `model_minimetnet2.py`
- `select_strong_rainy_sequences.py`
- `test_read_one.py`
- `train.py` (the archive contains a different early script with this name)
- `train_minimetnet2.py`
- `visualize_example.py`
- `visualize_multistep_metnet2.py`
- `visualize_multistep.py`
- `visualize_predictions.py`
- `visualize_random3_metnet2.py`
- `visualize_strong_rainy_metnet2.py`

The PDF appendix is therefore the only supplied copy of the final source
listings. Reconstructing executable files from a typeset PDF would require a
separate transcription and validation task.

## Known issues in the archived prototype

- `Cuda.py` contains a shell installation command and is not valid Python.
- `search for gpu.py` refers to `MetNet2Mini` without importing it.
- `train_radolan.py` imports `mini_metnet_demo`, while the archived filename is
  `mini_metnet.py`.
- `model.py` applies `softmax` before `CrossEntropyLoss`; PyTorch's loss expects
  logits, so this changes the intended optimization.
- `radolan_dataset.py` checks only for `.tar`, while the archived sample is a
  `.tar.gz` file.
- Multiple scripts contain absolute paths to the original Windows folder.
- The exploratory archive predicts one future frame in several scripts, while
  the submitted experiment predicts eight.
- The archived `requirements.txt` omits Pillow even though
  `radolan_dataset.py` imports it.

These files remain unchanged so that the repository does not blur archival
provenance with later repairs.

## Manuscript quality notes

- Discussion section 5.4 contains an unresolved `Table ??` cross-reference.
- Several words are missing spaces (for example, `simplifiedform`) and a few
  sentences need grammar cleanup.
- The abstract describes the model as a transparent, resource-constrained
  educational reproduction, which is consistent with the results and should
  remain the framing used by the portfolio.
