# Optimus 5-Prime PyTorch Reproduction

This repository provides one simple pipeline for predicting mean ribosome load (MRL) from 50-nucleotide 5′ UTR sequences. `Pipeline.ipynb` uses Python backend files for four models:

- Ridge Regression
- Random Forest
- XGBoost
- Optimized Multiscale CNN

The original CNN is not included.

## Dataset and split

The pipeline reads `data/GSM3130435_egfp_unmod_1.csv.gz`, which contains 326,033 sequences. It sorts rows by `total_reads` in descending stable order and creates these shared splits:

- High-read Test: first 20,000 rows
- Training: 240,000 rows randomly selected from rows 20,000 through 279,999
- Validation: the other 20,000 rows in that range
- Low-read Test: remaining 46,033 rows

The random split uses NumPy `default_rng(seed=42)`. Split indices are saved to `results/split_indexes.npz`. All models use `ATGC` One-Hot encoding. The CNN receives `(N, 4, 50)` arrays, and the baseline models receive flattened `(N, 200)` arrays.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m ipykernel install --user \
  --name optimus5prime-pytorch-repro \
  --display-name "Python (.venv: optimus5prime)"
```

## Running the pipeline

Open the notebook from the repository root:

```bash
jupyter lab Pipeline.ipynb
```

In the Training section, edit `models_to_train` to choose models. Run the four sections in order: Data Loading, Training, Inference, and Results and Visualization. Training may require substantial time and memory.

Each model runs in an isolated Python process. This prevents native runtime conflicts between PyTorch and XGBoost on macOS while preserving the shared deterministic data split. CUDA is preferred when available, followed by Apple MPS and then CPU for CNN training and inference.

Each model writes its checkpoint, predictions, and metrics to `results/<model>/`. The CNN also writes its training history. Combined metrics and split indices are stored directly in `results/`. Generated results and checkpoints are ignored by Git.

No training results are included or claimed in this refactor.

## References

- Sample, P. J. et al. [Human 5′ UTR design and variant effect prediction from a massively parallel translation assay](https://www.nature.com/articles/s41587-019-0164-5). *Nature Biotechnology* 37, 803–809 (2019).
- [Original Optimus 5-Prime source code](https://github.com/pjsample/human_5utr_modeling)

## Authorship and AI use

The initial CNN model was developed by Brain, with further optimization and the split protocol implemented by Evan Lin on `feature/CNN_optimized`. The remaining baseline models were implemented by Zehao on `feature/ml-baselines`. The repository's commit history preserves the original contributions.

This integration refactor was created with assistance from OpenAI Codex. Human review by the project authors is required before course submission; this repository does not claim that review or experimental validation has been completed.
