# Optimus 5-Prime PyTorch Reproduction

A PyTorch-based reproduction of Optimus 5-Prime for predicting mean ribosome load (MRL) from 50-nucleotide 5′ UTR sequences.

This project compares an optimized Multiscale CNN with three machine learning baseline models. All models use the same dataset and data split for consistent evaluation.

## 1. Models

Four models are included:

- **Ridge Regression:** Linear regression with L2 regularization.
- **Random Forest:** An ensemble of decision trees.
- **XGBoost:** Gradient boosting with decision trees.
- **Multiscale CNN:** A PyTorch model using parallel convolutions, dilated residual blocks, and auxiliary prediction.

The CNN uses convolution kernel sizes of 3, 7, and 11, followed by three residual blocks. It predicts MRL and 14 auxiliary ribosome-fraction values.

## 2. Dataset

The dataset contains 326,033 sequences from the original Optimus 5-Prime study.

File: `data/GSM3130435_egfp_unmod_1.csv.gz`

Sequences are sorted by `total_reads` in descending order.

| Dataset        | Samples | Description                                         |
| -------------- | ------: | --------------------------------------------------- |
| Training       | 240,000 | Randomly selected from the middle 260,000 sequences |
| Validation     |  20,000 | Remaining sequences from the same group             |
| High-read Test |  20,000 | Sequences with the highest read counts              |
| Low-read Test  |  46,033 | Remaining low-read sequences                        |

The random seed is 42.

All models use the same split and One-Hot encoding with nucleotide order `ATGC`. The CNN receives `(N, 4, 50)` inputs, while baseline models use flattened `(N, 200)` features.

The high-read test set follows the original paper's evaluation approach. The separate validation and low-read test sets are additions to this reproduction.

## 3. Installation

Clone the repository:

```bash
git clone https://github.com/LouiSyndrom/optimus5prime-pytorch-repro.git
cd optimus5prime-pytorch-repro
```

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The project requires Python 3.10 or newer.

## 4. Running the Pipeline

Start JupyterLab:

```bash
jupyter lab Pipeline.ipynb
```

The Notebook contains four sections:

1. **Data Loading:** Load the dataset and create shared data splits.
2. **Training:** Train selected models or reuse existing checkpoints.
3. **Inference:** Evaluate models on high-read and low-read test sets.
4. **Results and Visualization:** Compare performance and display prediction plots.

To select models, edit:

```python
models_to_train = ["ridge", "random_forest", "xgboost", "cnn"]
```

Existing checkpoints are reused by default. To train the selected models again, set:

```python
force_retrain = True
```

Models run in separate Python processes to avoid native library conflicts between PyTorch and XGBoost.

CNN training supports CUDA, Apple MPS, and CPU.

## 5. Results

The following results are saved in the repository under `results/`.

| Model              | High-read Pearson r² | High-read RMSE | Low-read Pearson r² |
| ------------------ | -------------------: | -------------: | ------------------: |
| Ridge              |               0.1277 |         1.9523 |              0.0233 |
| Random Forest      |               0.2367 |         1.8283 |              0.0636 |
| XGBoost            |               0.6463 |         1.2951 |              0.2068 |
| **Multiscale CNN** |           **0.9500** |     **0.5981** |          **0.3234** |

The Multiscale CNN achieved the highest Pearson r² and lowest RMSE on the high-read test set.

All models performed worse on low-read data. This may reflect increased measurement noise, differences in data distribution, or both.

Pearson r² is the squared Pearson correlation coefficient, not the standard regression R² score.

The pipeline also saves MSE, RMSE, Pearson r, and regression R² for each model.

## 6. Project Structure

```text
optimus5prime-pytorch-repro/
├── Pipeline.ipynb
├── src/
│   ├── config.py
│   ├── data.py
│   ├── models/
│   │   ├── baselines.py
│   │   └── cnn.py
│   ├── training.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── pipeline.py
│   └── runner.py
├── data/
├── results/
├── pyproject.toml
└── README.md
```

All model implementations, training functions, inference, and visualization functions are stored in Python files.

`Pipeline.ipynb` serves as the main interface.

The `results/` directory contains saved model checkpoints, predictions, evaluation metrics, training history, and figures.

The combined model results are available in `results/model_comparison.csv`.

## 7. References

Sample, P. J. et al. (2019). [Human 5′ UTR design and variant effect prediction from a massively parallel translation assay](https://www.nature.com/articles/s41587-019-0164-5). *Nature Biotechnology*, 37, 803–809.

Original implementation: [https://github.com/pjsample/human_5utr_modeling](https://github.com/pjsample/human_5utr_modeling)

## Authorship

The initial CNN architecture was developed by Brain, and the optimized Multiscale CNN and its data-splitting protocol were developed by Evan Lin.

Ridge Regression, Random Forest, and XGBoost baseline implementations were developed by Zehao.

The repository's branch and commit history preserves the original contributions.

