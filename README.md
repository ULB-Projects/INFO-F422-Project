# INFO-F422 Project: Learning a MIMO NARX model

Team project for **INFO-F-422 – Statistical Foundations of Machine Learning** (Université Libre de Bruxelles, 2025-2026).

The goal is to identify an unknown two-input / two-output **NARX** (Nonlinear AutoRegressive eXogenous) system from 1000 training samples (`Utr`, `Ytr`) and to predict its outputs over a **1000-step horizon** for two unseen input sequences (`Uts1`, `Uts2`). Predictions are evaluated with RMSE and NMSE.

## Approach

The full, documented implementation lives in [`THE Notebook.ipynb`](THE%20Notebook.ipynb). In short:

1. **Pilot simulation.** Two known NARX systems (NARX1: 1 input / 2 outputs, NARX2: 2 inputs / 2 outputs) are simulated, with configurable series length and noise variance, to design and validate the methodology before touching the real data.
2. **Long-horizon strategy.** Two ways of producing a 1000-step prediction are compared: a recursive **MIMO** strategy (predictions fed back as inputs, with the risk of error explosion) and a **MISMO** strategy that predicts blocks of `s` steps ahead. The final pipeline uses MISMO.
3. **Parameter selection.** The delay `d` is estimated from the cross-correlation between inputs and outputs; the output lag `n_a` and the block size `s` (or input lag `n_b`) are chosen by grid search on a chronological validation split.
4. **Baseline approach.** Non-deep regression models (linear regression, Ridge, k-NN, random forest, ...) are compared with time-series cross-validation and hyper-parameter tuning, and wrapped in a scikit-learn `Pipeline`.
5. **Deep learning approach.** The pipeline is extended with neural models (MLP and LSTM, implemented in PyTorch) and compared against the baseline.
6. **Deployment and submission.** The selected pipeline is trained only on `Utr` / `Ytr`, used to predict both test series, and the predictions (`Yhat1`, `Yhat2`, each of shape `(1000, 2)`) are saved to a `submission.npz` file.

## Repository structure

| Path | Content |
| --- | --- |
| `THE Notebook.ipynb` | Main notebook: the complete pipeline, from pilot simulation to submission |
| `StudentdataNARX.npz`, `data/` | Course dataset (`Utr`, `Ytr`, `Uts1`, `Uts2`) |
| `notebooks/` | Working notebooks produced during development (baseline approach, parameter selection, pipeline tests, earlier version of the main notebook) |
| `KamilaWork/` | Individual exploration notebook (in Spanish) |
| `results/predictions/`, `notebooks/submission.npz` | Saved prediction files (`Yhat1`, `Yhat2`) |

## Running the notebook

Requirements: Python 3 with `numpy`, `pandas`, `matplotlib`, `scikit-learn` and `torch`.

Start Jupyter **from the repository root**, since the notebook loads `StudentdataNARX.npz` from the working directory, and run `THE Notebook.ipynb` from top to bottom. Some outputs contain absolute paths from the authors' computers; they can be ignored.

## Authorship and privacy note

This project was developed as a team project together with other students:

- Guillaume Poth ([@GuillaumePoth](https://github.com/GuillaumePoth))
- Hamza ([@Hamza-RAOUI](https://github.com/Hamza-RAOUI))
- Hubert ([@Hubert99638](https://github.com/Hubert99638))
- Kamila Obregon ([@kmor8](https://github.com/kmor8))

The original course repository is private. This copy shares the final version of the work **without the commit history**, to protect the privacy of my teammates (their commits carry personal e-mail addresses that are not mine to publish).

## Academic integrity

This repository is shared as a portfolio of finished work. If you are taking INFO-F-422, remember that the course forbids copying code or reports from other teams, current or past.
