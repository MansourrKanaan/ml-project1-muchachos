# ML Project 1 – Heart Disease (MICHD) Prediction

EPFL CS-433 Machine Learning, Fall 2026.

## Team
- Mansour Kanaan
- 
- 

## Setup

Only NumPy (and matplotlib for plots) is required.

Download `x_train.csv`, `y_train.csv` and `x_test.csv` from the
[AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
and place them in a `data/` folder at the root of the repository (file names unchanged).

## Reproducing our submission

```
python run.py
```

This writes the predictions to `submission.csv`.

## Repository structure

| File | Description |
|---|---|
| `implementations.py` | The 6 required ML methods (GD, SGD, least squares, ridge, logistic, regularized logistic) |
| `helpers.py` | Data loading and submission helpers provided by the course |
| `run.py` | Full pipeline: preprocessing, training, prediction |
| `report/` | LaTeX source of the report |
