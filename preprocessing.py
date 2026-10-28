"""Data cleaning and feature engineering for the BRFSS heart disease dataset.

The pipeline is fitted on the training set only (column choices, medians,
means, categories...) and then applied identically to the test set:

    names = load_feature_names("data")
    params = fit_preprocessing(x_train, names)
    X_train = transform(x_train, params)
    X_test = transform(x_test, params)
"""

import os

import numpy as np

# fmt: off
# Survey metadata (dates, ids, phone/interview details, sampling weights).
# They describe how the interview was done, not the person, so we drop them.
METADATA_COLUMNS = [
    "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE",
    "SEQNO", "_PSU", "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES",
    "CELLFON3", "LADULT", "CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2",
    "CCLGHOUS", "CSTATE", "LANDLINE", "QSTVER", "QSTLANG", "_STSTR",
    "_STRWT", "_RAWRAKE", "_WT2RAKE", "_DUALUSE", "_DUALCOR", "_LLCPWT",
    "_CLLCPWT",
]

# Raw columns that have a cleaner computed version in the dataset
# (e.g. WEIGHT2/HEIGHT3 mix pounds/kg and feet/cm, WTKG3/HTIN4/_BMI5 do not).
REDUNDANT_COLUMNS = ["WEIGHT2", "HEIGHT3", "HTM4"]

# BRFSS answers "1xx / 2xx / 3xx" mean "xx times per day / week / month".
# These columns are converted to a single "times per day" value.
PER_PERIOD_COLUMNS = [
    "ALCDAY5", "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG",
    "VEGETAB1", "EXEROFT1", "EXEROFT2", "STRENGTH",
]
# fmt: on


def load_feature_names(data_path):
    """Read the feature names from the header of x_train.csv (without Id)."""
    with open(os.path.join(data_path, "x_train.csv")) as f:
        return f.readline().strip().split(",")[1:]


def per_period_to_per_day(col):
    """Convert BRFSS 1xx/2xx/3xx frequency codes to occurrences per day.

    101-199: per day, 201-299: per week, 301-399: per month,
    300 (less than once a month) and 555/888 (never) map to 0,
    777/999 (don't know / refused) map to NaN.
    """
    out = np.full(col.shape, np.nan)
    hundreds = col // 100
    count = col % 100
    out[hundreds == 1] = count[hundreds == 1]
    out[hundreds == 2] = count[hundreds == 2] / 7
    out[hundreds == 3] = count[hundreds == 3] / 30
    out[np.isin(col, [300, 555, 888])] = 0
    return out


def clean_special_codes(col):
    """Replace BRFSS special answer codes in a numeric column.

    The codes depend on the number of digits of the answers: for a column
    whose largest code is 99 (resp. 999, 9999...), 77 and 99 mean
    "don't know" / "refused" (set to NaN) and 88 means "none" (set to 0).
    """
    col = col.copy()
    valid = col[~np.isnan(col)]
    if len(valid) == 0:
        return col
    for n_digits in range(2, 7):
        nines = 10**n_digits - 1
        if valid.max() == nines:
            sevens = nines // 9 * 7
            eights = nines // 9 * 8
            col[(col == nines) | (col == sevens)] = np.nan
            col[col == eights] = 0
            break
    return col


def fit_preprocessing(x, names, max_nan_fraction=0.9, max_categories=10):
    """Learn all preprocessing parameters from the training data.

    Args:
        x: shape (N, D), raw training features.
        names: list of the D feature names.
        max_nan_fraction: columns with more missing values are dropped.
        max_categories: columns with at most this many distinct values are
            treated as categorical (one-hot encoded), the others as continuous.

    Returns:
        params: dict with everything transform() needs.
    """
    params = {"names": list(names), "categorical": [], "continuous": []}
    nan_fraction = np.isnan(x).mean(axis=0)

    for j, name in enumerate(names):
        if name in METADATA_COLUMNS or name in REDUNDANT_COLUMNS:
            continue
        if nan_fraction[j] > max_nan_fraction:
            continue

        values = np.unique(x[~np.isnan(x[:, j]), j])
        if len(values) <= 1:
            continue

        if len(values) <= max_categories and name not in PER_PERIOD_COLUMNS:
            # Categorical: one indicator per answer, plus one for "missing".
            # Special codes (7 = don't know, 9 = refused) simply become their
            # own category, so no information is lost.
            has_nan = nan_fraction[j] > 0
            params["categorical"].append((j, values, has_nan))
        else:
            col = clean_continuous(x[:, j], name)
            # Clip extreme values to the 1st-99th percentiles, then
            # standardize. Missing values are filled with the median.
            low, high = np.nanpercentile(col, [1, 99])
            col = np.clip(col, low, high)
            median = np.nanmedian(col)
            col = np.where(np.isnan(col), median, col)
            mean, std = col.mean(), col.std()
            params["continuous"].append(
                (j, low, high, median, mean, std if std > 0 else 1.0)
            )
    return params


def clean_continuous(col, name):
    """Apply the column-specific cleaning of a continuous column."""
    if name in PER_PERIOD_COLUMNS:
        return per_period_to_per_day(col)
    return clean_special_codes(col)


def transform(x, params):
    """Apply the fitted preprocessing to raw features.

    Args:
        x: shape (N, D), raw features (train or test).
        params: output of fit_preprocessing().

    Returns:
        X: shape (N, K), clean feature matrix with a leading column of ones
            (bias term).
    """
    names = params["names"]
    features = [np.ones(len(x))]

    for j, low, high, median, mean, std in params["continuous"]:
        col = clean_continuous(x[:, j], names[j])
        col = np.clip(col, low, high)
        col = np.where(np.isnan(col), median, col)
        features.append((col - mean) / std)

    for j, values, has_nan in params["categorical"]:
        col = x[:, j]
        for v in values:
            features.append((col == v).astype(float))
        if has_nan:
            features.append(np.isnan(col).astype(float))

    return np.column_stack(features)
