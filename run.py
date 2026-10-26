"""Reproduce our best AIcrowd submission: load data, preprocess, train, predict."""

import numpy as np

from helpers import load_csv_data, create_csv_submission
import implementations

DATA_PATH = "data"
SEED = 42


def main():
    np.random.seed(SEED)
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH)

    # Preprocessing, training and prediction go here.


if __name__ == "__main__":
    main()
