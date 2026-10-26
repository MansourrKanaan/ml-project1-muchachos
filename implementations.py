"""Implementations of the 6 basic ML methods required for project 1."""

import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: shape (N,), labels.
        tx: shape (N, D), features.
        initial_w: shape (D,), initial weights.
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: shape (D,), last weight vector.
        loss: scalar, MSE loss (with factor 1/2) of the last w.
    """
    w = initial_w
    for _ in range(max_iters):
        e = y - tx @ w
        gradient = -tx.T @ e / len(y)
        w = w - gamma * gradient
    e = y - tx @ w
    return w, np.mean(e**2) / 2


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1).

    Args:
        y: shape (N,), labels.
        tx: shape (N, D), features.
        initial_w: shape (D,), initial weights.
        max_iters: number of SGD steps.
        gamma: step size.

    Returns:
        w: shape (D,), last weight vector.
        loss: scalar, MSE loss (with factor 1/2) of the last w.
    """
    w = initial_w
    for _ in range(max_iters):
        n = np.random.randint(len(y))
        e = y[n] - tx[n] @ w
        gradient = -tx[n].T * e
        w = w - gamma * gradient
    e = y - tx @ w
    return w, np.mean(e**2) / 2


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Args:
        y: shape (N,), labels.
        tx: shape (N, D), features.

    Returns:
        w: shape (D,), optimal weights.
        loss: scalar, MSE loss (with factor 1/2).
    """
    # Gradient = 0 at the minimum gives the normal equations: tx^T tx w = tx^T y
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    e = y - tx @ w
    return w, np.mean(e**2) / 2


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Args:
        y: shape (N,), labels.
        tx: shape (N, D), features.
        lambda_: regularization parameter.

    Returns:
        w: shape (D,), optimal weights.
        loss: scalar, MSE loss (with factor 1/2), without the penalty term.
    """
    # Same as least squares, plus lambda' * I with lambda' = 2 * N * lambda_
    N, D = tx.shape
    A = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    e = y - tx @ w
    return w, np.mean(e**2) / 2


def sigmoid(t):
    """Squash any real number into a probability in (0, 1)."""
    return 1 / (1 + np.exp(-t))


def compute_logistic_loss(y, tx, w):
    """Average negative log-likelihood for labels y in {0, 1}.

    Uses log(1 + exp(z)) - y * z, which equals
    -[y * log(p) + (1 - y) * log(1 - p)] with p = sigmoid(z).
    """
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Gradient of the logistic loss: tx^T (sigmoid(tx w) - y) / N."""
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Args:
        y: shape (N,), labels in {0, 1}.
        tx: shape (N, D), features.
        initial_w: shape (D,), initial weights.
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: shape (D,), last weight vector.
        loss: scalar, negative log-likelihood (averaged) of the last w.
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w)
        w = w - gamma * gradient
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    The penalty term is lambda_ * ||w||^2 (y in {0, 1}).

    Args:
        y: shape (N,), labels in {0, 1}.
        tx: shape (N, D), features.
        lambda_: regularization parameter.
        initial_w: shape (D,), initial weights.
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: shape (D,), last weight vector.
        loss: scalar, negative log-likelihood (averaged) of the last w,
            without the penalty term.
    """
    w = initial_w
    for _ in range(max_iters):
        # The penalty lambda_ * ||w||^2 adds 2 * lambda_ * w to the gradient
        gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    return w, compute_logistic_loss(y, tx, w)
