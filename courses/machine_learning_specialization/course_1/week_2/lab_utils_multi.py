"""
lab_utils_multi.py
    Utility functions for multi-variable linear regression
"""

import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import compute_cost, compute_gradient

np.random.seed(1)
plt.style.use('./deeplearning.mplstyle')


def load_house_data():
    """
    Loads a dataset with 4 features for predicting house prices.
    Returns:
      X (ndarray): Shape (m, 4) with m examples and 4 features
      y (ndarray): Shape (m,) with target prices
    """
    data = np.array([
        [2104, 5, 1, 45, 460],
        [1416, 3, 2, 40, 232],
        [852, 2, 1, 35, 178],
        [1534, 3, 2, 38, 308],
        [3200, 4, 3, 8, 540],
        [1636, 3, 2, 41, 322],
        [1804, 2, 2, 53, 308],
        [1962, 4, 2, 42, 430],
        [3890, 3, 2, 54, 520],
        [1100, 3, 1, 46, 270],
        [1458, 3, 2, 37, 324],
        [2526, 3, 2, 38, 388],
        [2200, 3, 2, 39, 368],
        [2637, 3, 2, 36, 410],
        [1839, 2, 2, 30, 285],
        [1000, 1, 1, 49, 225],
        [2040, 4, 2, 45, 384],
        [3137, 3, 2, 52, 465],
        [1811, 4, 2, 53, 340],
        [1437, 3, 2, 57, 226],
        [1239, 3, 2, 48, 255],
        [2132, 4, 2, 42, 372],
        [4215, 5, 2, 34, 670],
        [2162, 4, 2, 35, 359],
        [1664, 2, 2, 41, 271],
        [2238, 3, 2, 40, 368],
        [2567, 4, 3, 39, 470],
        [1200, 2, 1, 55, 264],
        [852, 2, 1, 35, 178],
        [1852, 4, 2, 31, 380],
    ])

    X = data[:, :4]
    y = data[:, 4]

    return X, y


def run_gradient_descent(X, y, iterations, alpha, compute_cost=compute_cost,
                        compute_gradient=compute_gradient):
    """
    Run gradient descent and return final parameters, cost history, and gradient history.

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)): target values
      iterations (int): number of iterations to run
      alpha (float): learning rate
      compute_cost: function to compute cost
      compute_gradient: function to compute gradient

    Returns:
      w (ndarray (n,)): final parameters
      b (scalar): final bias
      hist (dict): history with 'cost', 'dj_dw', 'dj_db', 'w' for each iteration
    """
    import copy
    import math

    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    hist = {
        'cost': [],
        'dj_dw': [],
        'dj_db': [],
        'w': []
    }

    for i in range(iterations):
        # Compute gradients
        dj_db, dj_dw = compute_gradient(X, y, w, b)

        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Store history
        if i < 100000:  # Prevent memory exhaustion
            cost = compute_cost(X, y, w, b)
            hist['cost'].append(cost)
            hist['dj_dw'].append(dj_dw)
            hist['dj_db'].append(dj_db)
            hist['w'].append(w)

        # Print progress
        if i % math.ceil(iterations / 10) == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Iteration {i:4d}: Cost {cost:8.2f}")

    return w, b, hist


def norm_plot(ax, data):
    """
    Plot a histogram of data.

    Args:
      ax: matplotlib axis object
      data (ndarray): data to plot
    """
    ax.hist(data, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_ylabel("count")


def plt_equal_scale(X_train, X_norm, y_train):
    """
    Plot cost contours for unnormalized and normalized data to show the benefit
    of feature scaling.

    Args:
      X_train (ndarray): unnormalized training data
      X_norm (ndarray): normalized training data
      y_train (ndarray): target values
    """
    from lab_utils_common import compute_cost

    # Create figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Unnormalized contour plot
    w_range = np.linspace(-1, 8, 50)
    b_range = np.linspace(-1, 10, 50)
    W, B = np.meshgrid(w_range, b_range)
    Z_unnorm = np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            # For simplicity, compute cost using first two features
            w_test = np.array([W[i, j], B[i, j], 0, 0])
            b_test = 0
            Z_unnorm[i, j] = compute_cost(X_train, y_train, w_test, b_test)

    ax[0].contour(W, B, Z_unnorm, levels=20)
    ax[0].set_xlabel("w[0]")
    ax[0].set_ylabel("w[1]")
    ax[0].set_title("Unnormalized: Cost vs w[0], w[1]")
    ax[0].grid(True)

    # Normalized contour plot
    Z_norm = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            # For simplicity, compute cost using first two features
            w_test = np.array([W[i, j], B[i, j], 0, 0])
            b_test = 0
            Z_norm[i, j] = compute_cost(X_norm, y_train, w_test, b_test)

    ax[1].contour(W, B, Z_norm, levels=20)
    ax[1].set_xlabel("w[0]")
    ax[1].set_ylabel("w[1]")
    ax[1].set_title("Normalized: Cost vs w[0], w[1]")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_cost_i_w(X, y, hist):
    """
    Plot cost history and a parameter's history over iterations.

    Args:
      X (ndarray): training data
      y (ndarray): target values
      hist (dict): history dictionary from run_gradient_descent with 'cost', 'w', 'dj_dw' keys
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot cost
    ax1.plot(hist['cost'])
    ax1.set_ylabel("Cost")
    ax1.set_xlabel("Iteration")
    ax1.set_title("Cost vs. iteration")

    # Plot w[0] evolution
    w_0_values = [w[0] for w in hist['w']]
    ax2.plot(w_0_values, label='w[0]')
    ax2.set_ylabel("w[0]")
    ax2.set_xlabel("Iteration")
    ax2.set_title("Parameter w[0] vs. iteration")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()
