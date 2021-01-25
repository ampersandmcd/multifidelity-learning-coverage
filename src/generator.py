"""
Andrew McDonald
generator.py
A set of functions to generate test data and fit GP hyperparameters for multifidelity learning+coverage algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

import constants
from utils import Data
from gp import SFGP, MFGP


def fit_hyp(name):
    """
    Docstring.

    :param name:
    """
    # load data and construct random number generator
    data = Data(name)
    x1, x2, f_high, f_low = data.x1, data.x2, data.f_high, data.f_low
    rng = np.random.default_rng(seed=constants.seed)

    # convert x1, x2 meshgrid arrays into nx2 dimensional array and add noise to observed y
    X = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))
    y_high, y_low = f_high.reshape(-1, 1), f_low.reshape(-1, 1)
    y_high = y_high + rng.normal(scale=constants.sampling_noise, size=y_high.shape)
    y_low = y_low + rng.normal(scale=constants.sampling_noise, size=y_low.shape)

    # construct subset of data to train on
    n_training_points = 100
    idx = np.arange(X.shape[0])
    train_idx = rng.choice(idx, size=n_training_points, replace=False)
    X_train, y_high_train, y_low_train = X[train_idx, :], y_high[train_idx, :], y_low[train_idx, :]

    # # fit SFGP hyperparameters on a subset of data specified by train_idx
    sfgp = SFGP(X_train, y_high_train)
    print(sfgp)
    sfgp.train()
    print(sfgp)

    # save SFGP hyperparameters
    np.savetxt(f"../data/{name}_sfgp_hyp.csv", sfgp.hyp, delimiter=",")

    # fit MFGP hyperparameters on a subset of data specified by train_idx
    mfgp = MFGP(X_train, y_low_train, X_train, y_high_train)
    print(mfgp)
    mfgp.train()
    print(mfgp)

    # save MFGP hyperparameters
    np.savetxt(f"../data/{name}_mfgp_hyp.csv", mfgp.hyp, delimiter=",")


def visualize_fit_hyp(name):
    """
    Sanity check to ensure hyperparameters are meaningful.

    :param name:
    :return:
    """
    # load data and construct random number generator
    data = Data(name)
    x1, x2, f_high, f_low = data.x1, data.x2, data.f_high, data.f_low
    rng = np.random.default_rng(seed=constants.seed)

    # convert x1, x2 meshgrid arrays into nx2 dimensional array and add noise to observed y
    X = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))
    y_high, y_low = f_high.reshape(-1, 1), f_low.reshape(-1, 1)
    y_high = y_high + rng.normal(scale=constants.sampling_noise, size=y_high.shape)
    y_low = y_low + rng.normal(scale=constants.sampling_noise, size=y_low.shape)

    # construct (small) subset of data to train on
    n_training_points = 20
    idx = np.arange(X.shape[0])
    train_idx = rng.choice(idx, size=n_training_points, replace=False)
    X_train, y_high_train, y_low_train = X[train_idx, :], y_high[train_idx, :], y_low[train_idx, :]

    # construct models and init hyperparameters from pretrained set
    sfgp = SFGP(X_train, y_high_train)
    sfgp.hyp = np.loadtxt(f"../data/{name}_sfgp_hyp.csv", delimiter=",")
    sfgp.update()
    mfgp = MFGP(X_train, y_low_train, X_train, y_high_train)
    mfgp.hyp = np.loadtxt(f"../data/{name}_mfgp_hyp.csv", delimiter=",")
    mfgp.update()

    # get predictions at random set of points
    X_star = X
    x1_star, x2_star = x1, x2
    sf_mu, sf_var = sfgp.predict(X_star)
    mf_mu, mf_var = mfgp.predict(X_star)

    # configure plot
    fig = plt.figure(figsize=(12, 16))
    spec = gridspec.GridSpec(ncols=2, nrows=4, figure=fig, width_ratios=[1, 1], height_ratios=[0.2, 1, 1, 1])
    ax_text = fig.add_subplot(spec[0, :])
    ax_high = fig.add_subplot(spec[1, 0])
    ax_low = fig.add_subplot(spec[1, 1])
    ax_sf_mu = fig.add_subplot(spec[2, 0])
    ax_sf_var = fig.add_subplot(spec[2, 1])
    ax_mf_mu = fig.add_subplot(spec[3, 0])
    ax_mf_var = fig.add_subplot(spec[3, 1])
    f_levels = np.linspace(0, 1, constants.levels)
    var_levels = np.linspace(0, 0.05, constants.levels)

    # plot ground truth
    ax_text.text(x=0.5, y=0.5, s="Forrester 2007: GP Predictions", ha="center", va="center", fontsize=24)
    ax_text.axis("off")
    im = ax_high.contourf(x1, x2, f_high, levels=f_levels)
    plt.colorbar(im, ax=ax_high)
    im = ax_low.contourf(x1, x2, f_low, levels=f_levels)
    plt.colorbar(im, ax=ax_low)
    ax_high.set_title("High Fidelity")
    ax_low.set_title("Low Fidelity")

    # plot SFGP
    im = ax_sf_mu.contourf(x1_star, x2_star, sf_mu.reshape(x1_star.shape), levels=f_levels)
    plt.colorbar(im, ax=ax_sf_mu)
    im = ax_sf_var.contourf(x1_star, x2_star, np.diag(sf_var).reshape(x1_star.shape), levels=var_levels)
    plt.colorbar(im, ax=ax_sf_var)
    ax_sf_mu.set_title("SFGP Mean")
    ax_sf_var.set_title("SFGP Variance")

    # plot MFGP
    im = ax_mf_mu.contourf(x1_star, x2_star, mf_mu.reshape(x1_star.shape), levels=f_levels)
    plt.colorbar(im, ax=ax_mf_mu)
    im = ax_mf_var.contourf(x1_star, x2_star, np.diag(mf_var).reshape(x1_star.shape), levels=var_levels)
    plt.colorbar(im, ax=ax_mf_var)
    ax_mf_mu.set_title("MFGP Mean")
    ax_mf_var.set_title("MFGP Variance")

    # save & show plots
    fig.tight_layout()
    plt.savefig(f"../images/{name}_gp.png")
    plt.show()


def forrester2007(name, invert=False):
    """
    Generate 2D realization of function from Forrester 2007 "Multi-fidelity optimization via surrogate modeling"
    specified on page 3256. This function is defined in one dimension; we generalize it to 2 dimensions by
    multiplying two one-dimensional realizations.

    Since coverage is loosely-defined as maximization, we invert the function to its negative.

    :param name: String for use in file name
    :return: None. Saves data to CSV.
    """
    # generate grid
    x = np.arange(0, 1 + constants.dx/2, constants.dx)
    y = np.arange(0, 1 + constants.dx/2, constants.dx)
    x1, x2 = np.meshgrid(x, y, indexing="ij")

    # generate data
    f_high = ((6 * x1)**2 * np.sin(12 * x1 - 4)) + ((6 * x2)**2 * np.sin(12 * x2 - 4))
    a, b, c = 0.5, 10, -5
    f_low = (a * f_high + b * (x1 - 0.5) - c) + (a * f_high + b * (x2 - 0.5) - c)

    if invert:
        # invert data since the goal is maximization
        f_high, f_low = -f_high, -f_low

    # normalize data
    overall_min = min(np.amin(f_high), np.amin(f_low))
    overall_max = max(np.amax(f_high), np.amax(f_low))
    f_high = constants.f_max * (f_high - overall_min) / (overall_max - overall_min) + constants.div_by_zero_epsilon
    f_low = constants.f_max * (f_low - overall_min) / (overall_max - overall_min) + constants.div_by_zero_epsilon

    # configure plot
    fig = plt.figure(figsize=(12, 6))
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig, width_ratios=[1, 1, 0.05], height_ratios=[0.2, 1])
    ax_text = fig.add_subplot(spec[0, :])
    ax_high = fig.add_subplot(spec[1, 0])
    ax_low = fig.add_subplot(spec[1, 1])
    ax_colorbar = fig.add_subplot(spec[1, 2])
    f_levels = np.linspace(0, constants.f_max, constants.levels)


    # plot data
    ax_text.text(x=0.5, y=0.5, s="Forrester 2007", ha="center", va="center", fontsize=24)
    ax_text.axis("off")
    im = ax_high.contourf(x1, x2, f_high, levels=f_levels)
    im = ax_low.contourf(x1, x2, f_low, levels=f_levels)
    ax_high.set_title('High Fidelity')
    ax_low.set_title('Low Fidelity')
    plt.colorbar(im, cax=ax_colorbar)

    # save & show plots
    fig.tight_layout()
    plt.savefig(f"../images/{name}.png")
    plt.show()

    # save data
    np.savetxt(f"../data/{name}_hifi.csv", f_high, delimiter=",")
    np.savetxt(f"../data/{name}_lofi.csv", f_low, delimiter=",")
    np.savetxt(f"../data/{name}_x1.csv", x1, delimiter=",")
    np.savetxt(f"../data/{name}_x2.csv", x2, delimiter=",")


if __name__ == "__main__":
    name = "forrester2007"
    forrester2007(name)
