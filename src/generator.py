"""
Andrew McDonald
generator.py
A set of functions to generate test data for multifidelity learning+coverage algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import constants


def forrester2007():
    """
    Generate 2D realization of function from Forrester 2007 "Multi-fidelity optimization via surrogate modeling"
    specified on page 3256. This function is defined in one dimension; we generalize it to 2 dimensions by
    multiplying two one-dimensional realizations.

    Since coverage is loosely-defined as maximization, we invert the function to its negative.

    :return: None. Saves data to CSV.
    """
    name = "forrester2007"

    # generate grid
    x = np.arange(0, 1, constants.dx)
    y = np.arange(0, 1, constants.dx)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # generate data
    f_high = ((6 * xx)**2 * np.sin(12 * xx - 4)) + ((6 * yy)**2 * np.sin(12 * yy - 4))
    a, b, c = 0.5, 10, -5
    f_low = (a * f_high + b * (xx - 0.5) - c) + (a * f_high + b * (yy - 0.5) - c)

    # normalize data
    overall_min = min(np.amin(f_high), np.amin(f_low))
    overall_max = max(np.amax(f_high), np.amax(f_low))
    f_high = (f_high - overall_min) / (overall_max - overall_min) + constants.div_by_zero_epsilon
    f_low = (f_low - overall_min) / (overall_max - overall_min) + constants.div_by_zero_epsilon

    # configure plot
    fig = plt.figure(figsize=(12, 6))
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig, width_ratios=[1, 1, 0.05], height_ratios=[0.2, 1])
    ax_text = fig.add_subplot(spec[0, :])
    ax_high = fig.add_subplot(spec[1, 0])
    ax_low = fig.add_subplot(spec[1, 1])
    ax_colorbar = fig.add_subplot(spec[1, 2])

    # plot data
    ax_text.text(x=0.5, y=0.5, s="Forrester 2007", ha="center", va="center", fontsize=24)
    ax_text.axis("off")
    im = ax_high.contourf(xx, yy, f_high)
    im = ax_low.contourf(xx, yy, f_low)
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
    np.savetxt(f"../data/{name}_xx.csv", xx, delimiter=",")
    np.savetxt(f"../data/{name}_yy.csv", yy, delimiter=",")


if __name__ == "__main__":
    forrester2007()
