"""
Andrew McDonald
utils.py
Helper classes and functions for implementation of multifidelity learning+coverage algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import constants
from gp import SFGP, MFGP


#
# Begin class definitions.
#


class Experiment:
    """
    Object used to store experiment hyperparameters.
    """
    __slots__ = ["name", "algorithms", "fidelities", "n_agents", "n_simulations", "n_iterations", "n_prior_points",
                 "prior_fidelity", "noise", "todescato_constant", "dslc_alpha", "dslc_beta", "gossip"]

    def __init__(self, name):
        """
        Loads experiment hyperparameters from a config file

        :param name: Name of experiment.
        """
        filename = f"../data/{name}.config"
        with open(filename, mode="r") as file:
            text = file.read()
            kwargs = eval(text)

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def save(self, name):
        """
        Saves experiment hyperparameters to a config file

        :param name: Name of experiment.
        """
        filename = f"../data/{name}.config"
        d = {key: value for key, value in dir(self) if not key.startswith("__")}
        with open(filename, mode="w") as file:
            file.write(str(d))


class Data:
    """
    Object used to store experimental (input) data.
    """
    __slots__ = ["name", "x1", "x2", "f_high", "f_low"]

    def __init__(self, name):
        """
        Loads experiment (input) data from CSVs

        :param name: Name of experiment.
        """
        self.name = name
        self.x1 = np.loadtxt(f"../data/{name}_x1.csv", delimiter=",")
        self.x2 = np.loadtxt(f"../data/{name}_x2.csv", delimiter=",")
        self.f_high = np.loadtxt(f"../data/{name}_hifi.csv", delimiter=",")
        self.f_low = np.loadtxt(f"../data/{name}_lofi.csv", delimiter=",")


class Logger:
    """
    Object used to store experimental (output) results.
    """
    __slots__ = ["name", "results"]

    def __init__(self, name):
        """
        Initializes experimental log.

        :param name: Name of experiment.
        """
        self.name = name
        self.results = []

    def log(self, name, sim, iteration, fidelity, positions, centroids, max_var, argmax_var, p_explore, explore, distance, loss, regret):
        """
        Add an experimental result to the list of results.
        List of results stores dictionaries and will ultimately be converted to a DataFrame.

        :return: None.
        """
        for agent in range(positions.shape[0]):
            info = {
                "name": name,
                "fidelity": fidelity,
                "sim": sim,
                "iteration": iteration,
                "agent": agent,
                "x1": positions[agent, 0],
                "x2": positions[agent, 1],
                "centroid_x1": centroids[agent, 0],
                "centroid_x2": centroids[agent, 1],
                "max_var": max_var[agent],
                "argmax_var_x1": argmax_var[agent, 0],
                "argmax_var_x2": argmax_var[agent, 1],
                "p_explore": p_explore[agent],
                "explore": explore[agent],
                "distance": distance[agent],
                "loss": loss,
                "regret": regret
            }
            self.results.append(info)

    def save(self):
        """
        Save the results of a simulation

        :return: None. Saves data to CSV.
        """
        df = pd.DataFrame(self.results)
        df.to_csv(f"../logs/{self.name}.csv")


class Plotter:
    """
    Object used to visualize algorithm progression.
    """
    __slots__ = ["name", "fig", "spec", "axes", "regret", "f_levels", "var_levels", "active"]

    def __init__(self, name):
        """
        Initializes plotter.

        :param name: Name of experiment.
        """
        # optional setting to disable plotter
        self.active = True

        # short-circuit if disabled
        if not self.active:
            return

        # configure plotter
        self.name = name
        self.regret = []
        self.fig = plt.figure(num=0, figsize=(16, 9), dpi=120, facecolor='w')
        self.fig.subplots_adjust(wspace=0.5, hspace=0.5)
        self.spec = gridspec.GridSpec(ncols=4, nrows=3, figure=self.fig,
                                      width_ratios=[1, 1, 1, 1],  # full figures, colorbars
                                      height_ratios=[0.4, 1, 1])  # title, figures

        # store axes in a dict by keyword
        self.axes = {}
        self.axes["title"] = ax = self.fig.add_subplot(self.spec[0, :])
        ax.axis("off")

        self.axes["partitions"] = ax = self.fig.add_subplot(self.spec[1, 0])
        self.set_lims_and_title(ax, "Partitions")

        self.axes["samples"] = ax = self.fig.add_subplot(self.spec[1, 1])
        self.set_lims_and_title(ax, "Samples")

        self.axes["mean"] = ax = self.fig.add_subplot(self.spec[1, 2])
        self.set_lims_and_title(ax, "Posterior Mean")
        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="5%", pad=0.1, pack_start=False)
        self.fig.add_axes(cax)
        self.axes["meancbar"] = cax

        self.axes["var"] = ax = self.fig.add_subplot(self.spec[1, 3])
        self.set_lims_and_title(ax, "Posterior Variance")
        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="5%", pad=0.1, pack_start=False)
        self.fig.add_axes(cax)
        self.axes["varcbar"] = cax

        self.axes["regret"] = ax = self.fig.add_subplot(self.spec[2, 0:2])
        ax.set_title("Cumulative Regret")

        self.axes["lofi"] = ax = self.fig.add_subplot(self.spec[2, 3])
        self.set_lims_and_title(ax, "Low Fidelity Truth")

        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="5%", pad=0.1, pack_start=False)
        self.fig.add_axes(cax)
        self.axes["loficbar"] = cax

        self.axes["hifi"] = ax = self.fig.add_subplot(self.spec[2, 2])
        self.set_lims_and_title(ax, "High Fidelity Truth")
        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="5%", pad=0.1, pack_start=False)
        self.fig.add_axes(cax)
        self.axes["hificbar"] = cax

        # levels for contour plots
        self.f_levels = np.linspace(0, constants.f_max, constants.levels)
        self.var_levels = np.linspace(0, 0.05, constants.levels)

    def set_lims_and_title(self, ax, title):
        """
        Helper function to set limits and title on axes.
        :param ax: Axes object.
        :param title: Text to display on title.
        :return: None. Modifies Axes object.
        """
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def plot(self, positions, data, partition, estimate, estimate_var, regret):
        """
        Plot current simulation status
        """
        # short-circuit if disabled
        if not self.active:
            return

        # clear non-title axes
        for name, ax in self.axes.items():
            if name != "title":
                ax.cla()

        # plot partitions and label agents
        self.axes["partitions"].scatter(x=data.x1, y=data.x2, c=partition, marker="s", s=100)
        self.axes["partitions"].scatter(x=positions[:, 0], y=positions[:, 1], c="k", marker="o")
        for i in range(positions.shape[0]):
            self.axes["partitions"].text(x=positions[i, 0] + constants.dx,
                                         y=positions[i, 1] + constants.dx,
                                         s=str(i), fontsize=12)
        self.set_lims_and_title(self.axes["partitions"], "Partitions")


        # plot samples
        self.axes["samples"].text(x=0.5, y=0.5, s="Need to implement", ha="center", va="center")
        self.set_lims_and_title(self.axes["samples"], "Samples")

        # plot mean
        im = self.axes["mean"].contourf(data.x1, data.x2, estimate, levels=self.f_levels)
        plt.colorbar(im, cax=self.axes["meancbar"])
        self.set_lims_and_title(self.axes["mean"], "Posterior Mean")


        # plot var
        im = self.axes["var"].contourf(data.x1, data.x2, estimate_var, levels=self.var_levels)
        plt.colorbar(im, cax=self.axes["varcbar"])
        self.set_lims_and_title(self.axes["var"], "Posterior Variance")

        # plot lofi
        im = self.axes["lofi"].contourf(data.x1, data.x2, data.f_low, levels=self.f_levels)
        plt.colorbar(im, cax=self.axes["loficbar"])
        self.set_lims_and_title(self.axes["lofi"], "Low Fidelity Truth")

        # plot hifi
        im = self.axes["hifi"].contourf(data.x1, data.x2, data.f_high, levels=self.f_levels)
        plt.colorbar(im, cax=self.axes["hificbar"])
        self.set_lims_and_title(self.axes["hifi"], "High Fidelity Truth")

        # plot regret
        self.regret.append(regret)
        self.axes["regret"].plot([i for i in range(len(self.regret))], self.regret, 'ko-')
        self.axes["regret"].ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
        self.axes["regret"].set_title("Cumulative Regret")

        # show figure
        self.fig.show()

#
# Begin function definitions.
#


def initialize_gp(experiment, data, fidelity, rng):

    # select low-fidelity training data with n_prior_points
    X = np.hstack((data.x1.reshape(-1, 1), data.x2.reshape(-1, 1)))
    y_low = data.f_low.reshape(-1, 1)
    idx = np.arange(X.shape[0])
    train_idx = rng.choice(idx, size=experiment.n_prior_points, replace=False)
    X_train, y_low_train = X[train_idx, :], y_low[train_idx, :]

    # compute means for unit prior information
    X_mean, y_low_mean = np.mean(X_train, axis=0).reshape(1, 2), np.mean(y_low_train).reshape(1, 1)

    # select and initialize model according to fidelity
    if fidelity == "null_single":
        # assume unit prior information to avoid computational errors
        model = SFGP(X_mean, y_low_mean)
        model.hyp = np.loadtxt(f"../data/{experiment.name}_sfgp_hyp.csv", delimiter=",")
        model.update()
    elif fidelity == "single":
        # use prior information given by n_prior_points
        model = SFGP(X_train, y_low_train)
        model.hyp = np.loadtxt(f"../data/{experiment.name}_sfgp_hyp.csv", delimiter=",")
        model.update()
    elif fidelity == "multi":
        # use prior information given by n_prior_points for lofi and unit prior for hifi to avoid computational errors
        model = MFGP(X_train, y_low_train, X_mean, y_low_mean)
        model.hyp = np.loadtxt(f"../data/{experiment.name}_mfgp_hyp.csv", delimiter=",")
        model.update()
    else:
        raise ValueError("Unrecognized fidelity level chosen.")

    return model


def compute_partition(positions, data, prev_partition, gossip):

    if gossip and prev_partition is not None:

        neighbors = []
        for i in range(positions.shape[0]):
            for j in range(i, positions.shape[0]):
                # check if agent i, j are neighbors by finding min dist between their assigned points
                x1_i = data.x1[prev_partition == i]
                x2_i = data.x2[prev_partition == i]
                x1_j = data.x1[prev_partition == j]
                x2_j = data.x2[prev_partition == j]
                dists = (x1_i - x1_j) ** 2 + (x2_i - x2_j) ** 2
                if np.isclose(np.amin(dists), constants.dx):
                    # these agents are neighbors
                    neighbors.append((i, j))

        # select a pair of neighbors to update at random and update assignments ONLY between them
        random_edge = neighbors[np.random.randint(0, len(neighbors))]
        assignments = np.copy(prev_partition)
        for i in range(prev_partition.shape[0]):
            for j in range(prev_partition.shape[1]):
                if prev_partition[i, j] in random_edge:
                    # this point was previously assigned to one of the agents we are updating
                    # now, determine which agent this point is closer to
                    min_dist = np.inf
                    for agent in random_edge:
                        dx = data.x1[i, j] - positions[agent, 0]
                        dy = data.x2[i, j] - positions[agent, 1]
                        dist = dx ** 2 + dy ** 2
                        if dist < min_dist:
                            assignments[i, j] = agent

    else:

        assignments = np.zeros(shape=data.x1.shape)
        for i in range(data.x1.shape[0]):
            for j in range(data.x1.shape[1]):
                # find agent closest to this point and update assignment accordingly
                min_dist = np.inf
                for agent in range(positions.shape[0]):
                    dx = data.x1[i, j] - positions[agent, 0]
                    dy = data.x2[i, j] - positions[agent, 1]
                    dist = dx ** 2 + dy ** 2
                    if dist < min_dist:
                        min_dist = dist
                        assignments[i, j] = agent

    return assignments


def compute_centroids(positions, data, partition, estimate):

    centroids = np.zeros(shape=positions.shape)
    for i in range(positions.shape[0]):

        # find points in agent i's cell
        x1_i = data.x1[partition == i]
        x2_i = data.x2[partition == i]
        estimate_i = estimate[partition == i]
        area_i = len(estimate_i) * (constants.dx ** 2)

        # compute weighted points and integral
        total_weight = np.mean(estimate_i) * area_i
        weighted_x1_i = np.mean(x1_i * estimate_i) * area_i
        weighted_x2_i = np.mean(x2_i * estimate_i) * area_i
        centroid_x1 = weighted_x1_i / total_weight
        centroid_x2 = weighted_x2_i / total_weight
        centroid = np.array([centroid_x1, centroid_x2])

        # snap centroid to nearest discretized point
        multiplier = np.round(1 / constants.dx, decimals=4)
        centroid = np.round(centroid * multiplier) / multiplier
        centroids[i, :] = centroid

    return centroids


def compute_max_var(positions, data, partition, estimate_var):

    max_var, argmax_var = np.zeros(shape=positions.shape[0]), np.zeros(shape=positions.shape)
    for i in range(positions.shape[0]):

        # find points in agent i's cell
        x1_i = data.x1[partition == i]
        x2_i = data.x2[partition == i]
        estimate_var_i = estimate_var[partition == i]

        # compute max and argmax
        max_var_i = np.amax(estimate_var_i)
        idx = np.argmax(estimate_var_i)
        argmax_var_i = np.array([x1_i[idx], x2_i[idx]])

        # save max and argmax of this cell
        max_var[i], argmax_var[i, :] = max_var_i, argmax_var_i

    return max_var, argmax_var


def compute_distance(positions, prev_positions):
    return np.sqrt(np.sum((positions - prev_positions) ** 2, axis=1))


def compute_loss(positions, data, partition):

    loss = 0
    true_centroids = compute_centroids(positions, data, partition, estimate=data.f_high)
    for i in range(positions.shape[0]):
        # find points in agent i's cell
        x1_i = data.x1[partition == i]
        x2_i = data.x2[partition == i]
        f_i = data.f_high[partition == i]

        # compute weighted squared-distance loss to each point
        centroid = true_centroids[i, :]
        sq_dists = (x1_i - centroid[0]) ** 2 + (x2_i - centroid[1]) ** 2
        weighted_sq_dists = f_i * sq_dists
        area_i = len(x1_i) * (constants.dx ** 2)
        loss_i = np.mean(weighted_sq_dists) * area_i
        loss += loss_i

    return loss


def compute_regret(positions, data, partition):

    # compute loss of current configuration
    loss = compute_loss(positions, data, partition)

    # compute loss with fixed partition BUT new positions
    new_positions = compute_centroids(positions, data, partition, estimate=data.f_high)
    position_loss = compute_loss(new_positions, data, partition)

    # compute loss with fixed positions BUT new partition
    new_partition = compute_partition(positions, data, prev_partition=None, gossip=False)
    partition_loss = compute_loss(positions, data, new_partition)

    # compute regret following definition 2 in DSLC paper https://arxiv.org/abs/2101.04306
    regret = 2 * loss - position_loss - partition_loss
    return regret


def compute_todescato_explore(max_var, max_var_0):

    return np.sqrt(max_var / max_var_0)
