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
    __slots__ = ["name", "xx", "yy", "f_high", "f_low"]

    def __init__(self, name):
        """
        Loads experiment (input) data from CSVs

        :param name: Name of experiment.
        """
        self.name = name
        self.xx = np.loadtxt(f"../data/{name}_xx.csv", delimiter=",")
        self.yy = np.loadtxt(f"../data/{name}_yy.csv", delimiter=",")
        self.f_high = np.loadtxt(f"../data/{name}_hifi.csv", delimiter=",")
        self.f_low = np.loadtxt(f"../data/{name}_lofi.csv", delimiter=",")


class Log:
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

    def log(self, sim, iteration, positions, partition, centroids, distance, loss, regret):
        """
        Add an experimental result to the list of results.
        List of results stores dictionaries and will ultimately be converted to a DataFrame.

        :return: None.
        """
        result = []
        for agent in positions.shape[0]:
            info = {
                "sim": sim,
                "iteration": iteration,
                "x": positions[agent, 0],
                "y": positions[agent, 1],
                "centroid_x": centroids[agent, 0],
                "centroid_y": centroids[agent, 1],
                "distance": distance,
                "loss": loss,
                "regret": regret
            }
            result.append(info)

        self.results.append(result)

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
    __slots__ = ["name", "fig", "spec", "axes", "regret"]

    def __init__(self, name):
        """
        Initializes plotter.

        :param name: Name of experiment.
        """
        self.name = name
        self.regret = []
        self.fig = plt.figure(num=0, figsize=(16, 9), dpi=120, facecolor='w')
        self.spec = gridspec.GridSpec(ncols=4, nrows=3, figure=self.fig,
                                      width_ratios=[1, 1, 1, 1],  # full figures, colorbars
                                      height_ratios=[0.4, 1, 1])  # title, figures

        # store axes in a dict by keyword
        self.axes = {}
        self.axes["title"] = self.fig.add_subplot(self.spec[0, :])

        self.axes["partitions"] = ax = self.fig.add_subplot(self.spec[1, 0])
        ax.set_title("Partitions")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        self.axes["samples"] = ax = self.fig.add_subplot(self.spec[1, 1])
        ax.set_title("Samples")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        self.axes["mean"] = ax = self.fig.add_subplot(self.spec[1, 2])
        ax.set_title("Posterior Mean")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="5%", pad=0.1, pack_start=False)
        self.fig.add_axes(cax)
        self.axes["meancbar"] = cax

        self.axes["var"] = ax = self.fig.add_subplot(self.spec[1, 3])
        ax.set_title("Posterior Variance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="5%", pad=0.1, pack_start=False)
        self.fig.add_axes(cax)
        self.axes["varcbar"] = cax

        self.axes["regret"] = ax = self.fig.add_subplot(self.spec[2, 0:2])
        ax.set_title("Cumulative Regret")

        self.axes["lofi"] = ax = self.fig.add_subplot(self.spec[2, 2])
        ax.set_title("Low Fidelity Truth")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="5%", pad=0.1, pack_start=False)
        self.fig.add_axes(cax)
        self.axes["loficbar"] = cax

        self.axes["hifi"] = ax = self.fig.add_subplot(self.spec[2, 3])
        ax.set_title("High Fidelity Truth")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="5%", pad=0.1, pack_start=False)
        self.fig.add_axes(cax)
        self.axes["hificbar"] = cax

    def plot(self, positions, data, partition, estimate, estimate_var, regret):
        """
        Plot current simulation status
        """
        # plot partitions and label agents
        self.axes["partitions"].scatter(x=data.xx, y=data.yy, c=partition, marker="s", s=100)
        self.axes["partitions"].scatter(x=positions[:, 0], y=positions[:, 1], c="k", marker="o")
        for i in range(positions.shape[0]):
            self.axes["partitions"].text(x=positions[i, 0] + constants.dx,
                                         y=positions[i, 1] + constants.dx,
                                         s=str(i), fontsize=12)

        # plot samples
        self.axes["samples"].text(x=0.5, y=0.5, s="Need to implement", ha="center", va="center")

        # plot mean
        self.axes["mean"].contourf(x=data.xx, y=data.yy, z=estimate)

        # plot var
        self.axes["var"].contourf(x=data.xx, y=data.yy, z=estimate_var)

        # plot lofi
        self.axes["lofi"].contourf(x=data.xx, y=data.yy, z=data.f_low)

        # plot hifi
        self.axes["hifi"].contourf(x=data.xx, y=data.yy, z=data.f_high)

        # plot regret
        self.regret.append(regret)
        self.axes["regret"].plot([i for i in range(len(self.regret))], self.regret, 'ko-')
        self.axes["regret"].ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))


#
# Begin function definitions.
#


def compute_partition(positions, data, prev_partition, gossip):

    if gossip and prev_partition is not None:

        neighbors = []
        for i in range(positions.shape[0]):
            for j in range(i, positions.shape[0]):
                # check if agent i, j are neighbors by finding min dist between their assigned points
                xx_i = data.xx[prev_partition == i]
                yy_i = data.yy[prev_partition == i]
                xx_j = data.xx[prev_partition == j]
                yy_j = data.yy[prev_partition == j]
                dists = (xx_i - xx_j) ** 2 + (yy_i - yy_j) ** 2
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
                        dx = data.xx[i, j] - positions[agent, 0]
                        dy = data.yy[i, j] - positions[agent, 1]
                        dist = dx ** 2 + dy ** 2
                        if dist < min_dist:
                            assignments[i, j] = agent

    else:

        assignments = np.zeros(shape=data.xx.shape)
        for i in range(data.xx.shape[0]):
            for j in range(data.xx.shape[1]):
                # find agent closest to this point and update assignment accordingly
                min_dist = np.inf
                for agent in positions.shape[0]:
                    dx = data.xx[i, j] - positions[agent, 0]
                    dy = data.yy[i, j] - positions[agent, 1]
                    dist = dx ** 2 + dy ** 2
                    if dist < min_dist:
                        assignments[i, j] = agent

    return assignments


def compute_centroids(positions, data, partition, estimate):

    centroids = np.zeros(shape=positions)
    for i in range(positions.shape[0]):
        # find points in agent i's cell
        xx_i = data.xx[partition == i]
        yy_i = data.yy[partition == i]
        estimate_i = estimate[partition == i]
        area_i = len(estimate_i) * (constants.dx ** 2)

        # compute weighted points and integral
        total_weight = np.mean(estimate_i) * area_i
        weighted_xx_i = np.mean(xx_i * estimate_i) * area_i
        weighted_yy_i = np.mean(yy_i * estimate_i) * area_i
        centroid_xx = weighted_xx_i / total_weight
        centroid_yy = weighted_yy_i / total_weight
        centroid = np.concatenate((centroid_xx, centroid_yy)).reshape(1, 2)

        # snap centroid to nearest discretized point
        multiplier = np.round(1 / constants.dx, decimals=4)
        centroid = np.round(centroid * multiplier) / multiplier
        centroids[i, :] = centroid

    return centroids


def compute_distance(positions, prev_positions):
    return np.sqrt(np.sum((positions - prev_positions) ** 2, axis=1))


def compute_loss(positions, data, partition):

    loss = 0
    true_centroids = compute_centroids(positions, data, partition, estimate=data.f_high)
    for i in range(positions.shape[0]):
        # find points in agent i's cell
        xx_i = data.xx[partition == i]
        yy_i = data.yy[partition == i]
        f_i = data.f_high[partition == i]

        # compute weighted squared-distance loss to each point
        centroid = true_centroids[i, :]
        sq_dists = (xx_i - centroid[0]) ** 2 + (yy_i - centroid[1]) ** 2
        weighted_sq_dists = f_i * sq_dists
        area_i = len(xx_i) * (constants.dx ** 2)
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
