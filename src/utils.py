"""
Andrew McDonald
utils.py
Helper classes and functions for implementation of multifidelity learning+coverage algorithms.
"""

import copy
import sys
from datetime import datetime
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import constants
from gp import SFGP, MFGP

import six
sys.modules['sklearn.externals.six'] = six  # Workaround to import MLRose https://stackoverflow.com/a/62354885
import mlrose


#
# Begin class definitions.
#


class Experiment:
    """
    Object used to store experiment hyperparameters.
    """
    __slots__ = ["name", "algorithms", "fidelities", "n_agents", "n_simulations", "n_iterations", "n_prior_points",
                 "prior_fidelity", "noise", "alpha", "beta", "epoch_length_0", "gossip"]

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

        if not os.path.isdir("../logs"):
            os.mkdir("../logs")

    def save(self):
        """
        Saves experiment hyperparameters to a config file

        :param name: Name of experiment.
        """
        t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        filename = f"../logs/{self.name}_{t}.config"
        d = {key: getattr(self, key) for key in dir(self) if not key.startswith("__")}
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

        if not os.path.isdir("../logs"):
            os.mkdir("../logs")

    def log(self, name, sim, iteration, fidelity, positions, centroids, max_var, argmax_var,
            p_explore, explore, distance, loss, regret, mse):
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
                "regret": regret,
                "mse": mse
            }
            self.results.append(info)

    def save(self):
        """
        Save the results of a simulation

        :return: None. Saves data to CSV.
        """
        t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        df = pd.DataFrame(self.results)
        df.to_csv(f"../logs/{self.name}_{t}.csv")


class Plotter:
    """
    Object used to visualize algorithm progression.
    """
    __slots__ = ["name", "fig", "spec", "axes", "regret", "loss", "f_levels", "var_levels", "active"]

    def __init__(self, name):
        """
        Initializes plotter.

        :param name: Name of experiment.
        """
        # optional setting to disable plotter
        self.active = False

        # short-circuit if disabled
        if not self.active:
            return

        # configure plotter
        self.name = name
        self.regret, self.loss = [], []
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

        self.axes["loss"] = ax = self.fig.add_subplot(self.spec[2, 0:2])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
        ax.set_title("Loss")

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
        self.var_levels = np.linspace(0, constants.var_max, constants.levels)

    def set_lims_and_title(self, ax, title):
        """
        Helper function to set limits and title on axes.
        :param ax: Axes object.
        :param title: Text to display on title.
        :return: None. Modifies Axes object.
        """
        ax.set_title(title)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

    def plot(self, positions, data, partition, iteration, estimate, estimate_var, loss, regret,
             model=None, tsps0=None, tsps=None, save_prefix="default"):
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

        # plot samples and tsps atop partitions
        self.axes["samples"].scatter(x=data.x1, y=data.x2, c=partition, marker="s", s=100)
        if model:
            # plot points at which samples have been collected
            X_H = model.X_H if isinstance(model, MFGP) else model.X  # hifi points to scatter
            X_L = model.X_L if isinstance(model, MFGP) else np.empty((0, 2))  # lofi points to scatter
            self.axes["samples"].scatter(x=X_H[:, 0], y=X_H[:, 1], c="k", marker="^", s=50)
            self.axes["samples"].scatter(x=X_L[:, 0], y=X_L[:, 1], c="k", marker="v", s=5)
        if tsps0 and tsps:
            # plot tsp tours
            for i in range(positions.shape[0]):
                self.axes["samples"].plot(tsps0[i][:, 0], tsps0[i][:, 1], "-k")   # all tsp points
                self.axes["samples"].scatter(x=tsps[i][:, 0], y=tsps[i][:, 1], c="m", marker="^", s=50)   # remaining tsp points
                path = np.vstack((positions[i, :].reshape(-1, 2), tsps[i].reshape(-1, 2)))
                self.axes["samples"].plot(path[:, 0], path[:, 1], "-m") # path between current pos and remaining tsp points
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

        # plot loss
        self.loss.append(loss)
        self.axes["loss"].plot([i for i in range(len(self.loss))], self.loss, 'ko-')
        self.axes["loss"].ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
        self.axes["loss"].set_title("Loss")

        # show figure
        self.fig.savefig(f"uuraf/{self.name}_{save_prefix}_{iteration}.png")
        self.fig.show()

    def reset(self):
        """
        Reset plotter for new algorithm
        """
        # short-circuit if disabled
        if not self.active:
            return

        self.regret, self.loss = [], []
        self.axes["loss"].cla()
        self.axes["loss"].ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
        self.axes["loss"].set_title("Loss")


#
# Begin function definitions.
#


def initialize_positions(experiment, rng):

    # initialize array of positions
    positions = np.zeros((experiment.n_agents, 2))  # agent i position is in row i with [x, y]
    diffs = np.expand_dims(positions, 1) - np.expand_dims(positions, 0)
    distances = np.sum(diffs ** 2, axis=2)          # n_agents x n_agents distance matrix

    # generate random positions but ensure no two agents get assigned to same discretized point
    while np.amin(distances) < constants.dx / 2:    # there are two points on top of one another in discrete space
        positions = rng.random((experiment.n_agents, 2))
        multiplier = np.round(1 / constants.dx, decimals=4)
        positions = np.round(positions * multiplier) / multiplier
        diffs = np.expand_dims(positions, 1) - np.expand_dims(positions, 0)
        distances = np.sum(diffs ** 2, axis=2) + np.eye(experiment.n_agents)
        # add ones to diagonal since distance from agent to self is always 0

    print(f"Initial positions: {positions}")
    return positions

def initialize_gp(experiment, data, fidelity, rng):

    # select low-fidelity training data with n_prior_points
    X = np.hstack((data.x1.reshape(-1, 1), data.x2.reshape(-1, 1)))
    y_low = data.f_low.reshape(-1, 1)

    # random choice of low-fidelity data
    # idx = np.arange(X.shape[0])
    # train_idx = rng.choice(idx, size=experiment.n_prior_points, replace=False)  # random choice
    # X_train, y_low_train = X[train_idx, :], y_low[train_idx, :]

    # deterministic choice of low-fidelity data on interval-spaced grid of points
    n_points_1d = int(np.sqrt(experiment.n_prior_points) + constants.epsilon)
    interval = int(np.ceil(data.x1.shape[0] / n_points_1d))
    x1_train, x2_train = data.x1[::interval, ::interval].reshape(-1, 1), data.x2[::interval, ::interval].reshape(-1, 1)
    X_train = np.hstack((x1_train.reshape(-1, 1), x2_train.reshape(-1, 1)))
    y_low_train = data.f_low[::interval, ::interval].reshape(-1, 1)

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
                        dx1 = data.x1[i, j] - positions[agent, 0]
                        dx2 = data.x2[i, j] - positions[agent, 1]
                        dist = dx1 ** 2 + dx2 ** 2
                        if dist < min_dist:
                            assignments[i, j] = agent

    else:

        assignments = np.zeros(shape=data.x1.shape)
        for i in range(data.x1.shape[0]):
            for j in range(data.x1.shape[1]):
                # find agent closest to this point and update assignment accordingly
                min_dist = np.inf
                for agent in range(positions.shape[0]):
                    dx1 = data.x1[i, j] - positions[agent, 0]
                    dx2 = data.x2[i, j] - positions[agent, 1]
                    dist = dx1 ** 2 + dx2 ** 2
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
    for i in range(positions.shape[0]):
        # find points in agent i's cell
        x1_i = data.x1[partition == i]
        x2_i = data.x2[partition == i]
        f_i = data.f_high[partition == i]

        # compute weighted squared-distance loss to each point
        sq_dists = (x1_i - positions[i, 0]) ** 2 + (x2_i - positions[i, 1]) ** 2
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


def compute_mse(data, estimate):

    return np.mean((data.f_high - estimate)**2)


def compute_p_explore(max_var, max_var_0):

    return max_var / max_var_0


def compute_sampling_points(model, data, threshold):

    # create temporary model to query and leave original model unchanged
    query_model = copy.deepcopy(model)      # leave original model unchanged while determining sample points
    X_star = np.hstack((data.x1.reshape(-1, 1), data.x2.reshape(-1, 1)))

    # initialize return and prediction
    sampling_list = []
    mu_star, cov_star, var_star = query_model.predict(X_star)
    max_var = np.amax(var_star)

    # uncertainty reduction loop
    while max_var > threshold:

        print(f"Sample {len(sampling_list) + 1}, Max Var: {max_var}")

        # find argmax of variance and save point to sampling list
        idx = np.argmax(var_star)
        argmax_x, argmax_mu = X_star[idx], mu_star[idx]
        sampling_list.append(argmax_x)

        # update model with estimated mu at this x and re-predict
        model.update(argmax_x, argmax_mu)
        mu_star, cov_star, var_star = model.predict(X_star)
        max_var = np.amax(var_star)

    # convert list into nx2 array of points to sample
    sampling_array = np.array(sampling_list).reshape(-1, 2)
    return sampling_array


def compute_sampling_clusters(positions, data, partition, sampling_points):

    # initialize return list to store np array of sampling points for agent i at index i
    clusters = []

    for i in range(positions.shape[0]):
        # find points in agent i's cell
        x1_i = data.x1[partition == i]
        x2_i = data.x2[partition == i]

        # find sampling points which are in this cell and append
        idx_by_point = [np.logical_and(np.isclose(x1_i, sampling_points[j, 0]), np.isclose(x2_i, sampling_points[j, 1]))
                        for j in range(sampling_points.shape[0])]
        idx = np.logical_or.reduce(np.array(idx_by_point))      # take or across all points in cell
        cluster_i = np.hstack((x1_i[idx].reshape(-1, 1), x2_i[idx].reshape(-1, 1)))        # put into nx2 array
        clusters.append(cluster_i)

        # for j in range(sampling_points.shape[0]):
        #     # iterate over all sampling points and determine if this sampling point is in the partition of agent i
        #     dx1 = x1_i - sampling_points[j, 0]
        #     dx2 = x2_i - sampling_points[j, 1]
        #     dist = dx1 ** 2 + dx2 ** 2
        #
        #     if np.isclose(dist, 0):
        #         # assign this sampling point to agent i
        #         cluster_i.append(sampling_points[j, :])
        #
        # # convert cluster_i to array and append to clusters list
        # cluster_i_array = np.array(cluster_i)
        # clusters.append(cluster_i_array)

    return clusters


def compute_sampling_tsps(sampling_clusters):

    # initialize return list of np arrays of ordered sampling points by agent
    tours = []

    for cluster in sampling_clusters:

        # compute TSP tour using MLRose
        tour = np.empty((0, 2))
        if cluster.shape[0] > 0:                            # we have nontrivial set of points to sample
            coordinates = [tuple(x) for x in cluster]       # MLRose takes list of tuples, not ndarray
            problem = mlrose.TSPOpt(length=len(coordinates), coords=coordinates, maximize=False)
            solution, fitness = mlrose.genetic_alg(problem,
                                                   mutation_prob=constants.tsp_mutation,
                                                   max_attempts=constants.tsp_max_attempts,
                                                   random_state=constants.seed)
            tour = cluster[solution]    # solution is list of ordered indices
        tours.append(tour)

    return tours
