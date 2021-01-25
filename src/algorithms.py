"""
Andrew McDonald
algorithms.py
Implementations of multifidelity learning+coverage algorithms.
"""

import numpy as np
import utils
import constants


def cortes(experiment, data, logger, plotter, fidelity, sim):

    # init random number generator
    rng = np.random.default_rng()

    # initialize random starting positions and snap to nearest discretized point
    positions = rng.random((experiment.n_agents, 2))      # agent i position is in row i with [x, y]
    multiplier = np.round(1 / constants.dx, decimals=4)
    positions = np.round(positions * multiplier) / multiplier
    prev_positions = np.copy(positions)                   # save copy to compute distance travelled

    # initialize partition and true centroids
    partition = utils.compute_partition(positions, data, None, experiment.gossip)
    centroids = utils.compute_centroids(positions, data, partition, estimate=data.f_high)

    # initialize maximum posterior variance terms and set to zero, given knowledge of phi in this algorithm
    max_var = np.zeros(experiment.n_agents)             # array of max variance in each cell
    argmax_var = np.zeros((experiment.n_agents, 2))      # nx2 array of points of max variance

    # initialize probability and decision of exploration set to zero, given knowledge of phi in this algorithm
    p_explore = np.zeros(experiment.n_agents)
    explore = np.zeros(experiment.n_agents)

    for iteration in range(experiment.n_iterations):

        # compute distance travelled, loss and regret
        distance = utils.compute_distance(positions, prev_positions)
        loss = utils.compute_loss(positions, data, partition)
        regret = utils.compute_regret(positions, data, partition)

        # log and plot progress
        logger.log("cortes", sim, iteration, fidelity, positions, centroids,
                   max_var, argmax_var, p_explore, explore, distance, loss, regret)
        plotter.plot(positions, data, partition, estimate=data.f_high, estimate_var=np.zeros(data.x1.shape), regret=regret)

        # update partition and centroids given perfect knowledge
        partition = utils.compute_partition(centroids, data, partition, experiment.gossip)
        centroids = utils.compute_centroids(positions, data, partition, estimate=data.f_high)

        # perform Lloyd update
        prev_positions = np.copy(positions)
        positions = np.copy(centroids)


def todescato(experiment, data, logger, plotter, fidelity, sim):

    # init random number generator
    rng = np.random.default_rng()

    # initialize GP model using low-fidelity training data
    model = utils.initialize_gp(experiment, data, fidelity, rng)
    X_star = np.hstack((data.x1.reshape(-1, 1), data.x2.reshape(-1, 1)))
    mu_star, cov_star, var_star = model.predict(X_star)

    # initialize random starting positions and snap to nearest discretized point
    positions = rng.random((experiment.n_agents, 2))      # agent i position is in row i with [x, y]
    multiplier = np.round(1 / constants.dx, decimals=4)
    positions = np.round(positions * multiplier) / multiplier
    prev_positions = np.copy(positions)                   # save copy to compute distance travelled

    # initialize partition and estimated centroids
    partition = utils.compute_partition(positions, data, None, experiment.gossip)
    centroids = utils.compute_centroids(positions, data, partition, estimate=mu_star.reshape(data.x1.shape))

    # initialize maximum posterior variance terms and save initial max posterior variance
    max_var, argmax_var = utils.compute_max_var(positions, data, partition, estimate_var=var_star.reshape(data.x1.shape))
    max_var_0 = np.amax(max_var)

    # initialize probability and decision of exploration
    p_explore = utils.compute_todescato_explore(max_var, max_var_0)
    explore = (rng.random(experiment.n_agents) < p_explore) * 1

    for iteration in range(experiment.n_iterations):

        # agents sample environment if on explore step
        sampling_agents = np.where(explore == 1)[0]
        sampling_X = positions[sampling_agents]
        sampling_idx = [np.logical_and(np.isclose(data.x1, sampling_X[i, 0]), np.isclose(data.x2, sampling_X[i, 1]))
                        for i in range(sampling_agents.shape[0])]                     # compute indices at which to sample
        sampling_y = np.array([data.f_high[idx] for idx in sampling_idx]).flatten()   # take noise-free samples at these indices
        sampling_y = sampling_y + rng.normal(loc=0, scale=constants.sampling_noise, size=sampling_y.shape)

        # update model with new samples (if any) and recompute
        model.update(sampling_X, sampling_y)
        mu_star, cov_star, var_star = model.predict(X_star)

        # compute distance travelled, loss and regret
        distance = utils.compute_distance(positions, prev_positions)
        loss = utils.compute_loss(positions, data, partition)
        regret = utils.compute_regret(positions, data, partition)

        # log and plot progress
        logger.log("todescato", sim, iteration, fidelity, positions, centroids,
                   max_var, argmax_var, p_explore, explore, distance, loss, regret)
        plotter.plot(positions, data, partition, estimate=mu_star.reshape(data.x1.shape),
                     estimate_var=var_star.reshape(data.x1.shape), regret=regret)

        # update partition and centroids based on estimate and previous centroids
        partition = utils.compute_partition(centroids, data, partition, experiment.gossip)
        centroids = utils.compute_centroids(positions, data, partition, estimate=mu_star.reshape(data.x1.shape))

        # find new points of maximum posterior variance
        max_var, argmax_var = utils.compute_max_var(positions, data, partition, estimate_var=var_star.reshape(data.x1.shape))

        # compute probability and decision of exploration
        p_explore = utils.compute_todescato_explore(max_var, max_var_0)
        explore = (rng.random(experiment.n_agents) < p_explore) * 1

        # update positions based on decision of exploration
        prev_positions = np.copy(positions)
        for agent in range(experiment.n_agents):
            if explore[agent] == 1:     # explore -> go to max var
                positions[agent, :] = argmax_var[agent, :]
            else:                       # exploit -> go to centroid
                positions[agent, :] = centroids[agent, :]



def dslc(experiment, data, logger, plotter, fidelity, sim):
    pass