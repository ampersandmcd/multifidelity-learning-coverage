"""
Andrew McDonald
algorithms.py
Implementations of multifidelity learning+coverage algorithms.
"""

import copy
import numpy as np
import utils
import constants


def cortes(experiment, data, logger, plotter, fidelity, sim):

    # init random number generator and reset plots
    rng = np.random.default_rng()
    plotter.reset()

    # initialize random starting positions and snap to nearest discretized point
    positions = utils.initialize_positions(experiment, rng)
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
        print(f"Iteration: {iteration}, Loss: {loss}") if iteration % 10 == 0 else None

        # log and plot progress
        logger.log("cortes", sim, iteration, fidelity, positions, centroids,
                   max_var, argmax_var, p_explore, explore, distance, loss, regret)
        plotter.plot(positions, data, partition, estimate=data.f_high, estimate_var=np.zeros(data.x1.shape),
                     regret=regret, loss=loss)

        # update partition and centroids given perfect knowledge
        partition = utils.compute_partition(centroids, data, partition, experiment.gossip)
        centroids = utils.compute_centroids(positions, data, partition, estimate=data.f_high)

        # perform Lloyd update
        prev_positions = np.copy(positions)
        positions = np.copy(centroids)


def stochastic_multifidelity_learning_coverage(experiment, data, logger, plotter, fidelity, sim):

    # init random number generator and reset plots
    rng = np.random.default_rng()
    plotter.reset()

    # initialize GP model using low-fidelity training data
    model = utils.initialize_gp(experiment, data, fidelity, rng)
    X_star = np.hstack((data.x1.reshape(-1, 1), data.x2.reshape(-1, 1)))
    mu_star, cov_star, var_star = model.predict(X_star)

    # initialize random starting positions and snap to nearest discretized point
    positions = utils.initialize_positions(experiment, rng)
    prev_positions = np.copy(positions)                   # save copy to compute distance travelled

    # initialize partition and estimated centroids
    partition = utils.compute_partition(positions, data, None, experiment.gossip)
    centroids = utils.compute_centroids(positions, data, partition, estimate=mu_star.reshape(data.x1.shape))

    # initialize maximum posterior variance terms and save initial max posterior variance
    max_var, argmax_var = utils.compute_max_var(positions, data, partition, estimate_var=var_star.reshape(data.x1.shape))
    max_var_0 = np.amax(max_var)

    # initialize probability and decision of exploration
    p_explore = utils.compute_p_explore(max_var, max_var_0)
    explore = (rng.random(experiment.n_agents) < p_explore)

    for iteration in range(experiment.n_iterations):

        # agents sample environment if on explore step
        sampling_agents = np.where(explore)[0]
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
        print(f"Iteration: {iteration}, Loss: {loss}") if iteration % 10 == 0 else None

        # log and plot progress
        logger.log("smlc", sim, iteration, fidelity, positions, centroids,
                   max_var, argmax_var, p_explore, explore, distance, loss, regret)
        plotter.plot(positions, data, partition, estimate=mu_star.reshape(data.x1.shape),
                     estimate_var=var_star.reshape(data.x1.shape), regret=regret, loss=loss, model=model)

        # update partition and centroids based on estimate and previous centroids
        partition = utils.compute_partition(centroids, data, partition, experiment.gossip)
        centroids = utils.compute_centroids(positions, data, partition, estimate=mu_star.reshape(data.x1.shape))

        # find new points of maximum posterior variance
        max_var, argmax_var = utils.compute_max_var(positions, data, partition, estimate_var=var_star.reshape(data.x1.shape))

        # compute probability and decision of exploration
        p_explore = utils.compute_p_explore(max_var, max_var_0)
        explore = (rng.random(experiment.n_agents) < p_explore)

        # update positions based on decision of exploration
        prev_positions = np.copy(positions)
        for agent in range(experiment.n_agents):
            if explore[agent]:     # explore -> go to max var
                positions[agent, :] = argmax_var[agent, :]
            else:                  # exploit -> go to centroid
                positions[agent, :] = centroids[agent, :]


def deterministic_multifidelity_learning_coverage(experiment, data, logger, plotter, fidelity, sim):

    # init random number generator and reset plots
    rng = np.random.default_rng()
    plotter.reset()

    # initialize GP model using low-fidelity training data
    model = utils.initialize_gp(experiment, data, fidelity, rng)
    X_star = np.hstack((data.x1.reshape(-1, 1), data.x2.reshape(-1, 1)))
    mu_star, cov_star, var_star = model.predict(X_star)

    # initialize random starting positions and snap to nearest discretized point
    positions = utils.initialize_positions(experiment, rng)
    prev_positions = np.copy(positions)                   # save copy to compute distance travelled

    # initialize partition and estimated centroids
    partition = utils.compute_partition(positions, data, None, experiment.gossip)
    centroids = utils.compute_centroids(positions, data, partition, estimate=mu_star.reshape(data.x1.shape))

    # initialize maximum posterior variance terms and save initial max posterior variance
    max_var, argmax_var = utils.compute_max_var(positions, data, partition, estimate_var=var_star.reshape(data.x1.shape))
    max_var_0 = np.amax(max_var)

    # initialize iteration, epoch counter
    iteration, epoch = 0, 0

    while iteration < experiment.n_iterations:      # add 1 when comparing since we count from 0

        # compute threshold below which to reduce posterior variance, update epoch length
        threshold = (experiment.alpha ** (epoch + 1)) * max_var_0       # need to reduce one epoch ahead
        epoch_length = int((experiment.beta ** epoch) * experiment.epoch_length_0 + constants.epsilon)

        # compute sampling points, clusters and TSP tours to reduce posterior variance
        sampling_points = utils.compute_sampling_points(model, data, threshold)
        sampling_clusters = utils.compute_sampling_clusters(positions, data, partition, sampling_points)
        sampling_tsps = utils.compute_sampling_tsps(sampling_clusters)
        sampling_tsps_0 = copy.deepcopy(sampling_tsps)                              # copy TSP for plotting
        explore = np.array([bool(tsp.shape[0] > 0) for tsp in sampling_tsps])       # initialize bool state vector
        p_explore = explore * 1       # initialize {0, 1} p vector for logging

        # initialize positions according to explore state vector such that samples can be collected on first step
        for agent in range(experiment.n_agents):
            if explore[agent]:  # send to next tsp point and delete point
                positions[agent, :] = sampling_tsps[agent][0, :]
                sampling_tsps[agent] = np.delete(sampling_tsps[agent], 0, axis=0)
            else:  # exploit -> go to centroid
                positions[agent, :] = centroids[agent, :]

        # perform learning and coverage steps in this epoch
        for step in range(epoch_length):

            # collect samples according to state vector
            sampling_agents = np.where(explore)[0]
            sampling_X = positions[sampling_agents]
            sampling_idx = [np.logical_and(np.isclose(data.x1, sampling_X[i, 0]), np.isclose(data.x2, sampling_X[i, 1]))
                            for i in range(sampling_agents.shape[0])]                     # compute indices at which to sample
            sampling_y = np.array([data.f_high[idx] for idx in sampling_idx]).flatten()   # take samples
            sampling_y = sampling_y + rng.normal(loc=0, scale=constants.sampling_noise, size=sampling_y.shape)

            # update model with new samples (if any) and recompute
            model.update(sampling_X, sampling_y)
            mu_star, cov_star, var_star = model.predict(X_star)

            # compute distance travelled, loss and regret
            distance = utils.compute_distance(positions, prev_positions)
            loss = utils.compute_loss(positions, data, partition)
            regret = utils.compute_regret(positions, data, partition)
            print(f"Iteration: {iteration}, Loss: {loss}") if iteration % 10 == 0 else None

            # log and plot progress
            logger.log("dmlc", sim, iteration, fidelity, positions, centroids,
                       max_var, argmax_var, p_explore, explore, distance, loss, regret)
            plotter.plot(positions, data, partition, estimate=mu_star.reshape(data.x1.shape),
                         estimate_var=var_star.reshape(data.x1.shape), regret=regret, loss=loss, model=model,
                         tsps0=sampling_tsps_0, tsps=sampling_tsps)

            # update partition if and only if all agents are on coverage phase
            if not np.any(explore):
                partition = utils.compute_partition(centroids, data, partition, experiment.gossip)

            # update centroids based on estimate and previous centroids
            centroids = utils.compute_centroids(positions, data, partition, estimate=mu_star.reshape(data.x1.shape))

            # find new points of maximum posterior variance (for logging purposes only)
            max_var, argmax_var = utils.compute_max_var(positions, data, partition, estimate_var=var_star.reshape(data.x1.shape))

            # update state vector and p vector for next step
            explore = np.array([bool(tsp.shape[0] > 0) for tsp in sampling_tsps])
            p_explore = explore * 1  # {0, 1} p vector for logging

            # save old positions and update new positions
            prev_positions = np.copy(positions)
            for agent in range(experiment.n_agents):
                if explore[agent]:  # send to next tsp point and delete point
                    positions[agent, :] = sampling_tsps[agent][0, :]
                    sampling_tsps[agent] = np.delete(sampling_tsps[agent], 0, axis=0)
                else:  # exploit -> go to centroid
                    positions[agent, :] = centroids[agent, :]

            # increment iteration counter and repeat within this epoch
            iteration += 1

        # increment epoch counter and proceed to next epoch
        epoch += 1


