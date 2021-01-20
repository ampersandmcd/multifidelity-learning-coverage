"""
Andrew McDonald
algorithms.py
Implementations of multifidelity learning+coverage algorithms.
"""

import numpy as np
import utils


def cortes(experiment, data, logger, plotter, fidelity, sim):

    # initialize random starting positions
    positions = np.random.rand(experiment.n_agents, 2)      # agent i position is in row i with [x, y]
    prev_positions = np.copy(positions)                     # save copy to compute distance travelled

    # initialize partition and centroids
    partition = utils.compute_partition(positions, data, None, experiment.gossip)
    centroids = utils.compute_centroids(positions, data, partition, estimate=data.f_high)

    for iteration in range(experiment.n_iterations):

        # compute distance travelled, loss and regret
        distance = utils.compute_distance(positions, prev_positions)
        loss = utils.compute_loss(positions, data, partition)
        regret = utils.compute_regret(positions, data, partition)

        # log and plot progress
        logger.log(sim, iteration, positions, partition, centroids, distance, loss, regret)
        plotter.plot(positions, data, partition, estimate=data.f_high, estimate_var=0, regret=regret)

        # update partition and centroids given perfect knowledge
        partition = utils.compute_partition(positions, data, partition, experiment.gossip)
        centroids = utils.compute_centroids(positions, data, partition, estimate=data.f_high)

        # perform Lloyd update
        prev_positions = np.copy(positions)
        positions = np.copy(centroids)


def todescato(experiment, data, logger, plotter, fidelity, sim):
    pass


def dslc(experiment, data, logger, plotter, fidelity, sim):
    pass