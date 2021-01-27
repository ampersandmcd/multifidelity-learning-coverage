"""
Andrew McDonald
driver.py
Experimental engine to run learning+coverage algorithms.
"""

from algorithms import cortes, stochastic_multifidelity_learning_coverage, deterministic_multifidelity_learning_coverage
from utils import Experiment, Data, Logger, Plotter

if __name__ == "__main__":

    name = "corners"
    experiment = Experiment(name)
    data = Data(name)
    log = Logger(name)
    plotter = Plotter(name)

    for algorithm in experiment.algorithms:
        if algorithm == "cortes":
            for sim in range(experiment.n_simulations):
                cortes(experiment, data, log, plotter, "NA", sim)       # cortes does not need fidelity level
        else:
            for fidelity in experiment.fidelities:
                for sim in range(experiment.n_simulations):
                    if algorithm == "smlc":
                        stochastic_multifidelity_learning_coverage(experiment, data, log, plotter, fidelity, sim)
                    elif algorithm == "dmlc":
                        deterministic_multifidelity_learning_coverage(experiment, data, log, plotter, fidelity, sim)
                    else:
                        raise ValueError("Unknown algorithm type specified in config file.")

    log.save()