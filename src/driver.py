"""
Andrew McDonald
driver.py
Experimental engine to run learning+coverage algorithms.
"""

from algorithms import cortes, todescato, dslc
from utils import Experiment, Data, Logger, Plotter

if __name__ == "__main__":

    name = "forrester2007"
    experiment = Experiment(name)
    data = Data(name)
    log = Logger(name)
    plotter = Plotter(name)

    for algorithm in experiment.algorithms:
        for fidelity in experiment.fidelities:
            for sim in range(experiment.n_simulations):
                if algorithm == "cortes":
                    cortes(experiment, data, log, plotter, fidelity, sim)
                elif algorithm == "todescato":
                    todescato(experiment, data, log, plotter, fidelity, sim)
                elif algorithm == "dslc":
                    dslc(experiment, data, log, plotter, fidelity, sim)
                else:
                    raise ValueError("Unknown algorithm type specified in config file.")
