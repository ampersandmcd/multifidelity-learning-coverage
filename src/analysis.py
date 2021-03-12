"""
Andrew McDonald
analysis.py
Helper functions for simulation data post-processing and analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt

import constants


def plot_loss(df, filename):

    # collapse data by algorithm, fidelity and iteration number and take mean
    mean_df = df.groupby(by=["name", "fidelity", "iteration"]).mean().reset_index()
    mean_df["algorithm"] = mean_df["name"] + "_" + mean_df["fidelity"]
    mean_df = mean_df[["algorithm", "iteration", "loss"]]
    mean_pivot = mean_df.pivot(index="iteration", columns="algorithm", values="loss")

    # subset data by smlc and dmlc
    smlc = mean_pivot[[col for col in mean_pivot.columns if "smlc" in col or "cortes" in col]]
    dmlc = mean_pivot[[col for col in mean_pivot.columns if "dmlc" in col or "cortes" in col]]

    # reorganize to put cortes column at end
    smlc = smlc[[col for col in smlc.columns if "cortes" not in col] + ["cortes_na"]]
    dmlc = dmlc[[col for col in dmlc.columns if "cortes" not in col] + ["cortes_na"]]

    # plot mean loss
    fig, axs = plt.subplots(2, 1, figsize=constants.analysis_figsize)

    smlc.plot(ax=axs[0])
    axs[0].set_title("SMLC: Mean Loss by Iteration")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Iteration")

    dmlc.plot(ax=axs[1])
    axs[1].set_title("DMLC: Mean Loss by Iteration")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Iteration")

    fig.tight_layout()
    fig.subplots_adjust(wspace=1)
    fig.savefig(f"{filename}_loss.png")
    plt.show()


def plot_regret(df, filename):

    # collapse data by algorithm, fidelity and iteration number and take mean
    mean_df = df.groupby(by=["name", "fidelity", "iteration"]).mean().reset_index()
    mean_df["algorithm"] = mean_df["name"] + "_" + mean_df["fidelity"]
    mean_df = mean_df[["algorithm", "iteration", "regret"]]
    mean_pivot = mean_df.pivot(index="iteration", columns="algorithm", values="regret")

    # subset data by smlc and dmlc
    smlc = mean_pivot[[col for col in mean_pivot.columns if "smlc" in col or "cortes" in col]]
    dmlc = mean_pivot[[col for col in mean_pivot.columns if "dmlc" in col or "cortes" in col]]

    # reorganize to put cortes column at end
    smlc = smlc[[col for col in smlc.columns if "cortes" not in col] + ["cortes_na"]]
    dmlc = dmlc[[col for col in dmlc.columns if "cortes" not in col] + ["cortes_na"]]

    # plot mean loss
    fig, axs = plt.subplots(2, 1, figsize=constants.analysis_figsize)

    smlc.plot(ax=axs[0])
    axs[0].set_title("SMLC: Mean Regret by Iteration")
    axs[0].set_ylabel("Regret")
    axs[0].set_xlabel("Iteration")

    dmlc.plot(ax=axs[1])
    axs[1].set_title("DMLC: Mean Regret by Iteration")
    axs[1].set_ylabel("Regret")
    axs[1].set_xlabel("Iteration")

    fig.tight_layout()
    fig.subplots_adjust(wspace=1)
    fig.savefig(f"{filename}_regret.png")
    plt.show()


def plot_cumulative_loss(df, filename):

    # collapse data by algorithm, fidelity and iteration number and take mean
    mean_df = df.groupby(by=["name", "fidelity", "iteration"]).mean().reset_index()
    mean_df["algorithm"] = mean_df["name"] + "_" + mean_df["fidelity"]
    mean_df = mean_df[["algorithm", "iteration", "loss"]]
    mean_pivot = mean_df.pivot(index="iteration", columns="algorithm", values="loss")

    # subset data by smlc and dmlc
    smlc = mean_pivot[[col for col in mean_pivot.columns if "smlc" in col or "cortes" in col]]
    dmlc = mean_pivot[[col for col in mean_pivot.columns if "dmlc" in col or "cortes" in col]]

    # reorganize to put cortes column at end
    smlc = smlc[[col for col in smlc.columns if "cortes" not in col] + ["cortes_na"]]
    dmlc = dmlc[[col for col in dmlc.columns if "cortes" not in col] + ["cortes_na"]]

    # take cumulative sum of final table to get cumulative loss
    smlc = smlc.cumsum()
    dmlc = dmlc.cumsum()

    # plot cumulative mean loss
    fig, axs = plt.subplots(2, 1, figsize=constants.analysis_figsize)

    smlc.plot(ax=axs[0])
    axs[0].set_title("SMLC: Cumulative Mean Loss by Iteration")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Iteration")

    dmlc.plot(ax=axs[1])
    axs[1].set_title("DMLC: Cumulative Mean Loss by Iteration")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Iteration")

    fig.tight_layout()
    fig.subplots_adjust(wspace=1)
    fig.savefig(f"{filename}_cumulative_loss.png")
    plt.show()


def plot_var(df, filename):

    # collapse data by algorithm, fidelity and iteration number and take mean of max_var
    mean_df = df.groupby(by=["name", "fidelity", "iteration"]).mean().reset_index()
    mean_df["algorithm"] = mean_df["name"] + "_" + mean_df["fidelity"]
    mean_df = mean_df[["algorithm", "iteration", "max_var"]]
    mean_pivot = mean_df.pivot(index="iteration", columns="algorithm", values="max_var")

    # subset data by smlc and dmlc
    smlc = mean_pivot[[col for col in mean_pivot.columns if "smlc" in col or "cortes" in col]]
    dmlc = mean_pivot[[col for col in mean_pivot.columns if "dmlc" in col or "cortes" in col]]

    # reorganize to put cortes column at end
    smlc = smlc[[col for col in smlc.columns if "cortes" not in col] + ["cortes_na"]]
    dmlc = dmlc[[col for col in dmlc.columns if "cortes" not in col] + ["cortes_na"]]

    # plot mean loss
    fig, axs = plt.subplots(2, 1, figsize=constants.analysis_figsize)

    smlc.plot(ax=axs[0])
    axs[0].set_title("SMLC: Mean Maximum Posterior Variance by Iteration")
    axs[0].set_ylabel("Maximum Posterior Variance")
    axs[0].set_xlabel("Iteration")

    dmlc.plot(ax=axs[1])
    axs[1].set_title("DMLC: Mean Maximum Posterior Variance by Iteration")
    axs[1].set_ylabel("Maximum Posterior Variance")
    axs[1].set_xlabel("Iteration")

    fig.tight_layout()
    fig.subplots_adjust(wspace=1)
    fig.savefig(f"{filename}_variance.png")
    plt.show()


def plot_distance(df, filename):

    # collapse data by algorithm, fidelity and iteration number and take mean of distance travelled
    mean_df = df.groupby(by=["name", "fidelity", "iteration"]).mean().reset_index()
    mean_df["algorithm"] = mean_df["name"] + "_" + mean_df["fidelity"]
    mean_df = mean_df[["algorithm", "iteration", "distance"]]
    mean_pivot = mean_df.pivot(index="iteration", columns="algorithm", values="distance")

    # subset data by smlc and dmlc
    smlc = mean_pivot[[col for col in mean_pivot.columns if "smlc" in col or "cortes" in col]]
    dmlc = mean_pivot[[col for col in mean_pivot.columns if "dmlc" in col or "cortes" in col]]

    # reorganize to put cortes column at end
    smlc = smlc[[col for col in smlc.columns if "cortes" not in col] + ["cortes_na"]]
    dmlc = dmlc[[col for col in dmlc.columns if "cortes" not in col] + ["cortes_na"]]

    # plot mean loss
    fig, axs = plt.subplots(2, 1, figsize=constants.analysis_figsize)

    smlc.plot(ax=axs[0])
    axs[0].set_title("SMLC: Mean Distance Travelled by Iteration")
    axs[0].set_ylabel("Distance")
    axs[0].set_xlabel("Iteration")

    dmlc.plot(ax=axs[1])
    axs[1].set_title("DMLC: Mean Distance Travelled by Iteration")
    axs[1].set_ylabel("Distance")
    axs[1].set_xlabel("Iteration")

    fig.tight_layout()
    fig.subplots_adjust(wspace=1)
    fig.savefig(f"{filename}_distance.png")
    plt.show()


def plot_mse(df, filename):

    # collapse data by algorithm, fidelity and iteration number and take mean of distance travelled
    mean_df = df.groupby(by=["name", "fidelity", "iteration"]).mean().reset_index()
    mean_df["algorithm"] = mean_df["name"] + "_" + mean_df["fidelity"]
    mean_df = mean_df[["algorithm", "iteration", "mse"]]
    mean_pivot = mean_df.pivot(index="iteration", columns="algorithm", values="mse")

    # subset data by smlc and dmlc
    smlc = mean_pivot[[col for col in mean_pivot.columns if "smlc" in col or "cortes" in col]]
    dmlc = mean_pivot[[col for col in mean_pivot.columns if "dmlc" in col or "cortes" in col]]

    # reorganize to put cortes column at end
    smlc = smlc[[col for col in smlc.columns if "cortes" not in col] + ["cortes_na"]]
    dmlc = dmlc[[col for col in dmlc.columns if "cortes" not in col] + ["cortes_na"]]

    # plot mean mse
    fig, axs = plt.subplots(2, 1, figsize=constants.analysis_figsize)

    smlc.plot(ax=axs[0])
    axs[0].set_title("SMLC: MSE of Sensory Function by Iteration")
    axs[0].set_ylabel("MSE")
    axs[0].set_xlabel("Iteration")

    dmlc.plot(ax=axs[1])
    axs[1].set_title("DMLC: MSE of Sensory Function by Iteration")
    axs[1].set_ylabel("MSE")
    axs[1].set_xlabel("Iteration")

    fig.tight_layout()
    fig.subplots_adjust(wspace=1)
    fig.savefig(f"{filename}_mse.png")
    plt.show()


if __name__ == "__main__":

    name = "bump_01_28_2021_23_49_40"
    csv_filename, img_basename = f"../logs/{name}.csv", f"../plots/{name}"

    df = pd.read_csv(csv_filename, header="infer", index_col=0)
    df["fidelity"] = df["fidelity"].fillna("na")        # interpret na fidelity level as string to keep data
    df = df[~df["fidelity"].str.contains("null")]       # drop null simulations
    print()

    plot_loss(df, img_basename)
    plot_regret(df, img_basename)
    plot_cumulative_loss(df, img_basename)
    plot_var(df, img_basename)
    plot_distance(df, img_basename)
    plot_mse(df, img_basename)

    print()
