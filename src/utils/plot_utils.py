import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import pandas as pd

def shm_plotter(correct_counts, save_directory, data_generating_function):
    correct_counts = pd.DataFrame(correct_counts)
    # std = pd.DataFrame(std)
    # std = pd.melt(std, ['Env'])
    # correct_counts = pd.melt(correct_counts, ['Env'])
    # correct_counts['std'] = std['value']
    with sns.plotting_context("paper", rc={"font.size": 15, "axes.titlesize": 15,
                                           "axes.labelsize": 15, "legend.fontsize": 12,
                                           "lines.markersize": 8, "xtick.labelsize": 10,
                                           "ytick.labelsize": 10, 'lines.linewidth': 3}):
        ax = sns.lineplot(x='Env', y='Causal-de-Finetti', data=correct_counts, marker='o', linewidth=5,
                          label='Causal-de-Finetti', errorbar='ci')
        # sns.lineplot(x='Env', y='CD-NOD', data=correct_counts, marker='^', label='CD-NOD', ax=ax)
        sns.lineplot(x='Env', y='FCI', marker='X', data=correct_counts, label='FCI')
        sns.lineplot(x='Env', y='GES', marker = 'd', data=correct_counts, label='GES')
        sns.lineplot(x='Env', y='NOTEARS', marker = '*', data=correct_counts, label='NOTEARS')
        sns.lineplot(x='Env', y='PC', data=correct_counts, marker = 'p', label = 'PC', ax = ax, color = 'r', errorbar='ci')
        sns.lineplot(x='Env', y='Random', data=correct_counts, linestyle = '--', color='k', label = 'Random')
        # ax.set_title("Structural Hamming Distance")
        ax.set(xlabel="Number of environments", ylabel = 'Structural Hamming Distance \n (the lower the better)')
        ax.set_ylim(0)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.savefig(f'{save_directory}SHD_{data_generating_function.__name__}', bbox_inches="tight",
                    dpi=300)
        plt.close()


def correct_counts_plotter(correct_counts, save_directory, data_generating_function):
    correct_counts = pd.DataFrame(correct_counts)
    # correct_counts = pd.melt(correct_counts, ['Env'])
    # print(correct_counts)
    with sns.plotting_context("paper", rc={"font.size": 15, "axes.titlesize": 15,
                                           "axes.labelsize": 15, "legend.fontsize": 12,
                                           "lines.markersize": 8, "xtick.labelsize": 10,
                                           "ytick.labelsize": 10, 'lines.linewidth': 3}):
        ax = sns.lineplot(x='Env', y='Causal-de-Finetti', data = correct_counts, marker = 'o', linewidth = 5, label = 'Causal-de-Finetti')
        sns.lineplot(x='Env', y='FCI', marker='X', data=correct_counts, label='FCI', ax = ax)
        sns.lineplot(x='Env', y='GES', marker = 'd', data=correct_counts, label='GES', ax=ax)
        sns.lineplot(x='Env', y='NOTEARS', marker = '*', data=correct_counts, label='NOTEARS', ax=ax)
        sns.lineplot(x='Env', y='CD-NOD', data=correct_counts, marker='^', label='CD-NOD', ax=ax)
        sns.lineplot(x='Env', y='DirectLinGAM', data=correct_counts, marker = 's', label ='DirectLinGAM', ax = ax)
        sns.lineplot(x='Env', y='Random', data=correct_counts, linestyle = '--', color='k', label = 'Random', ax=ax)
        # ax.set_title("Proportion correct causal direction detected")
        ax.set(xlabel = "Number of environments", ylabel = 'Proportion correct')
        ax.set_ylim(0, 1)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.savefig(f'{save_directory}correct_counts_comparisons_{data_generating_function.__name__}', bbox_inches="tight", dpi=300)
        plt.close()

def computation_time_plotter(computation_time, save_directory, data_generating_function):
    computation_time = pd.DataFrame(computation_time)
    with sns.plotting_context("paper", rc={"font.size": 15, "axes.titlesize": 15,
                                           "axes.labelsize": 15, "legend.fontsize": 12,
                                           "lines.markersize": 8, "xtick.labelsize": 10,
                                           "ytick.labelsize": 10, 'lines.linewidth': 3}):
        ax = sns.lineplot(x='num_samples', y='value', hue = 'variable', data= pd.melt(computation_time, ['num_samples']), marker='o')
        ax.set_title("Computation time needed")
        ax.set(xlabel = "Number of samples", ylabel = 'Time to compute')
        plt.legend()
        plt.savefig(f'{save_directory}computation_time_comparisons_{data_generating_function.__name__}', bbox_inches="tight", dpi=300)
        plt.close()

def roc_plotter_overlay(tp: List, fp: List,
                        labels: List, save_directory: str):
    """ Plots the ROC curve for a given true positive and false positive rate.
    """
    path = save_directory + 'roc_scm_bivariate_binary.png'
    with sns.plotting_context("paper", rc={"font.size": 15, "axes.titlesize": 15,
                                           "axes.labelsize": 15, "legend.fontsize": 12,
                                           "lines.markersize": 8, "xtick.labelsize": 10,
                                           "ytick.labelsize": 10}):
        ax = sns.lineplot(x=fp[0], y=tp[0], ci=None, estimator=None, label=labels[0])
        for i in range(1, len(labels)):
            sns.lineplot(x=fp[i], y=tp[i], ci=None, estimator=None, label=labels[i], ax=ax)

        plt.plot([0, 1.], [0, 1.], color="red", linestyle="--")

        ax.set_title('ROC curve');
        ax.set(xlabel='False positive rate', ylabel='True positive rate');
        ax.set_aspect('equal')
        plt.legend(frameon=False)

        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()