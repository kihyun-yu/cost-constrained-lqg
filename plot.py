import numpy as np
import utils
from algorithms import OnlineSDP
from matplotlib import pyplot as plt
import os


def plot(regrets, title, fpath):

    mean_regret, std_regret = utils.get_mean_std(regrets)

    plt.figure(figsize=(12, 5))
    plt.plot(mean_regret)
    plt.fill_between(
        np.arange(len(mean_regret)),
        mean_regret - std_regret,
        mean_regret + std_regret,
        alpha=0.2,
    )
    # plt.title(title)
    fname = fpath + "_" + title + ".png"
    plt.savefig(fname)
