import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import yaml
import pickle

from env import CLQR
from plot import plot
from algorithms import *
from utils import *

tqdm.disable = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    # load config
    if args.config is not None:
        with open(args.config, "r") as read_config:
            config = yaml.safe_load(read_config)

    # save dir check
    os.makedirs(config["save_dir"], exist_ok=True)
    config["save_file_name"] = config["save_file_name"].format(**config)

    # set running seed
    if config["running_seed"] is None:
        config["running_seed"] = list(range(config["num_simul"]))

    # load a LQR instance
    load_path = os.path.join(config["load_env_dir"], config["load_env_file_name"])
    load_path += ".pkl"
    with open(load_path, "rb") as f:
        clqr: CLQR = pickle.load(f)

    regrets = []
    violations = []

    logs_est_error_AB = []
    logs_K_loss = []

    for nth in range(config["num_simul"]):
        np.random.seed(config["running_seed"][nth])
        osdp = OnlineSDP(
            clqr,
            beta=config["beta"],
            mu=config["mu"],
            lam=config["lam"],
            unknown_trans=config["unknown_trans"],
            init_tran=config["init_trans"],
        )
        osdp.run(T=config["T"], T_warmup=config["T_warmup"], verbose=config["verbose"])

        regrets.append(osdp.regret)
        violations.append(osdp.violation)
        logs_est_error_AB.append(osdp.log_est_error_AB)
        logs_K_loss.append(osdp.log_K_loss)

    # plot
    fpath = os.path.join(config["save_dir"], config["save_file_name"])
    plot(regrets, "Cumulative Regret", fpath)
    plot(violations, "Cumulative Violations", fpath)
    plot(logs_est_error_AB, "Estimation Error", fpath)
    plot(logs_K_loss, "Convergence", fpath)


if __name__ == "__main__":
    main()
