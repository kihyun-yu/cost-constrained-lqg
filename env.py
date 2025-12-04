import numpy as np
from utils import *
import argparse
import pickle
import os


class CLQR:
    def __init__(self, dx, du, sigma_square=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Dimensions
        self.dx = dx
        self.du = du

        # Generate CLQR parameters
        self.sigma_square = sigma_square
        self.Qf, self.Rf, self.Qg, self.Rg, self.A, self.B, self.cov = (
            self.generate_parameters()
        )

        # Optimal policy and its value functions
        self.K = get_dare(self.A, self.B, self.Qf, self.Rf)
        self.obj = eval_policy(self.A, self.B, self.Qf, self.Rf, self.K, self.cov)
        self.constraint_obj = eval_policy(
            self.A, self.B, self.Qg, self.Rg, self.K, self.cov
        )

        # budget; optimal policy's constraint violation + 0.01
        self.b = 0.01 + self.constraint_obj

        if seed is not None:
            np.random.seed(None)

    def generate_parameters(self):
        Qf = get_rand_psd_matrix(self.dx)
        Rf = get_rand_psd_matrix(self.du)
        Qg = get_rand_psd_matrix(self.dx)
        Rg = get_rand_psd_matrix(self.du)
        A = np.random.rand(self.dx, self.dx)
        B = np.random.rand(self.dx, self.du)
        if self.sigma_square is None:
            cov = get_rand_psd_matrix(self.dx)
        else:
            cov = np.eye(self.dx) * self.sigma_square
        return Qf, Rf, Qg, Rg, A, B, cov

    def display(self):
        print("\n--------- Env Description --------------")
        print("A:\n", self.A)
        print("\nB:\n", self.B)
        print("\nQf:\n", self.Qf)
        print("\nRf:\n", self.Rf)
        print("\nQg:\n", self.Qg)
        print("\nRg:\n", self.Rg)
        print("\nCovariance:\n", self.cov)
        print("\nOptimal Objective:", self.obj)
        print("Optimal Constraint Objective:", self.constraint_obj)
        print("Constraint bound b:", self.b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=int, default=4)
    parser.add_argument("--du", type=int, default=2)
    parser.add_argument("--inst_seed", type=int, default=42)
    parser.add_argument("--sigma_square", type=float, default=None)
    parser.add_argument("--save_dir", default="./envs")

    args = parser.parse_args()

    clqr = CLQR(
        dx=args.dx, du=args.du, sigma_square=args.sigma_square, seed=args.inst_seed
    )

    clqr.display()

    file_name = f"dx{args.dx}_du{args.du}_seed{args.inst_seed}_sigmaSquare{args.sigma_square}.pkl"

    fpath = os.path.join(args.save_dir, file_name)

    with open(fpath, "wb") as f:
        pickle.dump(clqr, f)
    print("\n-------Save Env Done---------\n")
