import numpy as np
from utils import *
from copy import deepcopy
import cvxpy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm

tqdm.disable = True


class OnlineSDP:
    def __init__(self, clqr, beta, mu, lam, unknown_trans=None, init_tran=None):
        self.clqr = deepcopy(clqr)
        self.unknown_trans = unknown_trans
        self.dx = self.clqr.dx
        self.du = self.clqr.du

        # true transition; avoid to use in the learning algorithm
        self.A = self.clqr.A
        self.B = self.clqr.B

        if init_tran == "set_perturb":
            self.A0 = self.A + 0.1 * np.random.standard_normal(self.A.shape)
            self.B0 = self.B + 0.1 * np.random.standard_cauchy(self.B.shape)

        elif init_tran == "set_zero":
            self.A0 = np.zeros_like(self.A)
            self.B0 = np.zeros_like(self.B)

        elif init_tran == "set_random":
            self.A0 = np.random.standard_normal(self.A.shape)
            self.B0 = np.random.standard_normal(self.B.shape)

        elif init_tran is None and unknown_trans == False:
            self.A0 = self.A
            self.B0 = self.B

        else:
            raise ValueError("Undefined input for init_tran")

        self.Qf = self.clqr.Qf
        self.Rf = self.clqr.Rf
        self.Qg = self.clqr.Qg
        self.Rg = self.clqr.Rg
        self.cov = self.clqr.cov
        self.b = self.clqr.b
        self.opt_obj = self.clqr.obj
        self.stable_K = self.get_stable_policy()
        self.beta = beta
        self.mu = mu
        self.lam = lam

        self.now = 1
        self.x = np.zeros(self.dx)
        self.regret = []
        self.violation = []
        self.zs = []
        self.xs = []

        self.log_est_error_AB = []
        self.log_K_loss = []
        self.log_K_cost = []

    def estimate_AB(self):
        if self.unknown_trans == False:
            return self.A, self.B
        elif len(self.xs) == 0 or len(self.zs) == 0:
            return self.A0, self.B0
        else:
            X = np.column_stack(self.xs)
            Z = np.column_stack(self.zs)
            M0 = np.hstack([self.A0, self.B0])

            M = (X @ Z.T + self.beta * self.lam * M0) @ np.linalg.inv(
                Z @ Z.T + self.beta * self.lam * np.eye(self.dx + self.du)
            )
            A = M[:, : self.dx]
            B = M[:, self.dx :]

            return A, B

    def get_stable_policy(self):
        for _ in range(10000):
            K = np.random.standard_normal((self.du, self.dx))
            if spectral_radius(self.A + self.B @ K) < 1:
                return K
        raise ValueError("Stable policy has not been found!")

    def step(self, K, x):
        u = K @ x
        w = np.random.multivariate_normal(np.zeros(self.dx), self.cov)  # shape (dx,)

        # Cost
        running_obj = self.x.T @ self.Qf @ self.x + u.T @ self.Rf @ u
        running_vio = self.x.T @ self.Qg @ self.x + u.T @ self.Rg @ u

        # Update regret and constraint violation
        self.regret.append(running_obj - self.opt_obj)
        self.violation.append(running_vio - self.b)

        # Update zs for estimating A,B
        self.zs.append(np.concatenate([self.x, u]))

        # Move to the next state
        self.x = (self.A @ self.x) + (self.B @ u) + w

        # Update xs for estimating A,B
        self.xs.append(self.x.copy())

        self.now += 1
        pass

    def reset(self):
        pass

    def solve_optimistic_sdp(self, A, B, V):

        d = self.dx + self.du
        Sigma = cp.Variable((d, d), symmetric=True)

        Sigma_xx = Sigma[: self.dx, : self.dx]
        Sigma_xu = Sigma[: self.dx, self.dx :]
        Sigma_ux = Sigma[self.dx :, : self.dx]
        Sigma_uu = Sigma[self.dx :, self.dx :]

        V_inv = np.linalg.inv(V)

        # Optimistic Constraints
        if self.unknown_trans == True:
            constraints = [
                cp.trace(self.Qg @ Sigma_xx) + cp.trace(self.Rg @ Sigma_uu) <= self.b,
                Sigma_xx
                >> cp.bmat([[A, B]]) @ Sigma @ cp.bmat([[A.T], [B.T]])
                + self.cov
                - self.mu * cp.trace(Sigma.T @ V_inv) * np.eye(self.dx),
                Sigma >> 0,
            ]

        # Non-optimistic Contraints
        if self.unknown_trans == False:
            constraints = [
                cp.trace(self.Qg @ Sigma_xx) + cp.trace(self.Rg @ Sigma_uu) <= self.b,
                Sigma_xx
                == cp.bmat([[A, B]]) @ Sigma @ cp.bmat([[A.T], [B.T]]) + self.cov,
                Sigma >> 0,
            ]

        objective = cp.Minimize(
            cp.trace(self.Qf @ Sigma_xx) + cp.trace(self.Rf @ Sigma_uu)
        )
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        if prob.status not in ["infeasible", "unbounded"]:
            self.Sigma_opt = Sigma.value
            # assert np.min(np.linalg.eigvals(self.Sigma_opt[:self.dx, : self.dx])) < 1e-2, "Sigma_xx is close to singular!"
            self.K = self.Sigma_opt[: self.dx, self.dx :].T @ np.linalg.inv(
                self.Sigma_opt[: self.dx, : self.dx]
            )
            return self.K
        else:
            return None

    def run(self, T, T_warmup, verbose=True):
        V = self.lam * np.eye(self.dx + self.du)
        det_init_epoch = self.lam

        debug_count_infeasible = 0

        for t in tqdm(range(1, T_warmup + T + 1)):
            if t <= T_warmup:
                K = self.stable_K
                # A, B = self.estimate_AB()
                A, B = self.A0, self.B0
            else:
                detV = np.linalg.det(V)
                if (
                    t == (T_warmup + 1) or detV > det_init_epoch * 2
                ):  # if new information arrives sufficiently, then update estimates
                    if t == 1:
                        A, B = self.A0, self.B0
                    else:
                        A, B = self.estimate_AB()
                        det_init_epoch = detV
                    K = self.solve_optimistic_sdp(A, B, V)
                else:
                    pass  # if new information does not arrive sufficiently, then use the previous estimates

            # if SDP is infeasible, run the known stable policy, to avoid divergence
            if K is None:
                debug_count_infeasible += 1
                K = self.stable_K

            # log estiamtion error of A,B
            est_error_AB = np.linalg.norm(A - self.A) + np.linalg.norm(B - self.B)
            self.log_est_error_AB.append(est_error_AB)

            # log convergence of policy
            K_loss = eval_policy(self.A, self.B, self.Qf, self.Rf, K, self.cov)
            K_cost = eval_policy(self.A, self.B, self.Qg, self.Rg, K, self.cov)
            if K_loss == "unstable":
                self.log_K_loss.append(self.log_K_loss[-1])
            else:
                self.log_K_loss.append(K_loss)
            if K_cost == "unstable":
                self.log_K_cost.append(self.log_K_cost[-1])
            else:
                self.log_K_cost.append(K_cost)

            # step
            self.step(K, self.x)
            z = self.zs[-1]
            V += (1 / self.beta) * np.outer(z, z)

            if t == T_warmup + T:
                self.last_obj = (
                    eval_policy(
                        K=self.K, A=self.A, B=self.B, Q=self.Qf, R=self.Rf, cov=self.cov
                    ),
                    eval_policy(
                        K=self.K, A=self.A, B=self.B, Q=self.Qg, R=self.Rg, cov=self.cov
                    ),
                )

        self.regret = np.cumsum(self.regret)
        self.violation = np.cumsum(self.violation)

        print(f"Num. of Infeasible Case: {debug_count_infeasible}")
