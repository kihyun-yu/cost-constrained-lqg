import numpy as np
import cvxpy as cp


def eval_policy(A, B, Q, R, K, cov):
    max_iter = 10000
    tol = 1e-3
    conv = False

    # method 1: simulate long trajectory; not accurate
    # We do not use method 1, since it depends on random seed
    # np.random.seed(4)
    # x = np.zeros(A.shape[0]).reshape(-1, 1)
    # loss = 0
    # for i in range(max_iter):
    #     u = K @ x
    #     w = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, x.shape[1]).T
    #     loss += x.T @ Q @ x + u.T @ R @ u
    #     x = A @ x + B @ u + w
    #     avg_loss = loss / (i + 1)
    #     if i > 0 and abs(avg_loss - prev_avg_loss) < tol:
    #         conv = True
    #         break
    #     prev_avg_loss = avg_loss
    # avg_loss = avg_loss.item()

    # if the input policy is not stable, return inf
    if spectral_radius(A + B @ K) >= (1.0 - 1e-5):
        return 1e9

    conv = False
    X = np.zeros((A.shape[0], A.shape[0]))
    for i in range(max_iter):
        X = (A + B @ K) @ X @ (A + B @ K).T + cov
        if i % 20 == 0:
            Y = (A + B @ K) @ X @ (A + B @ K).T + cov
            if np.linalg.norm(X - Y) < tol:
                conv = True
                break
    avg_loss_2 = np.trace(Q @ X) + np.trace(R @ K @ X @ K.T)

    if conv:
        return avg_loss_2
    else:
        return 1e9


def get_dare(A, B, Q, R):
    P = Q
    max_iter = 10000
    tol = 1e-9
    conv = False
    for _ in range(max_iter):
        P_next = (
            A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
        )
        if np.linalg.norm(P_next - P, ord="fro") < tol:
            conv = True
            break
        P = P_next
    K = -np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    if conv:
        return K
    else:
        raise ValueError("DARE did not converge")


def get_rand_psd_matrix(n):
    A = np.random.rand(n, n)
    A = (A + A.T) / 2
    eigvals = np.linalg.eigvalsh(A)
    min_eigval = np.min(eigvals)
    if min_eigval < 0.1:
        A += (0.1 - min_eigval) * np.eye(n)
    return A


def get_mean_std(regrets):
    """
    Args:
        regrets: list of np.arrays with shape (num_simul,T)

    Returns:
        mean_regret, std_regret with shape (T,)
    """
    regrets = np.stack(regrets, axis=0)  # shape (num_simul, T)
    mean_regret = regrets.mean(axis=0)  # length T
    std_regret = regrets.std(axis=0)

    return mean_regret, std_regret


def spectral_radius(A):
    eigvals = np.linalg.eigvals(A)
    return max(abs(eigvals))
