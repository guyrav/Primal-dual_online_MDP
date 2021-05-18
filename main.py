import numpy as np


def grads(q, P, r, mu, v):
    """

    :param q: shape: (S)
    :param P: shape: (S, A ,S)
    :param r: shape: (S, A, S)
    :param mu: shape: (H, S, A)
    :param v: shape: (H, S)
    :return: (g_mu, g_v) where g_mu.shape = (H, S, A) and gv.shape = (H, S)
    """
    S = len(q)
    A = P.shape[1]
    H = v.shape[0]

    q_mat = q.reshape((1, S))  # Shape: (1, S)
    nu_mat = np.einsum('hia,iaj->hj', mu[:-1], P)  # Shape: (H-1, S)
    chi_mat = np.einsum('hia->hi', mu)  # Shape: (H, S)
    g_v = np.concatenate(q_mat, nu_mat) - chi_mat  # Shape: (H, S)

    curr_val_mat = v.reshape((H, S, 1))  # Shape: (H, S, 1)
    next_val_mat = np.concatenate(
        np.einsum('iaj,hj->hia', P, v[1:]),
        np.zeros((1, S, A))
    )  # Shape: (H, S, A)
    expected_r = np.einsum('iaj,iaj->ia', P, r).reshape((1, S, A))  # Shape: (1, S, A)
    g_mu = next_val_mat + expected_r - curr_val_mat  # Shape: (H, S, A)

    return g_v, g_mu


def GD_step(v, g_v, eta):
    H = v.shape[0]
    return np.maximum(
        0,
        np.minimum(
            np.flip(np.arange(H) + 1).reshape(H, 1),
            v - eta*g_v
        )
    )


def hedge_step(mu, g_mu, eta):
    H, S, _ = mu.shape
    updated_mu = mu * np.exp(eta*g_mu)
    return updated_mu / np.einsum('...a->...', mu).reshape((H, S, 1))


def main():
    # Set MDP parameters
    S = 2
    A = 2
    H = 2
    q = np.ones(S)/S
    P = np.ones((S, A, S))/S
    r = [[[1, 0],
          [0, 1]],
         [[0, 0],
          [0, 0]]]

    # Set learning parameters
    eta_mu = 0.1
    eta_v = 0.1
    T = 100

    # Initialize mu
    mu = np.zeros((H, S, A))
    uniform_action = np.ones(A)/A
    mu[0] = np.outer(q, uniform_action)
    for i in range(H-1):
        mu[i+1] = np.outer(np.einsum('ia,iaj', mu[i], P), uniform_action)

    # Initialize v
    v = np.ones((H, S)) * np.flip(np.arange(H) + 1).reshape((H, 1))

    # Learn
    for t in range(T):
        g_v, g_mu = grads(q, P, r, mu, v)
        v = GD_step(v, g_v, eta_v)
        mu = hedge_step(mu, g_mu, eta_mu)


