import numpy as np
from matplotlib import pyplot as plt


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
    g_v = np.concatenate((q_mat, nu_mat)) - chi_mat  # Shape: (H, S)

    curr_val_mat = v.reshape((H, S, 1))  # Shape: (H, S, 1)
    next_val_mat = np.concatenate((
        np.einsum('iaj,hj->hia', P, v[1:]),
        np.zeros((1, S, A))
    ))  # Shape: (H, S, A)
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


def hedge_step(mu, g_mu, eta, q):
    H, S, A = mu.shape
    updated_mu = mu * np.exp(eta*g_mu)
    updated_mu_0 = updated_mu[0].reshape((1, S, A))
    updated_mu_0 = updated_mu_0 / np.einsum('hia->hi', updated_mu_0).reshape((1, S, 1)) * q.reshape((1, S, 1))
    updated_mu_other = updated_mu[1:]
    updated_mu_other = updated_mu_other / np.einsum('hia->h', updated_mu_other).reshape((H-1, 1, 1))
    updated_mu = np.concatenate((updated_mu_0, updated_mu_other))
    return updated_mu


def main():
    # Set MDP parameters
    S = 2
    A = 2
    H = 2
    q = np.array([0.5, 0.5])
    P = np.array([[[1, 0],
                   [0, 1]],
                  [[1, 0],
                   [0, 1]]])
    r = np.array([[[1, 0],
                   [0, 0]],
                  [[0, 0],
                   [0, 0]]])

    # Set learning parameters
    eta_mu = 0.1
    eta_v = 0.1
    T = 1000

    # Initialize mu
    mu = np.zeros((H, S, A))
    uniform_action = np.ones(A)/A
    mu[0] = np.outer(q, uniform_action)
    for i in range(H-1):
        mu[i+1] = np.outer(np.einsum('ia,iaj', mu[i], P), uniform_action)

    # Initialize v
    v = np.ones((H, S)) * np.flip(np.arange(H) + 1).reshape((H, 1))

    # Debug value
    mu_000 = np.zeros(T+1)
    mu_000[0] = mu[0, 0, 0]

    # Learn
    for t in range(T):
        g_v, g_mu = grads(q, P, r, mu, v)
        v = GD_step(v, g_v, eta_v)
        mu = hedge_step(mu, g_mu, eta_mu, q)
        mu_000[t+1] = mu[0, 0, 0]

        print(f"t={t}")
        print(f"v={v.flatten()},\tmu={mu.flatten()}")
        print(f"g_v={g_v.flatten()},\tg_mu={g_mu.flatten()}")

    plt.plot(range(T+1), mu_000)
    plt.show()


if __name__ == "__main__":
    main()


