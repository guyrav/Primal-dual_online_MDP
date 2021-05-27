import numpy as np
from decision_process import MDP


def ind(shape, inds):
    e = np.zeros(shape)
    e[inds] = 1
    return e


def grads(q: np.ndarray,
          states: np.ndarray,
          actions: np.ndarray,
          rewards: np.ndarray,
          sample_distribution: np.ndarray,
          mu: np.ndarray,
          v: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray]:
    """

    :param q: shape (S)
    :param states: shape (H+1)
    :param actions: shape (H)
    :param rewards: shape (H)
    :param sample_distribution: shape (H, S, A)
    :param mu: shape (H, S, A)
    :param v: shape (H, S)
    :return: g_v of shape (H, S), g_mu of shape (H, S, A)
    """
    H, S, A = mu.shape

    q_mat = q.reshape((1, S))  # Shape: (1, S)
    sample_chi = np.einsum('hia->hi', sample_distribution)  # Shape: (H, S)
    sample_pi = sample_distribution / sample_chi.reshape((H, S, 1))
    chi = np.einsum('hia->hi', mu)  # Shape: (H, S)
    nu_estimator = mu[(np.arange(H), states[:-1], actions)]
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


def initial_mu(H, S, A, P):
    mu = np.zeros((H, S, A))
    uniform_action = np.ones(A) / A
    mu[0] = np.outer(q, uniform_action)
    for i in range(H - 1):
        mu[i + 1] = np.outer(np.einsum('ia,iaj', mu[i], P), uniform_action)
    return mu


def initial_v(H, S):
    return np.ones((H, S)) * np.flip(np.arange(H) + 1).reshape((H, 1))


def main():
    # MDP Parameters
    H = 2
    S = 2
    A = 2
    actions = np.arange(A)
    P = np.array([[[1, 0],
                   [0, 1]],
                  [[1, 0],
                   [0, 1]]])
    q = np.ones(S) / S
    m = MDP(P=P,
            r=np.array([[[1, 0],
                         [0, 0]],
                        [[0, 0],
                         [0, 0]]]),
            q=q)

    # Number of epochs
    T = 1000

    # Initialize mu
    mu = np.zeros((T+1, H, S, A))
    mu_average = np.zeros((T+1, H, S, A))
    mu_average[0] = mu[0] = initial_mu(H, S, A, P)

    # Initialize v
    v = np.zeros((T+1, H, S))
    v_average = np.zeros((T+1, H, S))
    v_average[0] = v[0] = initial_v(H, S)

    np.random.seed(1)

    for t in range(T):
        states = np.zeros(H+1, dtype=int)
        actions = np.zeros(H, dtype=int)
        rewards = np.zeros(H, dtype=float)

        states[0] = m.get_initial_state()
        for h in range(H):
            actions[h] = np.random.choice(actions, p=mu_average[h, states[h]])[0]
            states[h+1], rewards[h] = m.perform_action(actions[h])


