import numpy as np


class MDP(object):
    def __init__(self, P: np.ndarray, r, q: np.ndarray):
        self.transition_probability_matrix = P
        if callable(r):
            self.reward_function = r
        elif '__getattribute__' in dir(r):
            self.reward_function = lambda i, a, j: r[i, a, j]
        self.initial_state_distribution = q
        self.states = np.arange(P.shape[0])
        self.actions = np.arange(P.shape[1])
        self.curr_state = None

    def perform_action(self, action):
        if self.curr_state is None:
            raise RuntimeError("Must call get_initial_state() before calling perform_action()")
        old_state = self.curr_state
        self.curr_state = np.random.choice(self.states,
                                           self.transition_probability_matrix[self.curr_state][action])[0]
        reward = self.reward_function(old_state, action, self.curr_state)
        return self.curr_state, reward

    def get_initial_state(self):
        self.curr_state = np.random.choice(self.states,
                                           self.initial_state_distribution)[0]
        return self.curr_state
