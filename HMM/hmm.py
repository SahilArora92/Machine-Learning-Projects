from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(X_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    # TODO
    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array delta[i, t] = P(X_t = s_i, Z_1:Z_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        alpha = alpha.T
        alpha[0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        for t in range(1, L):
            alpha[t] = self.B[:, self.obs_dict[Osequence[t]]] * np.dot(alpha[t-1], self.A)
        ###################################################
        return alpha.T

    # TODO:
    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array gamma[i, t] = P(Z_t+1:Z_T | X_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        beta[:, -1:] = 1
        for t in reversed(range(L-1)):
            for n in range(S):
                beta[n, t] = np.sum(self.A[n, :] * self.B[:, self.obs_dict[Osequence[t+1]]] * beta[:, t + 1])
        ###################################################
        return beta

    # TODO:
    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(Z_1:Z_T | λ)
        """
        prob = 0
        ###################################################
        alpha = self.forward(Osequence)
        prob = np.sum(alpha, axis=0)[-1]
        ###################################################
        return prob

    # TODO:
    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(X_t = i | O, λ)
        """
        prob = 0
        ###################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = np.sum(alpha, axis=0)[-1]
        prob = np.multiply(alpha, beta)/seq_prob
        ###################################################
        return prob

    # TODO:
    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """

        ###################################################
        L = len(Osequence)
        S = len(self.pi)
        path = np.zeros(L, dtype=np.int32)
        delta = np.zeros((L, S))
        cap_delta = np.zeros((L, S))

        delta[0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        for t in range(1, L):
            for s in range(S):
                delta[t, s] = self.B[s, self.obs_dict[Osequence[t]]] * np.max(self.A[:, s] * delta[t - 1])
                cap_delta[t, s] = np.argmax(self.A[:, s] * delta[t - 1])

        path[L-1] = np.argmax(delta[L-1])
        for t in range(L - 2, -1, -1):
            path[t] = cap_delta[t + 1, path[t + 1]]

        # map actual state path from state_dict
        state_dict = {y: x for x, y in self.state_dict.items()}
        path = np.vectorize(state_dict.get)(path)

        ###################################################
        return path.tolist()
