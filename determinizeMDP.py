import numpy as np
from functools import lru_cache

class DeterminizedMDP:
    def __init__(self, mdp):
        self.original_mdp = mdp
        self.S = range(mdp.n_states)
        self.s0 = mdp.initial_state 
        self.G = mdp.goal_state 
        self.n_states = mdp.n_states
        self.A = []
        self.P = {s: {} for s in self.S}
        self.R = np.zeros(self.n_states)
        self.constraints = set()
        self._create_deterministic_actions()

    def _create_deterministic_actions(self):
        for s in self.S:
            for a in range(self.original_mdp.n_actions):
                for s_next in self.S:
                    if self.original_mdp.P[s, a, s_next] > 0:
                        new_action = (a, s_next)
                        self.A.append(new_action)
                        self.P[s][new_action] = s_next
            self.R[s] = self.original_mdp.R[s]
        
        self.R[self.G] = self.original_mdp.R[self.G]
        
        for s in self.S:
            if not self.P[s]:
                self.P[s] = {(0, s): s}

    def add_constraint(self, state):
        self.constraints.add(state)

    def is_valid_state(self, state):
        return state not in self.constraints

    def value_iteration(self, gamma=0.95, epsilon=1e-6):
        V = np.zeros(self.n_states)
        V[self.G] = self.R[self.G]
        while True:
            delta = 0
            for s in self.S:
                if s == self.G or not self.is_valid_state(s):
                    continue
                v = V[s]
                if self.P[s]:
                    V[s] = max(self.R[s] + gamma * V[self.P[s][a]] 
                               for a in self.P[s] 
                               if self.is_valid_state(self.P[s][a]))
                delta = max(delta, abs(v - V[s]))
            if delta < epsilon:
                break
        return V

    def get_optimal_policy(self, V, gamma=0.95):
        policy = {}
        for s in self.S:
            if s == self.G or not self.is_valid_state(s):
                continue
            if self.P[s]:
                policy[s] = max(self.P[s].keys(),
                                key=lambda a: self.R[s] + gamma * V[self.P[s][a]] 
                                if self.is_valid_state(self.P[s][a]) else float('-inf'))
        return policy
    

def find_bottleneck_states(determinized_mdp, n=100, p=0.01, epsilon=1e-6, gamma=0.99):
    n_states = determinized_mdp.n_states
    goal_state = determinized_mdp.G
    initial_state = determinized_mdp.s0
    
    @lru_cache(maxsize=None)
    def value_iteration(R_tuple):
        R = np.array(R_tuple)
        V = np.zeros(n_states)
        while True:
            delta = 0
            for s in determinized_mdp.S:
                if s == goal_state:
                    V[s] = R[s]
                    continue
                v = V[s]
                if determinized_mdp.P[s]:
                    V[s] = R[s] + gamma * max(V[determinized_mdp.P[s][a]] for a in determinized_mdp.P[s])
                delta = max(delta, abs(v - V[s]))
            if delta < epsilon:
                break
        return V

    bottlenecks = []
    target_states = [s for s in determinized_mdp.S if s not in {initial_state, goal_state}]

    for s in target_states:
        R_modified = np.zeros(n_states)
        R_modified[s] = -n
        R_modified[goal_state] = p
        V = value_iteration(tuple(R_modified))
        if V[initial_state] <= 0:
            bottlenecks.append(s)

    return bottlenecks
