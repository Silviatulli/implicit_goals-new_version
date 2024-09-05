import numpy as np
from itertools import combinations
from determinizeMDP import DeterminizedMDP
from algorithm1 import find_maximal_achievable_subsets
import random

import numpy as np
from itertools import combinations

class QueryMDP:
    def __init__(self, bottlenecks, achievable_subsets, robot_mdp):
        self.B = bottlenecks
        self.I = achievable_subsets
        self.robot_mdp = robot_mdp
        self.n_states = robot_mdp.n_states
        self.s0 = robot_mdp.s0
        self.G = robot_mdp.G
        self.S = self._generate_state_space()
        self.A = list(range(len(self.B))) + ['terminate']  # add 'terminate' action
        self.query_s0 = (frozenset(), frozenset())
        self.gamma = 0.99
        self.C_Q = -10
        
        self.T = self._generate_transition_function()
        self.R = self._generate_reward_function()

    def _generate_state_space(self):
        states = []
        for i in range(len(self.B) + 1):
            for K_I in combinations(self.B, i):
                for j in range(len(self.B) - i + 1):
                    for K_not_I in combinations(set(self.B) - set(K_I), j):
                        states.append((frozenset(K_I), frozenset(K_not_I)))
        return states

    def _generate_transition_function(self):
        T = {}
        for s in self.S:
            for a in self.A:
                T[s, a] = self._compute_transition(s, a)
        return T

    def _compute_transition(self, s, a):
        if a == 'terminate':
            return {s: 1.0}  # terminate action keeps the state the same
        
        K_I, K_not_I = s
        b = self.B[a]
        
        if b in K_I or b in K_not_I:
            return {s: 1.0}
        
        s1 = (K_I.union({b}), K_not_I)
        s2 = (K_I, K_not_I.union({b}))
        
        return {s1: 0.5, s2: 0.5}

    def _generate_reward_function(self):
        R = {}
        for s in self.S:
            for a in self.A:
                R[s, a] = self._compute_reward(s, a)
        return R

    def _compute_reward(self, s, a):
        if a == 'terminate':
            return self._compute_terminal_reward(s)
        
        K_I, K_not_I = s
        potential_I = K_I.union(set(self.B) - K_not_I)
        
        if K_I.union(K_not_I) == set(self.B):
            # we've queried all bottlenecks
            return self._compute_terminal_reward(s)
        else:
            # we haven't queried all bottlenecks yet
            return self.C_Q

    def _compute_terminal_reward(self, s):
        K_I, K_not_I = s
        potential_I = K_I.union(set(self.B) - K_not_I)
        
        if any(potential_I.issubset(I) for I in self.I):
            # achievable state
            return self._compute_achievable_reward(potential_I)
        else:
            # unachievable state
            return 0

    def _compute_achievable_reward(self, potential_I):
        # create a copy of the robot_mdp with the potential implicit subgoals as constraints
        constrained_mdp = DeterminizedMDP(self.robot_mdp.original_mdp)
        for state in potential_I:
            constrained_mdp.add_constraint(state)
        
        # compute the optimal policy for the constrained MDP
        V = constrained_mdp.value_iteration()
        
        # compute the value of the initial state under this policy
        initial_state_value = V[self.s0]
        
        # normalize the reward
        max_possible_value = self.n_states  # assuming reward of 1 for each state
        normalized_reward = initial_state_value / max_possible_value
        
        # scale the reward to be significantly smaller than |C_Q|
        scaled_reward = normalized_reward * (abs(self.C_Q) / 10)
        
        return scaled_reward

    def value_iteration(self):
        V = {s: 0 for s in self.S}
        theta = 0.0001
        while True:
            delta = 0
            for s in self.S:
                v = V[s]
                values = [sum(p * (self.R[s, a] + self.gamma * V[s_next])
                              for s_next, p in self.T[s, a].items())
                          for a in self.A
                          if (s, a) in self.T]  # only consider valid actions
                V[s] = max(values) if values else 0  # if no valid actions, keep value at 0
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        return V

    def get_optimal_policy(self, V):
        policy = {}
        for s in self.S:
            values = {a: sum(p * (self.R[s, a] + self.gamma * V[s_next])
                             for s_next, p in self.T[s, a].items())
                      for a in self.A
                      if (s, a) in self.T}  # only consider valid actions
            policy[s] = max(values, key=values.get) if values else 'terminate'  # if no valid actions, terminate
        return policy

def optimal_query_strategy(robot_mdp, human_bottlenecks, robot_bottlenecks, query_cost=1):
    different_bottlenecks = list(set(human_bottlenecks) - set(robot_bottlenecks))
    achievable_subsets = find_maximal_achievable_subsets(robot_mdp, different_bottlenecks)
    query_mdp = QueryMDP(different_bottlenecks, achievable_subsets, robot_mdp)
    V = query_mdp.value_iteration()
    policy = query_mdp.get_optimal_policy(V)
    
    current_state = query_mdp.query_s0
    queried_bottlenecks = []
    total_cost = 0
    
    while True:
        action = policy[current_state]
        if action == 'terminate':
            break
        
        bottleneck = different_bottlenecks[action]
        
        response = random.choice(['y', 'n'])
        total_cost += query_cost
        
        K_I, K_not_I = current_state
        if response == 'y':
            new_state = (K_I.union({bottleneck}), K_not_I)
            queried_bottlenecks.append(bottleneck)
        else:
            new_state = (K_I, K_not_I.union({bottleneck}))
        
        if query_mdp._compute_reward(new_state, 'terminate') != query_mdp.C_Q:
            break
        
        current_state = new_state
    
    return queried_bottlenecks, total_cost

def update_robot_policy(mdp, additional_bottlenecks):
    for state in additional_bottlenecks:
        mdp.add_constraint(state)
    V = mdp.value_iteration()  # run value iteration again after adding constraints
    return mdp.get_optimal_policy(V)
