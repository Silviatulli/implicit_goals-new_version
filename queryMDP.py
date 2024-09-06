import numpy as np
from itertools import combinations
from determinizeMDP import DeterminizedMDP
from algorithm1 import find_maximal_achievable_subsets
import random
from unified_mdp import UnifiedMDP, UnifiedState, ActionSpace

class QueryMDP(UnifiedMDP):
    def __init__(self, bottlenecks, achievable_subsets, robot_mdp):
        self.B = bottlenecks
        self.I = achievable_subsets if achievable_subsets is not None else []
        self.robot_mdp = robot_mdp
        self.n_bottlenecks = len(self.B)
        self.gamma = 0.99
        self.C_Q = -10
        
        self.A = list(range(self.n_bottlenecks)) + ['terminate']
        
        # Pre-compute achievable rewards
        self.achievable_rewards = self._precompute_achievable_rewards()
        
        state_space = UnifiedState(range(2**self.n_bottlenecks), is_discrete=True)
        action_space = ActionSpace(self.A)
        
        super().__init__(state_space, action_space, self._transition_func, self._reward_func, self.gamma)

        print(f"QueryMDP initialized with {2**self.n_bottlenecks} states and {len(self.A)} actions.")
        print(f"Actions: {self.A}")

    def _transition_func(self, state, action):
        if action == 'terminate':
            return state
        return UnifiedState(state.data | (1 << action), is_discrete=True)

    def _reward_func(self, state, action, next_state):
        if action == 'terminate':
            return self._compute_terminal_reward(state.data)
        return self.C_Q

    def _compute_terminal_reward(self, state):
        visited_bottlenecks = frozenset(i for i in range(self.n_bottlenecks) if state & (1 << i))
        return self.achievable_rewards.get(visited_bottlenecks, 0)

    def _precompute_achievable_rewards(self):
        rewards = {}
        if not self.I:
            print("Warning: No achievable subsets found.")
            return rewards
        for I in self.I:
            if isinstance(self.robot_mdp, DeterminizedMDP):
                constrained_mdp = self.robot_mdp.copy()
            else:
                constrained_mdp = DeterminizedMDP(self.robot_mdp)
            for state in I:
                constrained_mdp.add_constraint(state)
            V = constrained_mdp.value_iteration()
            initial_state_value = V[self.robot_mdp.s0]
            max_possible_value = self.robot_mdp.n_states
            normalized_reward = initial_state_value / max_possible_value
            scaled_reward = normalized_reward * (abs(self.C_Q) / 10)
            rewards[frozenset(I)] = scaled_reward
        print(f"Precomputed {len(rewards)} achievable rewards.")
        return rewards

    def value_iteration(self, epsilon=1e-6, max_iterations=1000):
        V = np.zeros(2**self.n_bottlenecks)
        for iteration in range(max_iterations):
            delta = 0
            for state in range(2**self.n_bottlenecks):
                v = V[state]
                q_values = [self._reward_func(UnifiedState(state, is_discrete=True), action, 
                                              self._transition_func(UnifiedState(state, is_discrete=True), action)) + 
                            self.gamma * V[self._transition_func(UnifiedState(state, is_discrete=True), action).data]
                            for action in self.A]
                V[state] = max(q_values)
                delta = max(delta, abs(v - V[state]))
            if delta < epsilon:
                print(f"Value iteration converged after {iteration+1} iterations.")
                break
        return V

    def get_optimal_policy(self, V):
        policy = {}
        for state in range(2**self.n_bottlenecks):
            q_values = [self._reward_func(UnifiedState(state, is_discrete=True), action, 
                                          self._transition_func(UnifiedState(state, is_discrete=True), action)) + 
                        self.gamma * V[self._transition_func(UnifiedState(state, is_discrete=True), action).data]
                        for action in self.A]
            policy[state] = self.A[np.argmax(q_values)]
        return policy

# The optimal_query_strategy function remains mostly the same, with minor adjustments:

def optimal_query_strategy(robot_mdp, human_bottlenecks, robot_bottlenecks, query_cost=1):
    different_bottlenecks = list(set(human_bottlenecks) - set(robot_bottlenecks))
    achievable_subsets = find_maximal_achievable_subsets(robot_mdp, different_bottlenecks)
    
    print(f"Different bottlenecks: {different_bottlenecks}")
    print(f"Achievable subsets: {achievable_subsets}")
    
    if not achievable_subsets:
        print("No achievable subsets found.")
        return [], 0

    query_mdp = QueryMDP(different_bottlenecks, achievable_subsets, robot_mdp)
    
    V = query_mdp.value_iteration()
    policy = query_mdp.get_optimal_policy(V)
    
    current_state = 0  # Start with no bottlenecks visited
    queried_bottlenecks = []
    total_cost = 0
    
    while True:
        action = policy[current_state]
        if action == 'terminate':
            break
        
        bottleneck = different_bottlenecks[action]
        
        response = random.choice(['y', 'n'])
        total_cost += query_cost
        
        if response == 'y':
            current_state |= (1 << action)
            queried_bottlenecks.append(bottleneck)
        
        if query_mdp._compute_terminal_reward(current_state) != query_mdp.C_Q:
            break
    
    print(f"Queried bottlenecks: {queried_bottlenecks}")
    print(f"Total cost: {total_cost}")
    
    return queried_bottlenecks, total_cost