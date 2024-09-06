import numpy as np
from itertools import product, combinations
from typing import Set, List, Tuple, FrozenSet
from functools import lru_cache
import logging
from unified_mdp import UnifiedMDP, UnifiedState, ActionSpace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_ITERATIONS = 1000
GAMMA = 0.95
CONVERGENCE_THRESHOLD = 1e-6

from itertools import product

class SubgoalMDP(UnifiedMDP):
    def __init__(self, det_mdp, subgoals):
        self.det_mdp = det_mdp
        self.subgoals = [s for s in subgoals if s != det_mdp.G]  # remove goal state from subgoals
        self.n_subgoals = len(self.subgoals)
        self.A = list(det_mdp.A) 
        self.n_actions = len(self.A)
        
        # create extended state space
        self.S = list(product(range(det_mdp.n_states), product([0, 1], repeat=self.n_subgoals)))
        self.n_states = len(self.S)
        
        self.s0 = self.S.index((det_mdp.s0, tuple([0] * self.n_subgoals)))
        self.G = [i for i, (s, v) in enumerate(self.S) if s == det_mdp.G]
        
        self.P = {}
        self.R = np.zeros(self.n_states)
        
        self._build_transition_and_reward()

        state_space = UnifiedState(self.S, is_discrete=True)
        action_space = ActionSpace(self.A)
        super().__init__(state_space, action_space, self._transition_func, self._reward_func, det_mdp.gamma)

        print(f"SubgoalMDP initialized with {self.n_states} states and {self.n_subgoals} subgoals")
        print(f"Initial state: {self.S[self.s0]}")
        print(f"Subgoals: {self.subgoals}")

    def _build_transition_and_reward(self):
        for i, (s, v) in enumerate(self.S):
            self.P[i] = {}
            for a in self.A:
                if s in self.det_mdp.P and a in self.det_mdp.P[s]:
                    s_next = self.det_mdp.P[s][a]
                    new_v = list(v)
                    if s_next in self.subgoals:
                        new_v[self.subgoals.index(s_next)] = 1
                    new_v = tuple(new_v)
                    j = self.S.index((s_next, new_v))
                    
                    self.P[i][a] = j
                    
                    # set reward
                    if s_next == self.det_mdp.G:
                        if all(new_v):
                            self.R[i] = 100  # positive reward for reaching goal with all subgoals
                        else:
                            self.R[i] = -10  # negative reward for reaching goal without all subgoals
                    else:
                        self.R[i] = 0  # no reward for non-goal states
                else:
                    # if the action is not available in the original MDP, create a self-loop
                    self.P[i][a] = i
                    self.R[i] = 0

    def _transition_func(self, state, action):
        i = self.S.index(state.data)
        j = self.P[i][action]
        return UnifiedState(self.S[j], is_discrete=True)

    def _reward_func(self, state, action, next_state):
        i = self.S.index(state.data)
        return self.R[i]

    def get_optimal_policy(self, V):
        policy = {}
        for s in range(self.n_states):
            if s in self.G:
                policy[s] = self.A[0]  # any action for goal states
            elif s in self.P:
                policy[s] = max(self.P[s], key=lambda a: V[self.P[s][a]])
            else:
                print(f"Warning: No actions available for state {s}")
        return policy
    
    def print_rewards(self):
        print("Reward structure:")
        for i, (s, v) in enumerate(self.S):
            print(f"State {i} (MDP state: {s}, Subgoals: {v}): Reward = {self.R[i]}")


def find_maximal_achievable_subsets(det_mdp, human_subgoals):
    B = frozenset(s for s in human_subgoals if s != det_mdp.G)
    I: List[Set] = []
    fringe: List[Set] = [set(B)]
        
    print(f"Initial fringe: {fringe}")
        
    while fringe:
        current_set = fringe.pop(0)
        print(f"Current Set: {current_set}")
        if check_achievability(det_mdp, frozenset(current_set)):
            I.append(current_set)
            print(f"Achievable set found: {current_set}")
        else:
            for s in current_set:
                new_set = current_set - {s}
                if new_set and new_set not in fringe and not any(new_set.issubset(achieved_set) for achieved_set in I):
                    fringe.append(new_set)
        
    maximal_sets = [s for s in I if not any(s < t for t in I)]
    
    print(f"Maximal achievable subsets: {maximal_sets}")
    return maximal_sets if maximal_sets else None  # Return None if no maximal sets found

@lru_cache(maxsize=None)
def check_achievability(det_mdp, subset: FrozenSet) -> bool:
    subgoal_mdp = SubgoalMDP(det_mdp, subset)
    V = value_iteration(subgoal_mdp)
    return is_policy_achieving_subgoals(subgoal_mdp, V)

def value_iteration(subgoal_mdp, epsilon=CONVERGENCE_THRESHOLD, max_iterations=MAX_ITERATIONS):
    V = np.zeros(subgoal_mdp.n_states)
    for _ in range(max_iterations):
        delta = 0
        for s in range(subgoal_mdp.n_states):
            if s in subgoal_mdp.G:
                continue
            v = V[s]
            if s in subgoal_mdp.P and subgoal_mdp.P[s]:
                V[s] = max(subgoal_mdp.R[s] + GAMMA * V[subgoal_mdp.P[s][a]] for a in subgoal_mdp.P[s])
            delta = max(delta, abs(v - V[s]))
        if delta < epsilon:
            break
    return V

def is_policy_achieving_subgoals(subgoal_mdp, V):
    policy = subgoal_mdp.get_optimal_policy(V)
    current_state = subgoal_mdp.s0
    visited_subgoals = set()
    
    for _ in range(MAX_ITERATIONS):
        if current_state in subgoal_mdp.G:
            break
        if current_state not in policy:
            print(f"Warning: No policy for state {current_state}")
            break
        
        action = policy[current_state]
        next_state = subgoal_mdp.P[current_state][action]
        
        # Unpack the state
        mdp_state, subgoal_status = subgoal_mdp.S[next_state]
        
        # Check if we've visited a new subgoal
        for i, visited in enumerate(subgoal_status):
            if visited:
                visited_subgoals.add(subgoal_mdp.subgoals[i])
        
        current_state = next_state
    
    all_subgoals_visited = set(subgoal_mdp.subgoals) == visited_subgoals
    print(f"All subgoals visited: {all_subgoals_visited}")
    print(f"Visited subgoals: {visited_subgoals}")
    print(f"All subgoals: {set(subgoal_mdp.subgoals)}")
    return all_subgoals_visited

if __name__ == "__main__":
    # Test code can be added here
    pass