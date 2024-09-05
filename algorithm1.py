import numpy as np
from itertools import product
from typing import Set, List, Tuple
from functools import lru_cache
import logging

class SubgoalMDP:
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

    def _build_transition_and_reward(self):
        for i, (s, v) in enumerate(self.S):
            self.P[i] = {}
            for a in self.A:
                if s in self.det_mdp.P and a in self.det_mdp.P[s]:
                    s_next = self.det_mdp.P[s][a]
                    new_v = list(v)
                    if s in self.subgoals:
                        new_v[self.subgoals.index(s)] = 1
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



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_ITERATIONS = 1000
GAMMA = 0.95
CONVERGENCE_THRESHOLD = 1e-6


def find_maximal_achievable_subsets(det_mdp, human_subgoals):
    """
    Find maximal achievable subsets of subgoals.
    
    Args:
        det_mdp: The deterministic MDP.
        human_subgoals: The set of human-defined subgoals.
    
    Returns:
        A list of maximal achievable subsets of subgoals.
    """
    
    @lru_cache(maxsize=None)
    def check_achievability(subset: frozenset) -> bool:
        subgoal_mdp = SubgoalMDP(det_mdp, subset)
        V = value_iteration(subgoal_mdp)
        return is_policy_achieving_subgoals(subgoal_mdp, V)

    def value_iteration(subgoal_mdp):
        V = np.zeros(subgoal_mdp.n_states)
        for _ in range(MAX_ITERATIONS):
            delta = 0
            for s in range(subgoal_mdp.n_states):
                if s in subgoal_mdp.G:
                    continue
                v = V[s]
                if s in subgoal_mdp.P:
                    V[s] = max(subgoal_mdp.R[s] + GAMMA * V[subgoal_mdp.P[s][a]] for a in subgoal_mdp.P[s])
                delta = max(delta, abs(v - V[s]))
            if delta < CONVERGENCE_THRESHOLD:
                break
        return V

    def is_policy_achieving_subgoals(subgoal_mdp, V):
        policy = subgoal_mdp.get_optimal_policy(V)
        current_state = subgoal_mdp.s0
        visited_subgoals = set()
        
        for _ in range(MAX_ITERATIONS):
            if current_state in subgoal_mdp.G:
                break
            action = policy[current_state]
            next_state = subgoal_mdp.P[current_state][action]
            _, subgoals_visited = subgoal_mdp.S[next_state]
            visited_subgoals.update([s for s, v in zip(subgoal_mdp.subgoals, subgoals_visited) if v])
            current_state = next_state
        
        return set(subgoal_mdp.subgoals) == visited_subgoals

    logger.info(f"Goal: {det_mdp.G}")
    B = frozenset(s for s in human_subgoals if s != det_mdp.G)
    I: List[Set] = []
    fringe: List[Set] = [B]
    
    logger.info(f"Initial fringe: {fringe}")
    
    while fringe:
        current_set = fringe.pop(0)
        logger.debug(f"Current Set: {current_set}")
        if check_achievability(frozenset(current_set)):
            I.append(current_set)
        else:
            for s in current_set:
                new_set = current_set - {s}
                if new_set and new_set not in fringe and not any(new_set.issubset(achieved_set) for achieved_set in I):
                    fringe.append(new_set)
    
    maximal_sets = [s for s in I if not any(s < t for t in I)]
    
    return maximal_sets

# # Usage in main script
# if __name__ == "__main__":
#     from MDPgrid import GridWorldMDP
#     from determinizeMDP import DeterminizedMDP, find_bottleneck_states

#     grid = [
#         [1,  0,  3,  4,  5,  6,  7,  8],
#         [9,  10,  0,  0,  0,  0,  0, 0],
#         [12, 13,  0,  0,  0,  0,  0, 14],
#         [15, 16, 17, 18, 19, 20, 21, 22],
#         [23, 0,  0,  0,  0,  0,  24, 25],
#         [26, 0,  0,  0,  0,  0,  27, 28],
#         [29, 30, 31, 32, 33, 34, 35, 36]
#     ]
    
    
    
#     initial_state = 1
#     goal_state = 36

#     mdp = GridWorldMDP(grid, initial_state=initial_state, goal_state=goal_state)
#     det_mdp = DeterminizedMDP(mdp)


#     bottlenecks = [7,8] # this is 0-indexed

#     print("\nFinding Maximal Achievable Subsets:")
#     maximal_subsets = find_maximal_achievable_subsets(det_mdp, bottlenecks)
#     print("Maximal Achievable Subsets:")
#     for subset in maximal_subsets:
#         print([s + 1 for s in subset])  # Convert to 1-indexed states for display