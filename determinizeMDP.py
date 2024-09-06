import numpy as np
from functools import lru_cache
from unified_mdp import UnifiedMDP, UnifiedState, ActionSpace

class DeterminizedMDP:
    def __init__(self, mdp, gamma=0.99):
        self.original_mdp = mdp
        self.gamma = gamma  # Store gamma as an attribute
        self._initialize_state_space(mdp)
        self.s0 = self.get_initial_state(mdp)
        self.G = self.get_goal_state(mdp)
        self.A = []
        self.P = {s: {} for s in self.S}
        self.R = np.zeros(self.n_states)
        self.constraints = set()
        self._create_deterministic_actions()

    def _initialize_state_space(self, mdp):
        if hasattr(mdp, 'n_states'):
            self.n_states = mdp.n_states
            self.S = range(self.n_states)
        elif hasattr(mdp, 'state_space'):
            self.n_states = len(mdp.state_space.data)
            self.S = range(self.n_states)
        else:
            raise ValueError("Unable to determine the number of states in the MDP")
        
        print(f"Initialized DeterminizedMDP with {self.n_states} states")

    def get_initial_state(self, mdp):
        if hasattr(mdp, 'initial_state'):
            return mdp.initial_state if isinstance(mdp.initial_state, int) else self.state_to_index(mdp.initial_state.data)
        elif hasattr(mdp, 's0'):
            return mdp.s0
        else:
            print("Warning: No initial state found. Using state 0 as initial state.")
            return 0

    def get_goal_state(self, mdp):
        if hasattr(mdp, 'goal_state'):
            return mdp.goal_state if isinstance(mdp.goal_state, int) else self.state_to_index(mdp.goal_state.data)
        elif hasattr(mdp, 'G'):
            return mdp.G
        elif hasattr(mdp, 'goal_locs') and mdp.goal_locs:
            try:
                goal_index = self.state_to_index(mdp.goal_locs[0])
                print(f"Goal state found at index: {goal_index}")
                return goal_index
            except Exception as e:
                print(f"Error finding goal state: {e}")
                print(f"Goal location: {mdp.goal_locs[0]}")
                print(f"State space sample: {mdp.state_space.data[:5]}")
                print("Using last state as goal state.")
                return self.n_states - 1
        else:
            print(f"Warning: No goal state found. Using state {self.n_states - 1} as goal state.")
            return self.n_states - 1

    def state_to_index(self, state):
        if isinstance(state, (int, np.integer)):
            return state
        elif hasattr(self.original_mdp, 'state_space'):
            state_array = np.array(state)
            for i, s in enumerate(self.original_mdp.state_space.data):
                if np.array_equal(s, state_array):
                    return i
            raise ValueError(f"State {state} not found in state space")
        else:
            raise ValueError(f"Unable to convert state {state} to index")

    def _create_deterministic_actions(self):
        for s in self.S:
            state = self.get_state_from_index(s)
            for a in self.get_actions(state):
                try:
                    next_state = self.get_next_state(state, a)
                    s_next = self.state_to_index(next_state.data)
                    self.A.append((a, s_next))
                    self.P[s][(a, s_next)] = s_next
                except Exception as e:
                    print(f"Error in transition for state {state}, action {a}: {e}")
            
            try:
                self.R[s] = self.get_reward(state)
            except Exception as e:
                print(f"Error in reward function for state {state}: {e}")
                self.R[s] = 0  # Assign a default reward if there's an error
        
        try:
            goal_state = self.get_state_from_index(self.G)
            self.R[self.G] = self.get_reward(goal_state)
        except Exception as e:
            print(f"Error in reward function for goal state: {e}")
            self.R[self.G] = 1  # Assign a default positive reward for the goal state
        
        for s in self.S:
            if not self.P[s]:
                default_action = self.get_default_action()
                self.P[s] = {(default_action, s): s}

        print(f"Created {len(self.A)} deterministic actions")

    def get_state_from_index(self, index):
        if hasattr(self.original_mdp, 'state_space'):
            return self.original_mdp.state_space.data[index]
        else:
            return index

    def get_actions(self, state):
        if hasattr(self.original_mdp, 'action_space'):
            return self.original_mdp.action_space.actions
        elif hasattr(self.original_mdp, 'A'):
            return self.original_mdp.A
        else:
            raise ValueError("Unable to determine the action space of the MDP")

    def get_next_state(self, state, action):
        if hasattr(self.original_mdp, '_transition_func'):
            return self.original_mdp._transition_func(UnifiedState(state, is_discrete=True), action)
        elif hasattr(self.original_mdp, 'P'):
            s = self.state_to_index(state)
            s_next = self.original_mdp.P[s][action]
            return self.get_state_from_index(s_next)
        else:
            raise ValueError("Unable to determine the transition function of the MDP")

    def get_reward(self, state):
        try:
            if hasattr(self.original_mdp, '_reward_func'):
                return self.original_mdp._reward_func(UnifiedState(state, is_discrete=True), None, None)
            elif hasattr(self.original_mdp, 'R'):
                s = self.state_to_index(state)
                return self.original_mdp.R[s]
            else:
                raise ValueError("Unable to determine the reward function of the MDP")
        except Exception as e:
            print(f"Error in get_reward for state {state}: {e}")
            return 0  # Return a default reward of 0 if there's an error

    def get_default_action(self):
        if hasattr(self.original_mdp, 'action_space'):
            return self.original_mdp.action_space.actions[0]
        elif hasattr(self.original_mdp, 'A'):
            return self.original_mdp.A[0]
        else:
            return 0  # Default to action 0 if no action space is found

    def copy(self):
        new_mdp = DeterminizedMDP(self.original_mdp)
        new_mdp.constraints = self.constraints.copy()
        return new_mdp

    def add_constraint(self, state):
        self.constraints.add(self.state_to_index(state))

    def is_valid_state(self, state):
        return self.state_to_index(state) not in self.constraints

    @lru_cache(maxsize=None)
    def value_iteration(self, gamma=0.95, epsilon=1e-6):
        V = np.zeros(self.n_states)
        V[self.G] = self.R[self.G]
        while True:
            delta = 0
            for s in self.S:
                if s == self.G or not self.is_valid_state(self.get_state_from_index(s)):
                    continue
                v = V[s]
                if self.P[s]:
                    V[s] = max(self.R[s] + gamma * V[self.P[s][a]] 
                               for a in self.P[s] 
                               if self.is_valid_state(self.get_state_from_index(self.P[s][a])))
                delta = max(delta, abs(v - V[s]))
            if delta < epsilon:
                break
        return V

    def get_optimal_policy(self, V, gamma=0.95):
        policy = {}
        for s in self.S:
            if s == self.G or not self.is_valid_state(self.get_state_from_index(s)):
                continue
            if self.P[s]:
                policy[s] = max(self.P[s].keys(),
                                key=lambda a: self.R[s] + gamma * V[self.P[s][a]] 
                                if self.is_valid_state(self.get_state_from_index(self.P[s][a])) else float('-inf'))
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
