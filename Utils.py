from itertools import chain, combinations
import numpy as np
from typing import Any, Dict, List, Tuple
from scipy.sparse import csr_matrix, lil_matrix
import warnings
import time


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def ValueIteration(mdp, epsilon=0.001):
    V = {mdp.get_state_hash(s): 0 for s in mdp.get_state_space()}
    # print(mdp.discount)
    while True:
        delta = 0
        for s in mdp.get_state_space():
            s_hash = mdp.get_state_hash(s)
            v = V[s_hash]
            V[s_hash] = max([mdp.discount * sum([mdp.get_transition_probability(s, a, s_prime) * (mdp.get_reward(s, a, s_prime) + V[mdp.get_state_hash(s_prime)]) for s_prime in mdp.get_state_space()]) for a in mdp.get_actions()])
            delta = max(delta, abs(v - V[s_hash]))
        # print(delta)
        if delta < epsilon:
            break
    return V


def get_policy(mdp, V):
    policy = {}
    for s in mdp.get_state_space():
        s_hash = mdp.get_state_hash(s)
        best_action = None
        best_q_value = None
        for act in mdp.get_actions():
            current_q_value = sum([mdp.get_transition_probability(s, act, s_prime) * (mdp.get_reward(s, act, s_prime) + V[mdp.get_state_hash(s_prime)]) for s_prime in mdp.get_state_space()])
            if best_action is None:
                best_action = act
                best_q_value = current_q_value
            elif current_q_value > best_q_value:
                best_action = act
                best_q_value = current_q_value
        policy[s_hash] = best_action
    return policy



def vectorized_value_iteration(mdp, epsilon=0.001, max_iterations=1000):
    # Initialize value function
    V = {mdp.get_state_hash(s): 0 for s in mdp.get_state_space()}
    states = list(mdp.get_state_space())
    n_states = len(states)
    n_actions = len(mdp.get_actions())
    
    # Create mappings for faster lookups
    state_to_idx = {mdp.get_state_hash(s): i for i, s in enumerate(states)}
    
    # Pre-compute transition probabilities and rewards
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions, n_states))
    
    for i, s in enumerate(states):
        for a_idx, a in enumerate(mdp.get_actions()):
            for j, s_next in enumerate(states):
                #if mdp.get_transition_probability(s, a, s_next) > 0:
                #    print(s, a, s_next, mdp.get_reward(s, a, s_next))
                #    print(s, a, s_next, mdp.get_transition_probability(s, a, s_next))
                P[i, a_idx, j] = mdp.get_transition_probability(s, a, s_next)
                R[i, a_idx, j] = mdp.get_reward(s, a, s_next)

    V_array = np.zeros(n_states)
    
    for _ in range(max_iterations):
        V_prev = V_array.copy()
        
        # Vectorized Bellman update
        Q = np.sum(P * (R + mdp.discount * V_prev), axis=2)
        V_array = np.max(Q, axis=1)
        delta = np.max(np.abs(V_array - V_prev))
        # print(delta)
        if delta < epsilon:
            break
    
    # Convert back to dictionary
    for s, v in zip(states, V_array):
        V[mdp.get_state_hash(s)] = v
    
    return V


def optimized_value_iteration(mdp, epsilon=1e-6, max_iterations=1000):
    V = np.zeros(mdp.n_states)
    for _ in range(max_iterations):
        V_new = np.zeros_like(V)
        for s, state in enumerate(mdp.state_space):
            if mdp.is_terminal(state):
                V_new[s] = 0
                continue
            max_value = float('-inf')
            for action in mdp.action_space:
                value = sum(
                    mdp.get_transition_prob(state, action, next_state) *
                    (mdp.get_reward(state, action, next_state) + mdp.gamma * V[mdp.state_to_index[mdp.state_to_tuple(next_state)]])
                    for next_state in mdp.get_next_states(state, action)
                )
                max_value = max(max_value, value)
            V_new[s] = max_value
        if np.max(np.abs(V - V_new)) < epsilon:
            return V_new
        V = V_new
    return V

def robust_vectorized_value_iteration(mdp, epsilon=1e-6, max_iterations=1000):
    # Initialize value function
    V = {mdp.get_state_hash(s): 0 for s in mdp.get_state_space()}
    states = list(mdp.get_state_space())
    n_states = len(states)
    n_actions = len(mdp.get_actions())
    
    #print(f"Number of states: {n_states}")
    #print(f"Number of actions: {n_actions}")
    
    if n_states == 0:
        raise ValueError("The MDP has no states.")
    if n_actions == 0:
        raise ValueError("The MDP has no actions.")
    
    # Create mappings for faster lookups
    state_to_idx = {mdp.get_state_hash(s): i for i, s in enumerate(states)}
    
    # Pre-compute transition probabilities and rewards
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions, n_states))
    
    for i, s in enumerate(states):
        for a_idx, a in enumerate(mdp.get_actions()):
            for j, s_next in enumerate(states):
                P[i, a_idx, j] = mdp.get_transition_probability(s, a, s_next)
                R[i, a_idx, j] = mdp.get_reward(s, a, s_next)
    
    #print(f"Shape of P: {P.shape}")
    #print(f"Shape of R: {R.shape}")
    
    if np.all(P == 0):
        raise ValueError("All transition probabilities are zero.")

    V_array = np.zeros(n_states)
    
    for iteration in range(max_iterations):
        V_prev = V_array.copy()
        
        # Vectorized Bellman update
        Q = np.sum(P * (R + mdp.discount * V_prev), axis=2)
        
        #print(f"Iteration {iteration + 1}")
        #print(f"Shape of Q: {Q.shape}")
        #print(f"Min Q: {np.min(Q)}, Max Q: {np.max(Q)}")
        
        # Handle states with no valid actions
        valid_actions = np.any(P > 0, axis=2)
        if not np.any(valid_actions):
            raise ValueError("No valid actions found for any state.")
        
        try:
            V_array = np.where(
                np.any(valid_actions, axis=1),
                np.max(np.where(valid_actions, Q, -np.inf), axis=1),
                V_prev
            )
        except ValueError as e:
            #print(f"Error in iteration {iteration + 1}: {e}")
            #print(f"Shape of valid_actions: {valid_actions.shape}")
            #print(f"Number of states with valid actions: {np.sum(np.any(valid_actions, axis=1))}")
            #print(f"Q values for states with valid actions:")
            for s in range(n_states):
                if np.any(valid_actions[s]):
                    print(f"State {s}: {Q[s][valid_actions[s]]}")
            raise
        
        delta = np.max(np.abs(V_array - V_prev))
        #print(f"Delta: {delta}")
        
        if delta < epsilon:
            print(f"Converged after {iteration + 1} iterations.")
            break
    
    # Convert back to dictionary
    for s, v in zip(states, V_array):
        V[mdp.get_state_hash(s)] = v
    
    return V

def get_robust_policy(mdp, V):
    policy = {}
    for s in mdp.get_state_space():
        s_hash = mdp.get_state_hash(s)
        best_action = None
        best_value = float('-inf')
        for a in mdp.get_actions():
            value = sum(
                mdp.get_transition_probability(s, a, s_next) *
                (mdp.get_reward(s, a, s_next) + mdp.discount * V[mdp.get_state_hash(s_next)])
                for s_next in mdp.get_state_space()
            )
            if value > best_value:
                best_value = value
                best_action = a
        policy[s_hash] = best_action
    return policy



def check_bottleneck_achievability(robot_mdp, robot_bottlenecks, human_bottlenecks):
    achievable_bottlenecks = []

    for human_bottleneck in human_bottlenecks:
        print(f"\nChecking achievability of human bottleneck: {human_bottleneck}")

        # Check if the human bottleneck is an obstacle in the robot's model
        if robot_mdp.map[human_bottleneck[0]] == -1:
            print(f"Bottleneck {human_bottleneck} is an obstacle in the robot's model")
            continue

        def bottleneck_reward(state, action, next_state):
            if next_state[0] == human_bottleneck[0]:
                return 1000
            elif robot_mdp.check_goal_reached(next_state[0]):
                return 500  # Smaller reward for reaching the goal
            return 0

        robot_mdp.reward_func = bottleneck_reward

        V = vectorized_value_iteration(robot_mdp)
        initial_state = robot_mdp.get_init_state()
        initial_state_hash = robot_mdp.get_state_hash(initial_state)

        print(f"Initial state: {initial_state}")
        print(f"Initial state hash: {initial_state_hash}")
        print(f"Value of initial state: {V[initial_state_hash]}")

        # Check if there's a path with significant positive value
        if V[initial_state_hash] > 10:  # Adjust this threshold as needed
            achievable_bottlenecks.append(human_bottleneck)
            print(f"Bottleneck {human_bottleneck} is achievable")
        else:
            print(f"Bottleneck {human_bottleneck} is NOT achievable")

        # Print some path information
        path = extract_path(robot_mdp, V, initial_state, human_bottleneck[0])
        print(f"Path to bottleneck: {path}")
        print(f"Path length: {len(path)}")

    return achievable_bottlenecks



def extract_path(mdp, V, start_state, target_state):
    current_state = start_state
    path = [current_state]
    visited = set()  # To detect cycles

    while current_state[0] != target_state:  # Compare only the position part of the state
        if tuple(current_state) in visited:
            print(f"Cycle detected at state {current_state}")
            break

        visited.add(tuple(current_state))
        best_action = None
        best_value = float('-inf')
        best_next_state = None

        for action in mdp.get_actions():
            for next_state in mdp.get_state_space():
                prob = mdp.get_transition_probability(current_state, action, next_state)
                if prob > 0:
                    reward = mdp.get_reward(current_state, action, next_state)
                    value = reward + mdp.discount * V[mdp.get_state_hash(next_state)]
                    if value > best_value:
                        best_value = value
                        best_action = action
                        best_next_state = next_state

        if best_next_state is None:
            print(f"No valid next state found from {current_state}")
            break

        print(f"Moving from {current_state} to {best_next_state} with action {best_action}")
        current_state = best_next_state
        path.append(current_state)

        if len(path) > 100:  # Prevent infinite loops
            print("Path extraction stopped due to length limit")
            break

    return path


def sparse_value_iteration(mdp: Any, max_iterations: int = 1000, epsilon: float = 1e-6) -> np.ndarray:
    states = mdp.get_state_space()
    actions = mdp.get_actions()
    n_states = len(states)
    n_actions = len(actions)

    print("Number of states", str(n_states))
    # Create state index mapping
    state_to_idx = {mdp.get_state_hash(state): idx for idx, state in enumerate(states)}
    
    # Initialize sparse transition matrices and reward vectors for each action
    P_sparse = []
    R = np.zeros((n_states, n_actions))
    start_time = time.time()
    # Build sparse matrices action by action to save memory
    for a_idx, action in enumerate(actions):
        # Use lil_matrix for efficient construction
        P_a = lil_matrix((n_states, n_states), dtype=np.float64)
        
        for s_idx, state in enumerate(states):
            # Get non-zero transitions for this state-action pair
            for next_state in states:
                prob = mdp.get_transition_probability(state, action, next_state)
                if prob > 0:
                    ns_idx = state_to_idx[mdp.get_state_hash(next_state)]
                    P_a[s_idx, ns_idx] = prob
                    R[s_idx, a_idx] += prob * mdp.get_reward(state, action, next_state)
        
        # Convert to CSR format for efficient operations
        P_sparse.append(P_a.tocsr())
    print (f"Time to build the sparse matrix: {time.time() - start_time}")

    # Value iteration
    start_time = time.time()
    V = np.zeros(n_states)
    for _ in range(max_iterations):
        V_old = V.copy()
        
        # Compute Q-values for each action using sparse matrix operations
        Q = np.zeros((n_states, n_actions))
        for a_idx in range(n_actions):
            Q[:, a_idx] = R[:, a_idx] + mdp.discount * P_sparse[a_idx].dot(V_old)
        
        # Update value function
        V = np.max(Q, axis=1)
        
        # Check convergence
        if np.max(np.abs(V - V_old)) < epsilon:
            break
    print(f"Time to run value iteration: {time.time() - start_time}")
    return V

def get_sparse_policy(mdp: Any, V: np.ndarray) -> Dict[str, str]:
    states = mdp.get_state_space()
    actions = mdp.get_actions()
    policy = {}
    
    for state in states:
        state_hash = mdp.get_state_hash(state)
        max_value = float('-inf')
        best_action = None
        
        for action in actions:
            value = 0
            # Only compute for non-zero probability transitions
            for next_state in states:
                prob = mdp.get_transition_probability(state, action, next_state)
                if prob > 0:
                    ns_idx = list(states).index(next_state)
                    reward = mdp.get_reward(state, action, next_state)
                    value += prob * (reward + mdp.discount * V[ns_idx])
            
            if value > max_value:
                max_value = value
                best_action = action
        
        policy[state_hash] = best_action
    
    return policy
