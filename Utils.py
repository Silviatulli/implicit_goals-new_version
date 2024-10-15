from itertools import chain, combinations
import numpy as np

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


def sparse_value_iteration(mdp, epsilon=1e-6, max_iterations=1000):
       V = np.zeros(mdp.n_states)
       for _ in range(max_iterations):
           V_prev = V.copy()
           for s in range(mdp.n_states):
               V[s] = max(
                   sum(mdp.P(s, a, s_next) * (mdp.R(s, a, s_next) + mdp.gamma * V_prev[s_next])
                       for s_next in mdp.get_next_states(s, a))
                   for a in range(mdp.n_actions)
               )
           if np.max(np.abs(V - V_prev)) < epsilon:
               break
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
