from itertools import chain, combinations
import numpy as np

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
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


