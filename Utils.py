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


def extract_path(mdp, V, start_state, target_state):
    current_state = start_state
    path = [current_state]
    while current_state[0] != target_state:  # Compare only the position part of the state
        best_action = None
        best_value = float('-inf')
        best_next_state = None
        for action in mdp.get_actions():
            for next_state in mdp.get_state_space():
                prob = mdp.get_transition_probability(current_state, action, next_state)
                if prob > 0:
                    value = V[mdp.get_state_hash(next_state)]
                    if value > best_value:
                        best_value = value
                        best_action = action
                        best_next_state = next_state
        if best_next_state is None:
            print(f"No valid next state found from {current_state}")
            break
        current_state = best_next_state
        path.append(current_state)
        if len(path) > 100:  # Prevent infinite loops
            print("Path extraction stopped due to length limit")
            break
    return path

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