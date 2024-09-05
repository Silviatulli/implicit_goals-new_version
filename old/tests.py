from gridMDP import GridWorldMDP
import copy
import random
import numpy as np



def determinize_model(mdp):
    determinized = copy.deepcopy(mdp)
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            max_prob_index = np.argmax(mdp.P[s, a])
            determinized.P[s, a] = np.zeros(mdp.n_states)
            determinized.P[s, a, max_prob_index] = 1.0
    return determinized

# def generate_determinized_human_models(mdp, n_models=5):
#     human_models = generate_human_models(mdp, n_models)
#     determinized_models = [determinize_model(model) for model in human_models]
#     return determinized_models


def generate_human_models(mdp, n_models=5):
        human_models = []
        for _ in range(n_models):
            human_model = copy.deepcopy(mdp)
            # Modify transition probabilities
            for s in range(mdp.n_states):
                for a in range(mdp.n_actions):
                    # Randomly adjust probabilities
                    probs = mdp.P[s, a].copy()
                    probs += np.random.normal(0, 0.1, mdp.n_states)
                    probs = np.clip(probs, 0, 1)
                    probs /= probs.sum()
                    human_model.P[s, a] = probs
            
            # Randomly block some states (representing misunderstanding of the environment)
            n_blocked = random.randint(1, mdp.n_states // 5)  # Block up to 20% of states
            blocked_states = random.sample(range(mdp.n_states), n_blocked)
            blocked_states = [s for s in blocked_states if s != mdp.initial_state and s != mdp.goal_state]
            for s in blocked_states:
                human_model.P[s, :, :] = 0
                human_model.P[:, :, s] = 0
            
            # Ensure the initial and goal states are preserved
            human_model.initial_state = mdp.initial_state
            human_model.goal_state = mdp.goal_state
            
            # Ensure there's always a path from initial to goal state
            if not is_goal_reachable(human_model):
                continue  # Discard this model and generate a new one
            
            human_models.append(human_model)
        
        return human_models

def is_goal_reachable(mdp):
    visited = set()
    stack = [mdp.initial_state]
    while stack:
        state = stack.pop()
        if state == mdp.goal_state:
            return True
        if state in visited:
            continue
        visited.add(state)
        for action in range(mdp.n_actions):
            next_states = np.where(mdp.P[state, action] > 0)[0]
            stack.extend(next_states)
    return False

def generate_determinized_human_models(mdp, n_models=5):
    human_models = []
    while len(human_models) < n_models:
        model = generate_human_models(mdp, 1)[0]
        determinized_model = determinize_model(model)
        if is_goal_reachable(determinized_model):
            human_models.append(determinized_model)
    return human_models



def print_model_summary(mdp):
    print(f"Number of states: {mdp.n_states}")
    print(f"Number of actions: {mdp.n_actions}")
    print(f"Initial state: {mdp.initial_state}")
    print(f"Goal state: {mdp.goal_state}")
    print("Transition probabilities (non-zero):")
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            for s_next in range(mdp.n_states):
                if mdp.P[s, a, s_next] >= 0:
                    print(f"P(s'={s_next}|s={s},a={a}) = {mdp.P[s, a, s_next]:.2f}")


def find_common_bottlenecks(mdp, determinized_models, threshold=0.9):
    all_bottlenecks = []
    for model in determinized_models:
        bottlenecks = model.find_bottleneck_states()
        all_bottlenecks.append(set(bottlenecks))
    
    common_bottlenecks = set.intersection(*all_bottlenecks)
    
    # Include states that are bottlenecks in at least threshold fraction of models
    if threshold < 1.0:
        bottleneck_counts = {state: sum(state in b for b in all_bottlenecks) for state in range(mdp.n_states)}
        threshold_count = threshold * len(determinized_models)
        common_bottlenecks.update(state for state, count in bottleneck_counts.items() if count >= threshold_count)
    
    return list(common_bottlenecks)




# Example usage
grid = [
    [1, 2, 3, 4, 0],
    [5, 0, 6, 0, 0],
    [7, 8, 9, 10, 11],
    [12, 13, 0, 0, 14],
    [15, 16, 17, 18, 19]
]

original_mdp = GridWorldMDP(grid, initial_state=17, goal_state=4)


print("Original MDP:")
print_model_summary(original_mdp)

# Generate determinized human models
determinized_models = generate_determinized_human_models(original_mdp, n_models=3)

print("\nDeterminized Human Models:")
for i, model in enumerate(determinized_models):
    print(f"\nModel {i+1}:")
    print_model_summary(model)

common_bottlenecks = find_common_bottlenecks(original_mdp, determinized_models)
print("Common bottleneck states:", [s+1 for s in common_bottlenecks])



V = determinized_models[1].value_iteration()
# print("Value function:")
# print(V)

policy = determinized_models[1].extract_policy(V)
determinized_models[1].visualize_grid(V, policy)

exit()
mdp = GridWorldMDP(grid, initial_state=17, goal_state=4)

V = mdp.value_iteration()
# print("Value function:")
# print(V)

policy = mdp.extract_policy(V)

print("\nOptimal Policy:")
print(policy)

# P_G = mdp.compute_all_P_G(policy)
# print("\nProbability of reaching goal state (P_G):")
# print(P_G)





print("\nExample traces:")
for i in range(5):
    print(f"\nTrace {i+1}:")
    trace = mdp.generate_trace(policy, mdp.initial_state)
    mdp.print_trace(trace)

n_simulations = 1000
total_steps = 0
successful_runs = 0

for _ in range(n_simulations):
    trace = mdp.generate_trace(policy, mdp.initial_state)
    if trace[-1][0] == mdp.goal_state:
        total_steps += len(trace) - 1  # Subtract 1 to exclude the goal state
        successful_runs += 1

if successful_runs > 0:
    average_steps = total_steps / successful_runs
    print(f"\nAverage steps to reach goal: {average_steps:.2f}")
    print(f"Success rate: {successful_runs/n_simulations:.2%}")
else:
    print("\nNo successful runs to the goal state.")


bottlenecks = mdp.find_bottleneck_states()
print("Bottleneck states:", [s + 1 for s in bottlenecks])

mdp.visualize_grid(V, policy)

