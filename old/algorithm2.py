import numpy as np
from queue import Queue
from scipy.stats import entropy
from MDPgrid import GridWorldMDP
from old.determinizeMDP import DeterminizedMDP, find_bottleneck_states
from algorithm1 import find_maximal_achievable_subsets

def compute_state_visitation_frequency(mdp, policy):
    freq = np.zeros(len(mdp.S))
    visited = set()
    q = Queue()
    q.put(mdp.s0)
    visited.add(mdp.s0)

    while not q.empty():
        state = q.get()
        freq[state] += 1
        if state in policy:
            next_state = mdp.P[state][policy[state]]
            if next_state not in visited:
                q.put(next_state)
                visited.add(next_state)

    return freq / np.sum(freq)

def compute_information_gain(mdp, policy, bottlenecks, current_belief):
    state_freq = compute_state_visitation_frequency(mdp, policy)
    information_gain = {}

    for b in bottlenecks:
        posterior_included = current_belief.copy()
        posterior_included[b] = 1.0
        posterior_included /= np.sum(posterior_included)

        posterior_excluded = current_belief.copy()
        posterior_excluded[b] = 0.0
        posterior_excluded /= np.sum(posterior_excluded)

        ig_included = entropy(current_belief) - entropy(posterior_included)
        ig_excluded = entropy(current_belief) - entropy(posterior_excluded)
        information_gain[b] = state_freq[b] * max(ig_included, ig_excluded)

    return information_gain

def can_determine_policy(mdp, belief):
    # we'll assume it can be determined if the belief is certain for all states.
    return np.all((belief == 0) | (belief == 1))

def optimal_query(robot_mdp, human_mdp, traversed_bottlenecks, human_bottlenecks, robot_bottlenecks, current_belief):
    # Find bottlenecks that are in the human MDP but not in the robot MDP
    different_bottlenecks = set(human_bottlenecks) - set(robot_bottlenecks)
    non_traversed_bottlenecks = different_bottlenecks - set(traversed_bottlenecks)
    
    if not non_traversed_bottlenecks:
        print("No additional bottlenecks to query about.")
        return robot_mdp.get_optimal_policy(robot_mdp.value_iteration()), [], []

    # Find maximal achievable subsets
    maximal_subsets = find_maximal_achievable_subsets(robot_mdp, list(non_traversed_bottlenecks))
    
    if not maximal_subsets:
        print("No achievable bottlenecks found.")
        return robot_mdp.get_optimal_policy(robot_mdp.value_iteration()), [], list(non_traversed_bottlenecks)

    # Choose the largest maximal subset
    achievable_bottlenecks = max(maximal_subsets, key=len)
    unachievable_bottlenecks = non_traversed_bottlenecks - set(achievable_bottlenecks)

    print(f"Achievable bottlenecks: {[s + 1 for s in achievable_bottlenecks]}")
    print(f"Unachievable bottlenecks: {[s + 1 for s in unachievable_bottlenecks]}")

    V = robot_mdp.value_iteration()  # Run value iteration
    policy = robot_mdp.get_optimal_policy(V)  # Get optimal policy using V
    information_gain = compute_information_gain(robot_mdp, policy, achievable_bottlenecks, current_belief)

    def recursive_query(bottlenecks, belief):
        if not bottlenecks:
            return [], 0

        for b in bottlenecks:
            # check for single-step-optimal-solution
            new_belief_included = belief.copy()
            new_belief_included[b] = 1.0
            new_belief_included /= np.sum(new_belief_included)

            new_belief_excluded = belief.copy()
            new_belief_excluded[b] = 0.0
            new_belief_excluded /= np.sum(new_belief_excluded)

            if can_determine_policy(robot_mdp, new_belief_included) and can_determine_policy(robot_mdp, new_belief_excluded):
                return [b], 1

        # if no single-step solution, find recursive optimal solution
        min_cost = float('inf')
        best_query = None
        for b in bottlenecks:
            remaining_bottlenecks = bottlenecks - {b}
            _, cost_included = recursive_query(remaining_bottlenecks, new_belief_included)
            _, cost_excluded = recursive_query(remaining_bottlenecks, new_belief_excluded)
            avg_cost = 1 + 0.5 * (cost_included + cost_excluded)
            if avg_cost < min_cost:
                min_cost = avg_cost
                best_query = b

        return [best_query] + recursive_query(bottlenecks - {best_query}, belief)[0], min_cost

    optimal_queries, _ = recursive_query(set(achievable_bottlenecks), current_belief)

    user_selected_bottlenecks = []
    for state in optimal_queries:
        response = input(f"Is state {state + 1} important? (y/n): ").lower()
        if response == 'y':
            user_selected_bottlenecks.append(state)
        
        if input("Continue querying? (y/n): ").lower() != 'y':
            break

    updated_policy = update_robot_policy(robot_mdp, user_selected_bottlenecks)
    return updated_policy, user_selected_bottlenecks, list(unachievable_bottlenecks)

def update_robot_policy(mdp, additional_bottlenecks):
    for state in additional_bottlenecks:
        mdp.add_constraint(state)
    V = mdp.value_iteration()  # Run value iteration again after adding constraints
    return mdp.get_optimal_policy(V)

def test_optimal_query():
    # Create grid worlds
    human_grid = [
        [1,  2,  3,  4,  5],
        [6,  0,  7,  0,  8],
        [9,  0,  0,  0,  0],
        [11, 12, 13, 14, 15]
    ]

    robot_grid = [
        [1,  2,  3,  4,  5],
        [6,  0,  0,  0,  7],
        [8,  0,  9,  0,  0],
        [11, 12, 13, 14, 15]
    ]

    initial_state = 1
    goal_state = 15

    print("Human Grid:")
    for row in human_grid:
        print(row)
    print("\nRobot Grid:")
    for row in robot_grid:
        print(row)

    # Create MDPs
    human_mdp = GridWorldMDP(human_grid, initial_state, goal_state)
    robot_mdp = GridWorldMDP(robot_grid, initial_state, goal_state)

    # Create determinized MDPs
    human_det_mdp = DeterminizedMDP(human_mdp)
    robot_det_mdp = DeterminizedMDP(robot_mdp)

    print("\nHuman MDP States:", human_det_mdp.S)
    print("Robot MDP States:", robot_det_mdp.S)

    # Find bottlenecks
    print("\nFinding Human Bottlenecks:")
    human_bottlenecks = find_bottleneck_states(human_det_mdp)
    print("\nFinding Robot Bottlenecks:")
    robot_bottlenecks = find_bottleneck_states(robot_det_mdp)

    print("\nHuman bottlenecks:", [s + 1 for s in human_bottlenecks])
    print("Robot bottlenecks:", [s + 1 for s in robot_bottlenecks])

    # Set up test scenario
    traversed_bottlenecks = []  # Assume no bottlenecks have been traversed yet
    current_belief = np.ones(robot_det_mdp.n_states) / robot_det_mdp.n_states  # Uniform initial belief

    print("\nStarting Optimal Query Process:")
    # Run optimal query
    updated_policy, selected_bottlenecks, unachievable_bottlenecks = optimal_query(
        robot_det_mdp, human_det_mdp, traversed_bottlenecks, human_bottlenecks, robot_bottlenecks, current_belief
    )

    print("\nSelected Bottlenecks:", [s + 1 for s in selected_bottlenecks])
    print("\nUpdated Policy:")
    for state, action in updated_policy.items():
        print(f"State {state + 1}: Action {action}")

if __name__ == "__main__":
    test_optimal_query()