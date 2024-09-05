import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from matplotlib.colors import LinearSegmentedColormap
from queue import Queue
from scipy.stats import entropy
from itertools import combinations

class DeterminizedMDP:
    def __init__(self, S, A, s0, G):
        self.S = S  
        self.A = A 
        self.s0 = s0  # initial state
        self.G = G  
        self.P = {}  # transition function
        self.R = np.full(len(S), -1.0)  # negative reward for each state
        self.R[G] = 0.0  # zero reward for the goal state

    def add_transition(self, s, a, s_next):
        if s not in self.P:
            self.P[s] = {}
        self.P[s][a] = s_next

    def set_reward(self, s, r):
        self.R[s] = r

    def value_iteration(self, gamma=0.95, epsilon=1e-6):
        V = np.zeros(len(self.S))
        while True:
            delta = 0
            for s in range(len(self.S)):
                if s == self.G:
                    continue
                v = V[s]
                if s in self.P:
                    V[s] = self.R[s] + gamma * max(V[self.P[s][a]] for a in self.P[s])
                delta = max(delta, abs(v - V[s]))
            if delta < epsilon:
                break
        self.V = V
        return V
    
    def get_value_function(self):
        return self.V
    
    def get_optimal_policy(self):
        V = self.value_iteration()
        policy = {}
        for s in range(len(self.S)):
            if s in self.P:
                policy[s] = max(self.P[s], key=lambda a: V[self.P[s][a]])
        return policy



def find_bottleneck_states(mdp):
    bottlenecks = []
    original_rewards = mdp.R.copy()
    original_policy = mdp.get_optimal_policy()

    for s in range(len(mdp.S)):
        if s != mdp.s0 and s != mdp.G:
            # reset rewards
            mdp.R = np.zeros(len(mdp.S))
            mdp.R[mdp.G] = 0  # no reward for goal state initially

            # define a new reward function
            def new_reward_function(state, next_state):
                if state == s and next_state == mdp.G:
                    return 1  # positive reward only when transitioning from s to goal
                return 0

            # perform value iteration with the new reward function
            V = np.zeros(len(mdp.S))
            gamma = 0.95
            for _ in range(100): 
                for state in range(len(mdp.S)):
                    if state == mdp.G:
                        continue
                    if state in mdp.P:
                        V[state] = max(
                            new_reward_function(state, mdp.P[state][a]) + 
                            gamma * V[mdp.P[state][a]] 
                            for a in mdp.P[state]
                        )

            # check if the value of the initial state is positive
            if V[mdp.s0] > 0:
                bottlenecks.append(s)

    # restore original rewards and recompute optimal policy
    mdp.R = original_rewards
    mdp.value_iteration()
    mdp.get_optimal_policy()

    return bottlenecks


def find_max_bottleneck_policy(mdp, bottlenecks):
    def find_policy_for_bottlenecks(target_bottlenecks):
        # 1 find optimal policy for modified mdp
        original_rewards = mdp.R.copy()
        mdp.R[:] = 0 
        mdp.R[mdp.G] = 10

        for b in target_bottlenecks:
            mdp.R[b] = 1  # small positive reward for bottleneck states

        optimal_policy = mdp.get_optimal_policy()

        # 2 check if the traces from optimal policy cover all target bottlenecks
        def check_policy_coverage(policy, start, targets):
            visited = set()
            covered_targets = set()
            current = start

            while current not in visited and current != mdp.G:
                visited.add(current)
                if current in targets:
                    covered_targets.add(current)
                if current in policy:
                    current = mdp.P[current][policy[current]]
                else:
                    break

            return covered_targets == set(targets)

        is_valid = check_policy_coverage(optimal_policy, mdp.s0, target_bottlenecks)

        mdp.R = original_rewards
        return optimal_policy if is_valid else None

    visited_combinations = set()
    best_policies = {}

    def explore_combinations(current_set):
        if tuple(current_set) in visited_combinations:
            return

        visited_combinations.add(tuple(current_set))
        policy = find_policy_for_bottlenecks(current_set)

        if policy is not None:
            best_policies[tuple(sorted(current_set))] = policy
            return

        for item in current_set:
            new_set = current_set - {item}
            if new_set:
                explore_combinations(new_set)

    # start with all bottlenecks
    all_bottlenecks = set(bottlenecks)
    explore_combinations(all_bottlenecks)

    # find the policy that traverses the maximum number of bottlenecks
    if best_policies:
        max_traversed = max(best_policies.keys(), key=len)
        best_policy = best_policies[max_traversed]
        traversed_bottlenecks = list(max_traversed)
        non_traversed_bottlenecks = list(all_bottlenecks - set(max_traversed))
    else:
        # if no policy found for any combination, return the policy for reaching the goal
        best_policy = mdp.get_optimal_policy()
        _, traversed = check_policy_coverage(best_policy, mdp.s0, all_bottlenecks)
        traversed_bottlenecks = list(traversed)
        non_traversed_bottlenecks = list(all_bottlenecks - traversed)

    return best_policy, traversed_bottlenecks, non_traversed_bottlenecks

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
        # compute posterior if this bottleneck is included
        posterior_included = current_belief.copy()
        posterior_included[b] = 1.0
        posterior_included /= np.sum(posterior_included)

        # compute posterior if this bottleneck is not included
        posterior_excluded = current_belief.copy()
        posterior_excluded[b] = 0.0
        posterior_excluded /= np.sum(posterior_excluded)

        # compute expected information gain
        ig_included = entropy(current_belief) - entropy(posterior_included)
        ig_excluded = entropy(current_belief) - entropy(posterior_excluded)
        information_gain[b] = state_freq[b] * max(ig_included, ig_excluded)

    return information_gain

def information_theoretic_query(mdp, non_traversed_bottlenecks, current_policy, traversed_bottlenecks):
    all_bottlenecks = traversed_bottlenecks + non_traversed_bottlenecks
    current_belief = np.ones(len(mdp.S)) / len(mdp.S)  # initialize with uniform belief
    
    for b in traversed_bottlenecks:
        current_belief[b] = 1.0
    current_belief /= np.sum(current_belief)

    user_selected_bottlenecks = []
    cumulative_ig = 0
    
    while non_traversed_bottlenecks:
        information_gain = compute_information_gain(mdp, current_policy, non_traversed_bottlenecks, current_belief)
        state_to_query = max(information_gain, key=information_gain.get)
        max_ig = information_gain[state_to_query]
        
        print(f"Information gain for state {state_to_query + 1}: {max_ig:.4f}")
        
        response = input(f"Do you want to include state {state_to_query + 1} in the path? (y/n): ").lower()
        if response == 'y':
            user_selected_bottlenecks.append(state_to_query)
            current_belief[state_to_query] = 1.0
        else:
            current_belief[state_to_query] = 0.0
        
        current_belief /= np.sum(current_belief)
        non_traversed_bottlenecks.remove(state_to_query)

        cumulative_ig += max_ig
        print(f"cumulative information gain: {cumulative_ig:.4f}")
        
        if non_traversed_bottlenecks:
            continue_querying = input("do you want to continue querying? (y/n): ").lower()
            if continue_querying != 'y':
                break

    if not user_selected_bottlenecks:
        return current_policy, []

    # update rewards to encourage passing through user-selected bottlenecks
    original_rewards = mdp.R.copy()
    for state in user_selected_bottlenecks:
        mdp.R[state] = 10  # high reward for user-selected bottlenecks

    # compute new policy
    new_policy = mdp.get_optimal_policy()

    # restore original rewards
    mdp.R = original_rewards

    # check which user-selected bottlenecks are actually traversed
    actually_traversed = set(traversed_bottlenecks)
    visited = set()
    q = Queue()
    q.put(mdp.s0)
    visited.add(mdp.s0)

    while not q.empty():
        current = q.get()
        if current in user_selected_bottlenecks:
            actually_traversed.add(current)
        if current == mdp.G:
            break
        if current in new_policy:
            next_state = mdp.P[current][new_policy[current]]
            if next_state not in visited:
                q.put(next_state)
                visited.add(next_state)

    return new_policy, list(actually_traversed)


def visualize_grid(grid, bottlenecks, initial_state, goal_state, policy=None, traversed_bottlenecks=None):
    n_rows = len(grid)
    n_cols = len(grid[0])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect('equal')

    cmap = LinearSegmentedColormap.from_list("CustomMap", ["white", "lightblue"])
    
    for i in range(n_rows):
        for j in range(n_cols):
            state = grid[i][j]
            if state == 0:
                rect = Rectangle((j, n_rows-1-i), 1, 1, facecolor='gray', edgecolor='black')
                ax.add_patch(rect)
            else:
                state_idx = state - 1
                if state_idx == goal_state:
                    color = 'gold'
                elif state_idx in traversed_bottlenecks:
                    color = 'lightgreen'
                elif state_idx in bottlenecks:
                    color = 'orange'
                else:
                    color = cmap(0.5)
                rect = Rectangle((j, n_rows-1-i), 1, 1, facecolor=color, edgecolor='black')
                ax.add_patch(rect)
                
                ax.text(j+0.5, n_rows-1-i+0.5, f"S{state}", ha='center', va='center', fontsize=10, color='black')

                if state_idx == initial_state:
                    ax.text(j+0.5, n_rows-1-i+0.8, "Initial", ha='center', va='center', fontsize=12, color='blue', fontweight='bold')
                
                if state_idx == goal_state:
                    ax.text(j+0.5, n_rows-1-i+0.8, "Goal", ha='center', va='center', fontsize=12, color='black', fontweight='bold')

                if policy and state_idx in policy:
                    action = policy[state_idx]
                    dx, dy = 0, 0
                    if action == 'up': dx, dy = 0.0, 0.2
                    elif action == 'down': dx, dy = 0, -0.2
                    elif action == 'left': dx, dy = -0.2, 0
                    elif action == 'right': dx, dy = 0.2, 0
                    arrow = Arrow(j+0.5, n_rows-1-i+0.5, dx, dy, width=0.3, color='red')
                    ax.add_patch(arrow)

    ax.set_title("Grid World MDP: Bottleneck States and Policy")
    ax.set_xticks([])
    ax.set_yticks([])
    
    legend_elements = [
        plt.Rectangle((0,0),1,1,facecolor='lightgreen',edgecolor='black',label='Traversed Bottleneck'),
        plt.Rectangle((0,0),1,1,facecolor='orange',edgecolor='black',label='Non-traversed Bottleneck'),
        plt.Rectangle((0,0),1,1,facecolor='gold',edgecolor='black',label='Goal'),
        plt.Rectangle((0,0),1,1,facecolor=cmap(0.5),edgecolor='black',label='Normal'),
        plt.Rectangle((0,0),1,1,facecolor='gray',edgecolor='black',label='Wall'),
        Arrow(0,0,1,0,width=0.3,color='red',label='Policy Action')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    plt.tight_layout()
    plt.show()

def visualize_grid_with_values(grid, V, bottlenecks, initial_state, goal_state, policy=None):
    n_rows = len(grid)
    n_cols = len(grid[0])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect('equal')

    cmap = LinearSegmentedColormap.from_list("RedGreen", ["red", "white", "green"])
    norm = plt.Normalize(vmin=min(V), vmax=max(V))

    for i in range(n_rows):
        for j in range(n_cols):
            state = grid[i][j]
            if state == 0:
                rect = Rectangle((j, n_rows-1-i), 1, 1, facecolor='gray', edgecolor='black')
                ax.add_patch(rect)
            else:
                state_idx = state - 1
                color = 'gold' if state_idx == goal_state else cmap(norm(V[state_idx]))
                rect = Rectangle((j, n_rows-1-i), 1, 1, facecolor=color, edgecolor='black')
                ax.add_patch(rect)
                
                ax.text(j+0.5, n_rows-1-i+0.5, f"S{state}\n{V[state_idx]:.2f}", ha='center', va='center', fontsize=8, color='black')

                if state_idx == initial_state:
                    ax.text(j+0.5, n_rows-1-i+0.8, "Initial", ha='center', va='center', fontsize=10, color='blue', fontweight='bold')
                
                if state_idx == goal_state:
                    ax.text(j+0.5, n_rows-1-i+0.8, "Goal", ha='center', va='center', fontsize=10, color='black', fontweight='bold')
                
                if state_idx in bottlenecks:
                    ax.text(j+0.5, n_rows-1-i+0.2, "B", ha='center', va='center', fontsize=10, color='red', fontweight='bold')

                if policy and state_idx in policy:
                    action = policy[state_idx]
                    dx, dy = 0, 0
                    if action == 'up': dx, dy = 0.0, 0.2
                    elif action == 'down': dx, dy = 0, -0.2
                    elif action == 'left': dx, dy = -0.2, 0
                    elif action == 'right': dx, dy = 0.2, 0
                    arrow = Arrow(j+0.5, n_rows-1-i+0.5, dx, dy, width=0.3, color='black')
                    ax.add_patch(arrow)

    ax.set_title("Grid World MDP: State Values and Policy")
    ax.set_xticks([])
    ax.set_yticks([])
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('State Value')

    legend_elements = [
        plt.Rectangle((0,0),1,1,facecolor='gray',edgecolor='black',label='Wall'),
        plt.Rectangle((0,0),1,1,facecolor='gold',edgecolor='black',label='Goal'),
        plt.Line2D([0], [0], marker='o', color='w', label='Bottleneck', markerfacecolor='r', markersize=10),
        Arrow(0,0,1,0,width=0.3,color='black',label='Policy Action')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # define the grid
    grid = [
        [1,  2,  3,  4,  5,  6,  7,  8],
        [9,  10,  0,  0,  0,  0,  0, 11],
        [12, 13,  0,  0,  0,  0,  0, 14],
        [15, 16, 17, 18, 19, 20, 21, 22],
        [23, 0,  0,  0,  0,  0,  24, 25],
        [26, 0,  0,  0,  0,  0,  27, 28],
        [29, 30, 31, 32, 33, 34, 35, 36]
    ]

    # create the determinized MDP
    S = list(range(1, 37))
    A = ['up', 'down', 'left', 'right']
    s0 = 1 
    G = 36

    mdp = DeterminizedMDP(S, A, s0 - 1, G - 1) 

    # add transitions based on the grid
    for i, row in enumerate(grid):
        for j, state in enumerate(row):
            if state == 0:  # wall
                continue
            s = state - 1  # convert to 0-based indexing
            if i > 0 and grid[i-1][j] != 0:
                mdp.add_transition(s, 'up', grid[i-1][j] - 1)
            if i < len(grid) - 1 and grid[i+1][j] != 0:
                mdp.add_transition(s, 'down', grid[i+1][j] - 1)
            if j > 0 and grid[i][j-1] != 0:
                mdp.add_transition(s, 'left', grid[i][j-1] - 1)
            if j < len(grid[0]) - 1 and grid[i][j+1] != 0:
                mdp.add_transition(s, 'right', grid[i][j+1] - 1)
    
    # find bottleneck states
    bottlenecks = find_bottleneck_states(mdp)
    print("Bottleneck states:", [s + 1 for s in bottlenecks])  # add 1 for 1-based indexing in output

    # find policy with maximum bottleneck states
    best_policy, traversed_bottlenecks, non_traversed_bottlenecks = find_max_bottleneck_policy(mdp, bottlenecks)
    print("Traversed bottleneck states:", [s + 1 for s in traversed_bottlenecks])
    print("Non-traversed bottleneck states:", [s + 1 for s in non_traversed_bottlenecks])

    # use information-theoretic querying for non-traversed bottlenecks
    updated_policy, user_traversed_bottlenecks = information_theoretic_query(mdp, non_traversed_bottlenecks, best_policy, traversed_bottlenecks)
    
    print("User-selected traversed bottleneck states:", [s + 1 for s in user_traversed_bottlenecks])
    
    # update traversed and non-traversed bottlenecks
    all_traversed_bottlenecks = list(set(traversed_bottlenecks + user_traversed_bottlenecks))
    final_non_traversed_bottlenecks = list(set(bottlenecks) - set(all_traversed_bottlenecks))
    
    print("Final traversed bottleneck states:", [s + 1 for s in all_traversed_bottlenecks])
    print("Final non-traversed bottleneck states:", [s + 1 for s in final_non_traversed_bottlenecks])

    # visualize the grid with policy
    visualize_grid(grid, bottlenecks, s0 - 1, G - 1, best_policy, traversed_bottlenecks)

    V = mdp.get_value_function()
    visualize_grid_with_values(grid, V, bottlenecks, s0 - 1, G - 1, updated_policy)