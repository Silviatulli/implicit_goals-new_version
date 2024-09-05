import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from matplotlib.colors import LinearSegmentedColormap
from functools import lru_cache
from queue import Queue
from collections import defaultdict


class GridWorldMDP:
    def __init__(self, grid, initial_state=0, goal_state=18, discount_factor=0.9, goal_reward=10, 
                 move_prob=0.9, stay_prob=0.1):
        self.grid = grid
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
        self.n_states = max(max(row) for row in grid if row) + 1
        self.n_actions = 4  # Up, Down, Left, Right
        self.initial_state = initial_state - 1
        self.goal_state = goal_state - 1
        self.discount_factor = discount_factor
        self.move_prob = move_prob
        self.stay_prob = stay_prob
        
        # Initialize rewards: 0 for all states except goal
        self.R = np.zeros(self.n_states)
        self.R[self.goal_state] = goal_reward

        # Initialize transition probabilities
        self.P = self._initialize_transitions()

    def _initialize_transitions(self):
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.grid[i][j] == 0:  # Wall
                    continue
                s = self.grid[i][j] - 1  # Convert to 0-indexed
                
                # Define possible moves and their corresponding states
                moves = {
                    0: (i-1, j) if i > 0 else (i, j),           # Up
                    1: (i+1, j) if i < self.n_rows-1 else (i, j),  # Down
                    2: (i, j-1) if j > 0 else (i, j),           # Left
                    3: (i, j+1) if j < self.n_cols-1 else (i, j)   # Right
                }
                
                for a, (next_i, next_j) in moves.items():
                    if self.grid[next_i][next_j] != 0:  # If not a wall
                        next_s = self.grid[next_i][next_j] - 1
                        P[s, a, next_s] = self.move_prob
                        P[s, a, s] += self.stay_prob
                    else:  # If wall, stay in the same state
                        P[s, a, s] += self.move_prob + self.stay_prob
                
                # Distribute remaining probability to other valid moves
                for a in range(self.n_actions):
                    remaining_prob = 1 - P[s, a].sum()
                    if remaining_prob > 0:
                        valid_moves = [m for m in range(self.n_actions) if m != a and P[s, m, s] < 1]
                        if valid_moves:
                            for m in valid_moves:
                                P[s, a] += remaining_prob * P[s, m] / len(valid_moves)
        
        return P

    def value_iteration(self, theta=0.0001, max_iterations=1000):
        V = np.zeros(self.n_states)
        for _ in range(max_iterations):
            delta = 0
            for s in range(self.n_states):
                if s == self.goal_state:
                    continue
                v = V[s]
                V[s] = max(np.sum(self.P[s, a] * (self.R + self.discount_factor * V)) for a in range(self.n_actions))
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        return V

    # def extract_policy(self, V):
    #     policy = np.zeros(self.n_states, dtype=int)
    #     for s in range(self.n_states):
    #         if s == self.goal_state:
    #             continue
    #         Q = np.array([np.sum(self.P[s, a] * (self.R + self.discount_factor * V)) for a in range(self.n_actions)])
    #         policy[s] = np.argmax(Q)
    #     return policy

    def extract_policy(self, V):
        policy = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            if s == self.goal_state:
                continue
            action_values = [np.sum(self.P[s, a] * (self.R + self.discount_factor * V)) for a in range(self.n_actions)]
            best_action = np.argmax(action_values)
            print(action_values, best_action)
            policy[s, best_action] = 1.0
        
        return policy

    def epsilon_greedy_policy(self, policy, epsilon=0.2):
        def choose_action(state):
            if np.random.random() < epsilon:
                return np.random.choice(self.n_actions)
            else:
                return np.random.choice(self.n_actions, p=policy[state])
        return choose_action

    def generate_trace(self, policy, start_state, max_depth=100):
        trace = []
        current_state = start_state
        choose_action = self.epsilon_greedy_policy(policy)
        
        for _ in range(max_depth):
            if current_state == self.goal_state:
                break
            
            action = choose_action(current_state)
            trace.append((current_state, action))
            
            next_state = np.random.choice(self.n_states, p=self.P[current_state, action])
            
            current_state = next_state
        
        trace.append((current_state, None))
        return trace

    def compute_P_G(self, policy, state, max_depth=100, epsilon=1e-6):
        @lru_cache(maxsize=None)
        def P_G_helper(s, depth):
            if s == self.goal_state:
                return 1.0
            if depth >= max_depth:
                return 0.0
            
            a = policy[s]
            prob = 0.0
            for s_next in range(self.n_states):
                if self.P[s, a, s_next] > epsilon:
                    prob += self.P[s, a, s_next] * P_G_helper(s_next, depth + 1)
            return prob

        return P_G_helper(state, 0)

    def compute_all_P_G(self, policy, max_depth=100):
        return np.array([self.compute_P_G(policy, s, max_depth) for s in range(self.n_states)])

    # def generate_trace(self, policy, start_state, max_depth=100):
    #     trace = []
    #     current_state = start_state
        
    #     for _ in range(max_depth):
    #         if current_state == self.goal_state:
    #             break
            
    #         action = policy[current_state]
    #         trace.append((current_state, action))
            
    #         # Determine next state based on transition probabilities
    #         next_state = np.random.choice(self.n_states, p=self.P[current_state, action])
            
    #         current_state = next_state
        
    #     # Add the final state to the trace
    #     trace.append((current_state, None))
        
    #     return trace

    def generate_all_traces(self, policy, start_state, n_traces=100, max_depth=100):
        return [self.generate_trace(policy, start_state, max_depth) for _ in range(n_traces)]

    def print_trace(self, trace):
        action_names = ['Up', 'Down', 'Left', 'Right']
        for i, (state, action) in enumerate(trace):
            if i == len(trace) - 1:
                print(f"State {state+1} (Goal)" if state == self.goal_state else f"State {state+1} (Max Depth Reached)")
            else:
                print(f"State {state+1} -> Action: {action_names[action]}")

 
    

    def find_bottleneck_states(self):
        paths = self.find_all_paths(self.initial_state, self.goal_state)
        # print(f"All paths: {paths}")
        
        if not paths:
            # print("No valid paths found")
            return []
        
        potential_bottlenecks = set(range(self.n_states))
        potential_bottlenecks.remove(self.initial_state)
        potential_bottlenecks.remove(self.goal_state)
        # print(f"Initial potential bottlenecks: {potential_bottlenecks}")
        
        for path in paths:
            potential_bottlenecks &= set(path)
            # print(f"After considering path {path}: {potential_bottlenecks}")
        
        return list(potential_bottlenecks)
    

    def find_all_paths(self, start, end):
        def dfs(current, path):
            if current == end:
                paths.append(path)
                # print(f"Found path: {path}")
                return
            
            for next_state in range(self.n_states):
                if any(self.P[current, :, next_state] > 0) and next_state not in path:
                    dfs(next_state, path + [next_state])
        
        paths = []
        dfs(start, [start])
        return paths
    


    def print_transitions(self):
        for s in range(self.n_states):
            print(f"State {s+1}:")
            for a in range(self.n_actions):
                for s_next in range(self.n_states):
                    if self.P[s, a, s_next] > 0:
                        print(f"  -> {s_next+1} (action {a})")

    def print_grid(self):
        for row in self.grid:
            print(' '.join(f'{cell:2d}' if cell != 0 else ' #' for cell in row))

    def visualize_grid(self, V, policy):
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, self.n_cols)
        ax.set_ylim(0, self.n_rows)
        ax.set_aspect('equal')

        cmap = LinearSegmentedColormap.from_list("RedGreen", ["red", "white", "green"])
        norm = plt.Normalize(vmin=min(V), vmax=max(V))

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                state = self.grid[i][j]
                if state == 0:
                    rect = Rectangle((j, self.n_rows-1-i), 1, 1, facecolor='gray', edgecolor='black')
                    ax.add_patch(rect)
                else:
                    state_idx = state - 1
                    color = 'gold' if state_idx == self.goal_state else cmap(norm(V[state_idx]))
                    rect = Rectangle((j, self.n_rows-1-i), 1, 1, facecolor=color, edgecolor='black')
                    ax.add_patch(rect)
                    
                    ax.text(j+0.5, self.n_rows-1-i+0.5, f"S{state}\n{V[state_idx]:.2f}", ha='center', va='center', fontsize=10, color='black')

                    if state_idx == self.initial_state:
                        ax.text(j+0.5, self.n_rows-1-i+0.8, "Initial", ha='center', va='center', fontsize=12, color='blue', fontweight='bold')
                    
                    if state_idx == self.goal_state:
                        ax.text(j+0.5, self.n_rows-1-i+0.8, "Goal", ha='center', va='center', fontsize=12, color='black', fontweight='bold')
                    
                    if state_idx != self.goal_state:
                        # action = policy[state_idx]
                        choose_action = self.epsilon_greedy_policy(policy, epsilon=0)
                        action = choose_action(state_idx)
                        dx, dy = 0, 0
                        if action == 0: dx, dy = 0.0, 0.2  # Up
                        elif action == 1: dx, dy = 0, -0.2  # Down
                        elif action == 2: dx, dy = -0.2, 0  # Left
                        elif action == 3: dx, dy = 0.2, 0  # Right

                        start_x, start_y = j+0.5, self.n_rows-1-i+0.2
                        if action == 0:
                            start_y -= 0.2 
                        
                        arrow = Arrow(start_x, start_y, dx, dy, width=0.3, color='black')
                        ax.add_patch(arrow)

        ax.set_title("Grid World MDP: Value Function and Optimal Policy")
        ax.set_xticks([])
        ax.set_yticks([])
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('State Value')

        plt.tight_layout()
        plt.show()




if __name__ == "__main__":

    grid = [
        [1, 2, 3, 4, 0],
        [5, 0, 6, 0, 0],
        [7, 8, 9, 10, 11],
        [12, 13, 0, 0, 14],
        [15, 16, 17, 18, 19]
    ]
    mdp = GridWorldMDP(grid, initial_state=17, goal_state=4)

    V = mdp.value_iteration()
    # print("Value function:")
    # print(V)

    policy = mdp.extract_policy(V)

    # print("\nOptimal Policy:")
    # print(policy)

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



