import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow # noqa: F401
from matplotlib.colors import LinearSegmentedColormap # noqa: F401
import matplotlib.patches as patches
from functools import lru_cache
from collections import deque
import copy


class GridWorldMDP:
    def __init__(self, grid, initial_state, goal_state, discount_factor=0.9, goal_reward=10, 
                 move_prob=0.9, stay_prob=0.1):
        """
        Initialize the GridWorldMDP.

        Args:
            grid (List[List[int]]): The grid layout. 0 represents walls, other numbers represent states.
            initial_state (int): The initial state (1-indexed).
            goal_state (int): The goal state (1-indexed).
            discount_factor (float): The discount factor for future rewards.
            goal_reward (float): The reward for reaching the goal state.
            move_prob (float): The probability of moving in the intended direction.
            stay_prob (float): The probability of staying in the same state.
        """
        self.grid = grid
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
        self.n_states = max(max(row) for row in grid if row) 
        self.n_actions = 4  # Up, Down, Left, Right
        self.initial_state = initial_state - 1  # Convert to 0-indexed
        self.goal_state = goal_state - 1  # Convert to 0-indexed
        self.discount_factor = discount_factor
        self.move_prob = move_prob
        self.stay_prob = stay_prob
        
        # Initialize transitions and rewards
        self.P, self.R = self._initialize_transitions_and_rewards(goal_reward)

    def _initialize_transitions_and_rewards(self, goal_reward):
        R = np.zeros((self.n_states))
        R[self.goal_state] = goal_reward  # reward for reaching the goal state
        
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.grid[i][j] == 0:  # wall
                    continue
                s = self.grid[i][j] - 1  # convert to 0-indexed
                
                if s == self.goal_state:
                    # make goal state absorbing
                    for a in range(self.n_actions):
                        P[s, a, s] = 1.0
                    continue
                
                moves = {
                    0: (i-1, j) if i > 0 else (i, j),           # up
                    1: (i+1, j) if i < self.n_rows-1 else (i, j),  # down
                    2: (i, j-1) if j > 0 else (i, j),           # left
                    3: (i, j+1) if j < self.n_cols-1 else (i, j)   # right
                }
                
                for a, (next_i, next_j) in moves.items():
                    if self.grid[next_i][next_j] != 0:  # if not a wall
                        next_s = self.grid[next_i][next_j] - 1
                        P[s, a, next_s] = self.move_prob
                        P[s, a, s] += self.stay_prob
                    else:  # if wall, stay in the same state
                        P[s, a, s] = 1
                        # R[s, a, s] = -1  # small negative reward for hitting a wall
               
                # distribute remaining probability to other valid moves
                for a in range(self.n_actions):
                    remaining_prob = 1 - P[s, a].sum()
                    if remaining_prob > 0:
                        valid_moves = [m for m in range(self.n_actions) if m != a and P[s, m, s] < 1]
                        if valid_moves:
                            for m in valid_moves:
                                P[s, a] += remaining_prob * P[s, m] / len(valid_moves)      
        return P, R

    def value_iteration(self, theta=0.0001, max_iterations=1000):
        """
        Args:
            theta (float): Convergence threshold.
            max_iterations (int): Maximum number of iterations.
        Returns:
            tuple: (Q-function, V-function)
        """
        Q = np.zeros((self.n_states, self.n_actions))
        for _ in range(max_iterations):
            Q_new = np.zeros((self.n_states, self.n_actions))
            V = np.max(Q, axis=1)
            for s in range(self.n_states):
                if s == self.goal_state:
                    Q_new[s, :] = self.R[self.goal_state]
                else:
                    for a in range(self.n_actions):
                        Q_new[s, a] = np.sum(self.P[s, a] * (self.R[s] + self.discount_factor * V))
            if np.linalg.norm(Q - Q_new) < theta:
                break
            Q = Q_new
        V = np.max(Q, axis=1)
        return Q, V

    def compute_deterministic_policy(self, Q):
        """
        Compute a deterministic policy from the Q-function.
        Args:
            Q (np.array): The Q-function.
        Returns:
            np.array: A deterministic policy. policy[s] is the action to take in state s.
        """
        policy = np.argmax(Q, axis=1)
        policy[self.goal_state] = np.random.randint(self.n_actions)  # random action for goal state
        return policy

    def compute_stochastic_policy(self, Q, temperature=1.0):
        """
        Compute a stochastic policy from the Q-function using softmax.
        Args:
            Q (np.array): The Q-function.
            temperature (float): Temperature parameter for softmax. Higher values make the policy more random.
        Returns:
            np.array: A stochastic policy. policy[s, a] is the probability of taking action a in state s.
        """
        policy = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            if s == self.goal_state:
                policy[s] = np.ones(self.n_actions) / self.n_actions  # uniform distribution for goal state
            else:
                exp_Q = np.exp((Q[s] - np.max(Q[s])) / temperature)  # subtract max(Q) for numerical stability
                policy[s] = exp_Q / np.sum(exp_Q)
        return policy

    def get_optimal_policies(self, temperature=1.0):
        """
        Compute both deterministic and stochastic optimal policies.
        Args:
            temperature (float): Temperature parameter for the stochastic policy.
        Returns:
            tuple: (deterministic policy, stochastic policy, V-function)
        """
        Q, V = self.value_iteration()
        deterministic_policy = self.compute_deterministic_policy(Q)
        stochastic_policy = self.compute_stochastic_policy(Q, temperature)
        return deterministic_policy, stochastic_policy, V


    def policy_iteration(self, max_iterations=100):
        """
        Perform policy iteration to compute the optimal policy.
        Args:
            max_iterations (int): Maximum number of iterations.
        Returns:
            tuple: (optimal value function, optimal policy)
        """
        def policy_evaluation(policy):
            V = np.zeros(self.n_states)
            while True:
                delta = 0
                for s in range(self.n_states):
                    if s == self.goal_state:
                        continue
                    v = V[s]
                    V[s] = np.sum(policy[s] * np.sum(self.P[s, a] * (self.R[s, a] + self.discount_factor * V)) for a in range(self.n_actions))
                    delta = max(delta, abs(v - V[s]))
                if delta < 1e-4:
                    break
            return V

        policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        for _ in range(max_iterations):
            V = policy_evaluation(policy)
            policy_stable = True
            for s in range(self.n_states):
                if s == self.goal_state:
                    continue
                old_action = np.argmax(policy[s])
                Q = [np.sum(self.P[s, a] * (self.R[s, a] + self.discount_factor * V)) for a in range(self.n_actions)]
                best_action = np.argmax(Q)
                if old_action != best_action:
                    policy_stable = False
                policy[s] = np.eye(self.n_actions)[best_action]
            if policy_stable:
                break
        return V, policy

    def generate_trace(self, policy, start_state, max_depth=500):
        """
        Generate a trace of states and actions following the given policy.
        Args:
            policy (np.array): The policy to follow.
            start_state (int): The starting state.
            max_depth (int): Maximum number of steps.

        Returns:
            list: A list of (state, action) tuples representing the trace.
        """
        trace = []
        current_state = start_state
        
        for _ in range(max_depth):
            if current_state == self.goal_state:
                break
            
            action = np.random.choice(self.n_actions, p=policy[current_state])
            trace.append((current_state, action))
            
            next_state = np.random.choice(self.n_states, p=self.P[current_state, action])
            
            current_state = next_state
        
        trace.append((current_state, None))
        return trace

   
    def find_all_traces(self, policy, max_depth=1000):
        """
        Find all possible traces from the initial state to the goal state following the given policy.
        Args:
            policy (np.array): The policy to follow.
            max_depth (int): Maximum depth of the trace to prevent infinite loops.

        Returns:
            list: A list of all possible traces, where each trace is a list of (state, action) tuples.
        """
        all_traces = []
        queue = deque([(self.initial_state, [], set())])
        
        while queue:
            current_state, current_trace, visited = queue.popleft()
            
            if current_state == self.goal_state:
                all_traces.append(current_trace + [(current_state + 1, None)])
                continue
            
            if len(current_trace) >= max_depth:
                continue
            
            if isinstance(policy[current_state], np.ndarray):
                # Stochastic policy
                possible_actions = np.where(policy[current_state] > 0)[0]
            else:
                # Deterministic policy
                possible_actions = [policy[current_state]]
            
            for action in possible_actions:
                for next_state in range(self.n_states):
                    if self.P[current_state, action, next_state] > 0 and next_state != current_state:
                        if next_state not in visited:
                            new_trace = current_trace + [(current_state + 1, action)]
                            new_visited = visited.copy()
                            new_visited.add(next_state)
                            queue.append((next_state, new_trace, new_visited))
        
        return all_traces

    def find_bottleneck_states(self, policy, n=100, p=0.01, epsilon=1e-6, gamma=0.99):
        """
        Find bottleneck states using value iteration and reward modification.
        
        Args:
            policy (np.array): The policy to analyze (not used in this implementation but kept for consistency).
            n (int): Negative reward for potential bottleneck states.
            p (float): Positive reward for the goal state.
            epsilon (float): Convergence threshold for value iteration.
            gamma (float): Discount factor for value iteration.
        
        Returns:
            list: A list of bottleneck states.
        """
        @lru_cache(maxsize=None)
        def value_iteration(R_tuple):
            R = np.array(R_tuple)
            V = np.zeros(self.n_states)
            while True:
                delta = 0
                for s in range(self.n_states):
                    if s == self.goal_state:
                        V[s] = R[s]
                        continue
                    v = V[s]
                    V[s] = R[s] + gamma * max(
                        np.sum(self.P[s, a] * V) for a in range(self.n_actions)
                    )
                    delta = max(delta, abs(v - V[s]))
                if delta < epsilon:
                    break
            return V

        bottlenecks = []
        target_states = [s for s in range(self.n_states) if s not in {self.initial_state, self.goal_state}]

        for s in target_states:
            R_modified = np.zeros(self.n_states)
            R_modified[s] = -n
            R_modified[self.goal_state] = p
            V = value_iteration(tuple(R_modified))
            if V[self.initial_state] <= 0:
                bottlenecks.append(s + 1)  # Convert back to 1-indexed state

        return bottlenecks
    
 
 
    def visualize_policy(self, policy, V, is_deterministic=True, fig_size=(10, 8)):
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.axis('off')
        
        cell_size = 1
        margin = 0.1
        
        # draw grid
        for i in range(self.n_rows + 1):
            ax.plot([0, self.n_cols], [i, i], color='black', linewidth=2)
        for j in range(self.n_cols + 1):
            ax.plot([j, j], [0, self.n_rows], color='black', linewidth=2)

        # fill cells and add text
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                y = self.n_rows - 1 - i
                x = j
                state = self.grid[i][j]
                if state == 0:  # Wall
                    rect = patches.Rectangle((x, y), cell_size, cell_size, edgecolor='none', facecolor='gray')
                    ax.add_patch(rect)
                else:
                    state_idx = state - 1
                    # extended_state = self.extended_states.index((state_idx, tuple([0] * len(self.subgoals))))
                    
                    if state_idx == self.goal_state:
                        rect = patches.Rectangle((x, y), cell_size, cell_size, edgecolor='none', facecolor='lightgreen')
                        ax.add_patch(rect)

                    # Add state number
                    ax.text(x + 0.5, y + 0.5, f'S{state}',
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=10)

                    # Add policy visualization
                    if state_idx != self.goal_state:
                        if is_deterministic:
                            a = policy[state_idx]
                            self._draw_action_marker(ax, a, x, y, 1.0)
                        else:
                            for a, prob in enumerate(policy[state_idx]):
                                if prob > 0.01:  # Draw for all non-negligible probabilities
                                    self._draw_action_marker(ax, a, x, y, prob)

        ax.set_xlim(-margin, self.n_cols + margin)
        ax.set_ylim(-margin, self.n_rows + margin)
        plt.title("Grid World MDP: Value Function and Policy", fontsize=16)
        plt.tight_layout()
        plt.show()

    def _draw_action_marker(self, ax, action, x, y, probability):
        markers = ['▲', '▼', '◀', '▶']  # Up, Down, Left, Right
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # different color for each action
        positions = [(0.5, 0.9), (0.5, 0.1), (0.1, 0.5), (0.9, 0.5)]  # Top, Bottom, Left, Right
        
        dx, dy = positions[action]
        ax.text(x + dx, y + dy, markers[action],
                horizontalalignment='center', verticalalignment='center',
                fontsize=20, color=colors[action], 
                fontweight='bold', alpha=min(1.0, probability + 0.3))

        # add probability text
        prob_positions = [(0.5, 0.75), (0.5, 0.25), (0.25, 0.5), (0.75, 0.5)]  # adjusted for each direction
        prob_dx, prob_dy = prob_positions[action]
        ax.text(x + prob_dx, y + prob_dy, f'{probability:.2f}',
                horizontalalignment='center', verticalalignment='center',
                fontsize=8, color=colors[action],
                alpha=min(1.0, probability + 0.3))

    def print_policy(self, policy, is_deterministic=True):
        """
        Print a text representation of the policy.

        Args:
            policy (np.array): The policy to print.
            is_deterministic (bool): Whether the policy is deterministic.
        """
        action_symbols = ['↑', '↓', '←', '→']
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                state = self.grid[i][j]
                if state == 0:
                    print(' # ', end='')
                elif state - 1 == self.goal_state:
                    print(' G ', end='')
                else:
                    state_idx = state - 1
                    if is_deterministic:
                        action = policy[state_idx]
                        print(f' {action_symbols[action]} ', end='')
                    else:
                        probs = policy[state_idx]
                        action = np.argmax(probs)
                        prob = probs[action]
                        print(f'{action_symbols[action]}{prob:.1f}', end='')
            print()  # new line for new row


if __name__ == "__main__":
    
    grid = [
        [1, 2, 3],
        [4, 5, 0],
        [6, 7, 8]
    ]
    initial_state = 1
    goal_state = 8
    
    mdp = GridWorldMDP(grid, initial_state=initial_state, goal_state=goal_state)
    det_policy, stoch_policy, V = mdp.get_optimal_policies(temperature=0.1)
    det_bottlenecks = mdp.find_bottleneck_states(det_policy)


    # visualize policy
    mdp.visualize_policy(det_policy, V, is_deterministic=True)
    mdp.visualize_policy(stoch_policy, V, is_deterministic=False)


    # find bottleneck states for deterministic policy
    det_bottlenecks = mdp.find_bottleneck_states(det_policy)
    print("\nDeterministic Policy Bottleneck States:", det_bottlenecks)

    # find bottleneck states for stochastic policy
    stoch_bottlenecks = mdp.find_bottleneck_states(stoch_policy)
    print("\nStochastic Policy Bottleneck States:", stoch_bottlenecks)