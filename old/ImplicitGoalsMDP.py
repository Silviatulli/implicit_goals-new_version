import numpy as np
from itertools import product
from collections import deque



class ImplicitGoalsMDP:
    def __init__(self, base_mdp, potential_subgoals, R1=10, R2=100):
        self.base_mdp = base_mdp
        self.subgoals = [g - 1 for g in potential_subgoals]  # Convert to 0-indexed
        self.n_subgoals = len(self.subgoals)
        self.R1 = R1  # Base reward for visiting a subgoal
        self.R2 = R2  # Additional reward for reaching the goal state
        
        self.states = list(product(range(base_mdp.n_states), product([0, 1], repeat=self.n_subgoals)))
        self.n_states = len(self.states)
        self.n_actions = base_mdp.n_actions
        
        self.initial_state = self.states.index((base_mdp.initial_state, tuple([0] * self.n_subgoals)))
        self.goal_states = [i for i, (s, v) in enumerate(self.states) if s == base_mdp.goal_state]

        self.goal_state = base_mdp.goal_state
        self._build_transition_and_reward()
        
        # print(f"Initialized BottleneckMDP with {self.n_states} states and {self.n_actions} actions")
        # print(f"Initial state: {self.initial_state}, Goal states: {self.goal_states}")
        # print(f"Subgoals: {self.subgoals}")

    def _build_transition_and_reward(self):
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        for i, (s, v) in enumerate(self.states):
            for a in range(self.n_actions):
                for j, (s_next, v_next) in enumerate(self.states):
                    if self.base_mdp.P[s, a, s_next] > 0:
                        new_v = list(v)
                        if s_next in self.subgoals:
                            new_v[self.subgoals.index(s_next)] = 1
                        if tuple(new_v) == v_next:
                            self.P[i, a, j] = self.base_mdp.P[s, a, s_next]
                            self.R[i, a, j] = self.base_mdp.R[s, a, s_next]
                            
                            # Calculate additional reward based on number of subgoals visited
                            subgoals_visited = sum(v_next)
                            additional_reward = self._calculate_subgoal_reward(subgoals_visited)
                            self.R[i, a, j] += additional_reward

                            # Extra reward for reaching the goal state
                            if s_next == self.base_mdp.goal_state:
                                self.R[i, a, j] += self.R2 * (subgoals_visited / len(self.subgoals))

    def _calculate_subgoal_reward(self, subgoals_visited):
        # Calculates the reward based on the number of subgoals visited
        # The reward increases non-linearly with more subgoals visited
        base_reward = self.R1
        scaling_factor = 1.2  # Adjust this to change how quickly the reward increases
        return base_reward * (subgoals_visited ** scaling_factor)

    def _get_next_state(self, state, action):
        row, col = divmod(state, self.base_mdp.n_cols)
        if action == 0:  # Up
            next_row, next_col = row - 1, col
        elif action == 1:  # Down
            next_row, next_col = row + 1, col
        elif action == 2:  # Left
            next_row, next_col = row, col - 1
        elif action == 3:  # Right
            next_row, next_col = row, col + 1
        
        if 0 <= next_row < self.base_mdp.n_rows and 0 <= next_col < self.base_mdp.n_cols:
            next_state = self.base_mdp.grid[next_row][next_col] - 1  # Convert to 0-indexed
            if next_state != -1 or next_state !=0:  # Not a wall
                return next_state
        return None  # Wall or out of bounds

    # def value_iteration(self, gamma=0.99, epsilon=1e-6, max_iterations=1000, stochastic=False):
    #     V = np.zeros(self.n_states)
    #     for _ in range(max_iterations):
    #         V_new = np.zeros_like(V)
    #         for s in range(self.n_states):
    #             V_new[s] = max([np.sum(self.P[s, a] * (self.R[s, a] + gamma * V)) for a in range(self.n_actions)])
    #         if np.max(np.abs(V_new - V)) < epsilon:
    #             break
    #         V = V_new
        
    #     if stochastic:
    #         policy = np.zeros((self.n_states, self.n_actions))
    #         for s in range(self.n_states):
    #             Q = np.array([np.sum(self.P[s, a] * (self.R[s, a] + gamma * V)) for a in range(self.n_actions)])
    #             exp_Q = np.exp(Q - np.max(Q)) 
    #             policy[s] = exp_Q / np.sum(exp_Q)  # Softmax distribution
    #     else:
    #         policy = np.zeros(self.n_states, dtype=int)
    #         for s in range(self.n_states):
    #             policy[s] = np.argmax([np.sum(self.P[s, a] * (self.R[s, a] + gamma * V)) for a in range(self.n_actions)])
        
    #     return V, policy
    
    def value_iteration(self, gamma=0.99, theta=1e-6, max_iterations=500):
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
                    Q_new[s, :] = self.R[s, :, self.goal_state]
                else:
                    for a in range(self.n_actions):
                        Q_new[s, a] = np.sum(self.P[s, a] * (self.R[s, a] + gamma * V))
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
        policy[self.goal_state] = np.random.randint(self.n_actions)  # Random action for goal state
        return policy

    def compute_stochastic_policy(self, Q):
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
                policy[s] = np.ones(self.n_actions) / self.n_actions  # Uniform distribution for goal state
            else:
                exp_Q = np.exp((Q[s] - np.max(Q[s])))  # Subtract max(Q) for numerical stability
                policy[s] = exp_Q / np.sum(exp_Q)
        return policy

    def get_optimal_policies(self):
        """
        Compute both deterministic and stochastic optimal policies.
        Args:
            temperature (float): Temperature parameter for the stochastic policy.
        Returns:
            tuple: (deterministic policy, stochastic policy, V-function)
        """
        Q, V = self.value_iteration()
        deterministic_policy = self.compute_deterministic_policy(Q)
        stochastic_policy = self.compute_stochastic_policy(Q)
        return deterministic_policy, stochastic_policy, V


    def find_all_traces(self, policy, stochastic, max_depth=1000):
        all_traces = []
        queue = deque([(self.initial_state, [])])
        visited = set()
        
        while queue:
            current_state_idx, current_trace = queue.popleft()
            current_base_state, current_visitation = self.states[current_state_idx]
            
            # print(f"Exploring state {current_state_idx}: base state {current_base_state}, visitation {current_visitation}")
            
            if current_state_idx in self.goal_states:
                all_traces.append(current_trace + [(current_state_idx, None)])
                # print(f"Goal reached. Trace: {[self.states[s][0] for s, _ in current_trace + [(current_state_idx, None)]]}")
                continue
            
            if len(current_trace) >= max_depth or current_state_idx in visited:
                continue
            
            visited.add(current_state_idx)
            
            if stochastic:
                possible_actions = np.where(policy[current_state_idx] > 0)[0]
            else:
                possible_actions = [policy[current_state_idx]]
                        
            for action in possible_actions:
                for next_state_idx in range(self.n_states):
                    if self.P[current_state_idx, action, next_state_idx] > 0:
                        new_trace = current_trace + [(current_state_idx, action)]
                        # print(f"Transition found: {current_state_idx} --({action})--> {next_state_idx}")
                        queue.append((next_state_idx, new_trace))
        
        print(f"Found {len(all_traces)} traces")
        # for i, trace in enumerate(all_traces):
        #     print(f"Trace {i + 1}: {[self.states[s][0] for s, _ in trace]}")
        return all_traces


    def find_maximal_bottleneck_goals(self, policy, stochastic=False):
        
        traces = self.find_all_traces(policy, stochastic)
        
        if not traces:
            print("No valid traces found")
            return []
        
        bottleneck_goals = set(self.subgoals)
        for trace in traces:
            trace_states = [self.states[s][0] for s, _ in trace]
            trace_subgoals = set(s for s in trace_states if s in self.subgoals)
            # print(f"Subgoals in trace: {trace_subgoals}")
            bottleneck_goals &= trace_subgoals
        
        # print(f"Final set of bottleneck goals: {bottleneck_goals}")
        return [g + 1 for g in bottleneck_goals]  # Convert back to 1-indexed
    

    def find_maximal_bottleneck_goals_threshold(self, policy, stochastic=False, threshold=0.95):
        
        traces = self.find_all_traces(policy, stochastic=stochastic)
        
        if not traces:
            print("No valid traces found")
            return []
        
        subgoal_counts = {subgoal: 0 for subgoal in self.subgoals}
        total_traces = len(traces)
        
        for trace in traces:
            trace_states = {self.states[s][0] for s, _ in trace}
            for subgoal in self.subgoals:
                if subgoal in trace_states:
                    subgoal_counts[subgoal] += 1
        
        bottleneck_goals = {subgoal for subgoal, count in subgoal_counts.items() if count / total_traces >= threshold}
        
        print("Subgoal frequencies:")
        for subgoal, count in subgoal_counts.items():
            print(f"State {subgoal + 1}: {count}/{total_traces} ({count/total_traces:.2%})")
        
        print(f"Final set of bottleneck goals: {bottleneck_goals}")
        return [g + 1 for g in bottleneck_goals]  # Convert back to 1-indexed
        

# Example usage
if __name__ == "__main__":
    from new_gridMDP import GridWorldMDP
    
    grid = [
        [1, 2, 3, 4, 0],
        [0, 0, 6, 0, 0],
        [7, 8, 9, 10, 11],
        [0, 0, 0, 12, 14],
        [15, 16, 17, 18, 19]
    ]
    

    initial_state = 1
    goal_state = 19
    potential_subgoals = [4, 15, 7]
    
    base_mdp = GridWorldMDP(grid, initial_state, goal_state)
    bottleneck_mdp = ImplicitGoalsMDP(base_mdp, potential_subgoals, R1=1, R2=10)

    # Get deterministic and stochastic optimal policies
    det_policy, stoch_policy, V = bottleneck_mdp.get_optimal_policies()

    # First method uses intersection of subgoals visited in all traces
    print("\nFirst method: Find the intersection of subgoals visited in all traces.")
    # Deterministic policy
    maximal_bottleneck_goals = bottleneck_mdp.find_maximal_bottleneck_goals(det_policy, stochastic=False)
    print(f"Maximal set of bottleneck goals (deterministic): {maximal_bottleneck_goals}")
    
    # Stochastic policy
    # maximal_bottleneck_goals_stochastic = bottleneck_mdp.find_maximal_bottleneck_goals(stochastic=True)
    maximal_bottleneck_goals_stochastic = bottleneck_mdp.find_maximal_bottleneck_goals(stoch_policy, stochastic=True)
    print(f"Maximal set of bottleneck goals (stochastic): {maximal_bottleneck_goals_stochastic}")

    print("\n\n")

    # Second method uses a threshold to determine if a subgoal is a bottleneck goal
    print("\nSecond method: Find the subgoals that are bottleneck goals based on a threshold.")
    threshold = 0.6 # % of traces to visit the subgoal 

    # Deterministic policy
    maximal_bottleneck_goals_stochastic = bottleneck_mdp.find_maximal_bottleneck_goals_threshold(det_policy, stochastic=False, threshold=threshold)
    print(f"Maximal set of bottleneck goals (deterministic, threshold={threshold}): {maximal_bottleneck_goals_stochastic}")

    # Stochastic policy
    maximal_bottleneck_goals_stochastic = bottleneck_mdp.find_maximal_bottleneck_goals_threshold(stoch_policy, stochastic=True, threshold=threshold)
    print(f"Maximal set of bottleneck goals (stochastic, threshold={threshold}): {maximal_bottleneck_goals_stochastic}")


