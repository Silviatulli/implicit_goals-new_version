import numpy as np
from GridWorldClass import GridWorld
from Utils import ValueIteration
from DeterminizedMDP import DeterminizedMDP

""" The PuddleWorld class extends the GridWorld class 
    - it is a 2D grid where an agent can move in four directions: up, down, left, and right
    - the grid has a start state and a goal state
    - "puddles" are areas on the grid that the agent should avoid
    - puddles are randomly placed on the grid
    - moving into a puddle incurs a negative reward (penalty)
    - reaching the goal state provides a positive reward (1)
    - regular moves (not in puddles, not reaching the goal) have a zero reward
    - the agent's task is to learn a policy that navigates from the start to the goal while avoiding puddles as much as possible
    - the agent must balance taking the shortest path with avoiding puddles
    - the agent has 10% chance to slip and move in a direction different from the intended one
"""

class PuddleWorld(GridWorld):
    def __init__(self, size=5, start=None, goal=None, obstacles_percent=0.1,
                 puddle_percent=0.2, puddle_penalty=-1, goal_reward=1,
                 slip_prob=0.1, discount=0.99, max_tries=100, obstacle_seed=1):
        super().__init__(size=size, start=start, goal=goal, 
                         obstacles_percent=obstacles_percent,
                         slip_prob=slip_prob, discount=discount, 
                         max_tries=max_tries, obstacle_seed=obstacle_seed)
        
        self.puddle_percent = puddle_percent
        self.puddle_penalty = puddle_penalty
        self.goal_reward = goal_reward
        self.place_puddles()
        self.reward_func = self.puddle_reward_func

    def place_puddles(self):
        total_puddles = int(self.size * self.size * self.puddle_percent)
        for _ in range(total_puddles):
            x = np.random.randint(self.size)
            y = np.random.randint(self.size)
            if self.map[x, y] == 0:  # we only place puddles in empty spaces
                self.map[x, y] = 0.5  # use 0.5 to represent puddles

    def puddle_reward_func(self, state, action, next_state):
        x, y = next_state[0]
        if self.check_goal_reached(next_state):
            return self.goal_reward
        elif self.map[x, y] == 0.5:  # puddle
            return self.puddle_penalty
        else:
            return 0  # no reward for regular moves

    def visualize(self):
        """ layout of the PuddleWorld environment """
        for i in range(self.size):
            for j in range(self.size):
                if self.map[i, j] == -1:
                    print("█", end=" ")  # obstacle
                elif self.map[i, j] == 0.5:
                    print("~", end=" ")  # puddle
                elif (i, j) == self.start_pos:
                    print("S", end=" ")  # start
                elif (i, j) == self.goal_pos:
                    print("G", end=" ")  # goal
                else:
                    print(".", end=" ")  # empty
            print()

    def print_value_iteration_grid(self, num_iterations=100):
        """  result of running the Value Iteration algorithm on the PuddleWorld """
        V = ValueIteration(self, num_iterations)
        value_grid = np.zeros((self.size, self.size))

        for state in self.get_state_space():
            x, y = state[0]
            value_grid[x, y] = V[self.get_state_hash(state)]

        print("\nValue Iteration Grid:")
        for i in range(self.size):
            for j in range(self.size):
                if self.map[i, j] == -1:
                    print("  ##  ", end=" ")  # obstacle
                elif (i, j) == self.goal_pos:
                    print(" GOAL ", end=" ")  # goal
                else:
                    print(f"{value_grid[i, j]:6.2f}", end=" ")
            print()
    
    def get_bottleneck_states(self):
        det_mdp = DeterminizedMDP(self)
        init_state_hash = det_mdp.get_state_hash(det_mdp.get_init_state())
        det_mdp.reward_func = det_mdp.bottleneck_reward
        bottleneck_list = []
        for state in det_mdp.get_state_space():
            det_mdp.bottleneck_state = det_mdp.get_state_hash(state)
            V = ValueIteration(det_mdp)
            if V[init_state_hash] <= 0:
                bottleneck_list.append(state)
        return bottleneck_list

    def print_bottleneck_states(self):
        bottleneck_states = self.get_bottleneck_states()
        print("\nBottleneck States:")
        for state in bottleneck_states:
            print(f"- {state}")

        print("\nGrid with Bottleneck States:")
        for i in range(self.size):
            for j in range(self.size):
                if self.map[i, j] == -1:
                    print("█", end=" ")  # Obstacle
                elif self.map[i, j] == 0.5:
                    print("~", end=" ")  # Puddle
                elif (i, j) == self.start_pos:
                    print("S", end=" ")  # Start
                elif (i, j) == self.goal_pos:
                    print("G", end=" ")  # Goal
                elif any(state[0] == (i, j) for state in bottleneck_states):
                    print("B", end=" ")  # Bottleneck
                else:
                    print(".", end=" ")  # Empty
            print()

if __name__ == "__main__":
    puddle_world = PuddleWorld(size=5, start=(0, 0), goal=(4, 4), 
                               obstacles_percent=0.1, puddle_percent=0.2,
                               puddle_penalty=-1, goal_reward=10)
    print("PuddleWorld:")
    puddle_world.visualize()
    puddle_world.print_value_iteration_grid()
    puddle_world.print_bottleneck_states()