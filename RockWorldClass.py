import numpy as np
from GridWorldClass import GridWorld
from Utils import ValueIteration
from DeterminizedMDP import DeterminizedMDP

""" The RockWorld class extends the GridWorld class
    - there are two types of rocks - valuable (represented by 1 in the map) and dangerous (represented by 2).
    - rocks are placed randomly on empty cells with a specified percentage and ratio of valuable to dangerous rocks.
    - collecting a valuable rock gives a positive reward.
    - stepping on a dangerous rock incurs a penalty.
    - reaching the goal gives the same reward as collecting a valuable rock.
    - each move has a small negative reward to encourage efficiency.
    - valuable rocks are removed from the grid once collected. """

class RockWorld(GridWorld):
    def __init__(self, size=5, start=None, goal=None, obstacles_percent=0.1,
                 rock_percent=0.3, valuable_rock_ratio=0.4, 
                 valuable_rock_reward=10, dangerous_rock_penalty=-5,
                 slip_prob=0.1, discount=0.99, max_tries=100, obstacle_seed=1):
        super().__init__(size=size, start=start, goal=goal, 
                         obstacles_percent=obstacles_percent,
                         slip_prob=slip_prob, discount=discount, 
                         max_tries=max_tries, obstacle_seed=obstacle_seed)
        
        self.rock_percent = rock_percent
        self.valuable_rock_ratio = valuable_rock_ratio
        self.valuable_rock_reward = valuable_rock_reward
        self.dangerous_rock_penalty = dangerous_rock_penalty
        self.place_rocks()
        self.reward_func = self.rock_reward_func

    def place_rocks(self):
        total_rocks = int(self.size * self.size * self.rock_percent)
        valuable_rocks = int(total_rocks * self.valuable_rock_ratio)
        dangerous_rocks = total_rocks - valuable_rocks

        for _ in range(valuable_rocks):
            self.place_rock(1)  # 1 represents valuable rock
        for _ in range(dangerous_rocks):
            self.place_rock(2)  # 2 represents dangerous rock

    def place_rock(self, rock_type):
        while True:
            x = np.random.randint(self.size)
            y = np.random.randint(self.size)
            if self.map[x, y] == 0:  # only place rocks in empty spaces
                self.map[x, y] = rock_type
                break

    def rock_reward_func(self, state, action, next_state):
        x, y = next_state[0]
        if self.check_goal_reached(next_state):
            return self.valuable_rock_reward  # reaching the goal is as rewarding as collecting a valuable rock
        elif self.map[x, y] == 1:  # valuable rock
            self.map[x, y] = 0  # remove the rock after collection
            return self.valuable_rock_reward
        elif self.map[x, y] == 2:  # dangerous rock
            return self.dangerous_rock_penalty
        else:
            return -1  # small penalty for each move to encourage efficiency

    def visualize(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.map[i, j] == -1:
                    print("█", end=" ")  # obstacle
                elif self.map[i, j] == 1:
                    print("V", end=" ")  # valuable rock
                elif self.map[i, j] == 2:
                    print("D", end=" ")  # dangerous rock
                elif (i, j) == self.start_pos:
                    print("S", end=" ")  # start
                elif (i, j) == self.goal_pos:
                    print("G", end=" ")  # goal
                else:
                    print(".", end=" ")  # empty
            print()

    def print_value_iteration_grid(self, num_iterations=100):
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
            x, y = state[0]
            cell_type = "Empty"
            if self.map[x, y] == 1:
                cell_type = "Valuable Rock"
            elif self.map[x, y] == 2:
                cell_type = "Dangerous Rock"
            elif self.map[x, y] == -1:
                cell_type = "Obstacle"
            print(f"- {state}: {cell_type}")

        print("\nGrid with Bottleneck States:")
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.start_pos:
                    print("S", end=" ")  # Start
                elif (i, j) == self.goal_pos:
                    print("G", end=" ")  # Goal
                elif any(state[0] == (i, j) for state in bottleneck_states):
                    print("B", end=" ")  # Bottleneck
                elif self.map[i, j] == -1:
                    print("█", end=" ")  # Obstacle
                elif self.map[i, j] == 1:
                    print("V", end=" ")  # Valuable rock
                elif self.map[i, j] == 2:
                    print("D", end=" ")  # Dangerous rock
                else:
                    print(".", end=" ")  # Empty
            print()

if __name__ == "__main__":
    rock_world = RockWorld(size=8, start=(0, 0), goal=(7, 7), 
                           obstacles_percent=0.1, rock_percent=0.3,
                           valuable_rock_ratio=0.4, valuable_rock_reward=10,
                           dangerous_rock_penalty=-5)
    print("RockWorld:")
    rock_world.visualize()
    rock_world.print_value_iteration_grid()
    rock_world.print_bottleneck_states()