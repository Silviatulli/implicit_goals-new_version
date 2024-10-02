import numpy as np
from GridWorldClass import GridWorld
from Utils import ValueIteration
from DeterminizedMDP import DeterminizedMDP

"""
The TaxiWorld class extends the GridWorld class
- It is a 2D grid where a taxi can move in four directions: up, down, left, and right
- The grid has a start state, a passenger location, and a destination
- Obstacles are randomly placed on the grid
- The taxi's task is to pick up the passenger and drop them off at the destination
- Actions available: move (up, down, left, right), pickup, and dropoff
- Rewards:
  - Successfully dropping off the passenger at the destination: +20
  - Picking up the passenger: 0
  - Dropping off the passenger at the wrong location: -10
  - Any other action: -1 (time penalty)
- The taxi has a 10% chance to slip and move in a direction different from the intended one
- The agent must balance taking the shortest path with efficiently picking up and dropping off the passenger
"""

class TaxiWorld(GridWorld):
    def __init__(self, size=5, start=None, passenger_loc=None, destination=None, 
                 obstacles_percent=0.1, slip_prob=0.1, discount=0.99, max_tries=100, 
                 obstacle_seed=1):
        self.size = size  # we need to set size before calling place_random_location
        self.passenger_loc = passenger_loc if passenger_loc is not None else self.place_random_location()
        
        super().__init__(size=size, start=start, goal=destination, 
                         obstacles_percent=obstacles_percent, slip_prob=slip_prob, 
                         discount=discount, max_tries=max_tries, obstacle_seed=obstacle_seed)
        
        self.destination = self.goal_pos  # reuse goal_pos from GridWorld as destination
        self.reward_func = self.taxi_reward_func  # override reward function

    def place_random_location(self):
        while True:
            x, y = np.random.randint(self.size), np.random.randint(self.size)
            if not hasattr(self, 'map') or self.map[x, y] != -1:
                return (x, y)

    def place_start_and_goal(self):
        super().place_start_and_goal()

    def get_actions(self):
        return super().get_actions() + ["pickup", "dropoff"]

    def create_state_space(self):
        self.state_space = []
        for i in range(self.size):
            for j in range(self.size):
                for passenger_in_taxi in [False, True]:
                    self.state_space.append([(i, j), passenger_in_taxi])

    def get_transition_probability(self, state, action, state_prime):
        x, y = state[0]
        passenger_in_taxi = state[1]
        x_prime, y_prime = state_prime[0]
        passenger_in_taxi_prime = state_prime[1]

        if action in ["up", "down", "left", "right"]:
            move_prob = super().get_transition_probability(state, action, state_prime)
            return move_prob if passenger_in_taxi == passenger_in_taxi_prime else 0

        elif action == "pickup":
            if (x, y) != self.passenger_loc or passenger_in_taxi:
                return 1 if state == state_prime else 0
            else:
                return 1 if state_prime == [(x, y), True] else 0

        elif action == "dropoff":
            if not passenger_in_taxi:
                return 1 if state == state_prime else 0
            else:
                return 1 if state_prime == [(x, y), False] else 0

        return 0  # default case, should not reach here

    def taxi_reward_func(self, state, action, next_state):
        if action == "dropoff" and next_state[0] == self.destination and state[1] and not next_state[1]:
            return 20  # successfully dropped off passenger at destination
        elif action == "pickup" and state[0] == self.passenger_loc and not state[1] and next_state[1]:
            return 0  # successfully picked up passenger
        elif action == "dropoff" and state[1] and not next_state[1]:
            return -10  # dropped off passenger at wrong location
        else:
            return -1  # time penalty for each action

    def get_init_state(self):
        return [self.start_pos, False]  # taxi starts without passenger

    def visualize(self):
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.start_pos:
                    print("T", end=" ")
                elif (i, j) == self.passenger_loc:
                    print("P", end=" ")
                elif (i, j) == self.destination:
                    print("D", end=" ")
                elif self.map[i, j] == -1:
                    print("█", end=" ")
                else:
                    print(".", end=" ")
            print()

    def get_goal_states(self):
        return [[self.destination, False]]  # passenger dropped off at destination

    def print_value_iteration_grid(self, num_iterations=100):
        V = ValueIteration(self, num_iterations)
        value_grid = np.zeros((self.size, self.size))

        for state in self.get_state_space():
            x, y = state[0]
            value_grid[x, y] = max(V[self.get_state_hash(state)], value_grid[x, y])

        print("\nValue Iteration Grid:")
        for i in range(self.size):
            for j in range(self.size):
                if self.map[i, j] == -1:
                    print("  ##  ", end=" ")  # obstacle
                elif (i, j) == self.destination:
                    print(" DEST ", end=" ")  # destination
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
                    print("█", end=" ")  # obstacle
                elif (i, j) == self.start_pos:
                    print("T", end=" ")  # taxi start
                elif (i, j) == self.passenger_loc:
                    print("P", end=" ")  # passenger
                elif (i, j) == self.destination:
                    print("D", end=" ")  # destination
                elif any(state[0] == (i, j) for state in bottleneck_states):
                    print("B", end=" ")  # bottleneck
                else:
                    print(".", end=" ")  # empty
            print()

if __name__ == "__main__":
    taxi_world = TaxiWorld(size=5, start=(0, 0), passenger_loc=(2, 2), destination=(4, 4), 
                           obstacles_percent=0.1)
    print("TaxiWorld:")
    taxi_world.visualize()
    taxi_world.print_value_iteration_grid()
    taxi_world.print_bottleneck_states()