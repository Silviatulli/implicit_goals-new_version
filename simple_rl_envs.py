import numpy as np
from unified_mdp import UnifiedMDP, UnifiedState, ActionSpace
import random

class GridWorldUnifiedMDP(UnifiedMDP):
    def __init__(self, width, height, init_loc, goal_locs, walls=[], gamma=0.99, slip_prob=0.1):
        self.width = width
        self.height = height
        self.init_loc = init_loc
        self.goal_locs = goal_locs
        self.walls = walls
        self.slip_prob = slip_prob
        
        state_space = UnifiedState([(x, y) for x in range(width) for y in range(height) if (x, y) not in walls], is_discrete=True)
        action_space = ActionSpace(["up", "down", "left", "right"])
        
        super().__init__(state_space, action_space, self._transition_func, self._reward_func, gamma)
        
        self.initial_state = UnifiedState(init_loc, is_discrete=True)
        self.n_states = len(state_space.data)

    def _transition_func(self, state, action):
        x, y = state.data
        
        if random.random() < self.slip_prob:
            # Agent slips and moves in a random direction
            action = random.choice(self.action_space.actions)
        
        new_x, new_y = x, y  # Initialize new_x and new_y with current position

        if action == "up":
            new_y = min(y + 1, self.height - 1)
        elif action == "down":
            new_y = max(y - 1, 0)
        elif action == "left":
            new_x = max(x - 1, 0)
        elif action == "right":
            new_x = min(x + 1, self.width - 1)
        
        # Check if the new position is a wall
        if (new_x, new_y) in self.walls:
            new_x, new_y = x, y  # If it's a wall, stay in the current position
        
        return UnifiedState((new_x, new_y), is_discrete=True)

    def _reward_func(self, state, action, next_state):
        if state is None or next_state is None:
            return 0.0  # Return a default reward if state or next_state is None
        
        if isinstance(next_state, UnifiedState):
            next_state_data = next_state.data
        else:
            next_state_data = next_state
        
        if next_state_data in self.goal_locs:
            return 1.0
        else:
            return 0.0

    def is_terminal(self, state):
        return state.data in self.goal_locs

    def reset(self):
        self.current_state = self.initial_state
        return self.current_state

class TaxiUnifiedMDP(UnifiedMDP):
    def __init__(self, width, height, passenger_locs, destination_locs, gamma=0.99):
        self.width = width
        self.height = height
        self.passenger_locs = passenger_locs
        self.destination_locs = destination_locs
        
        # State space: (taxi_x, taxi_y, passenger_idx, destination_idx)
        # passenger_idx: -1 if in taxi, otherwise index of passenger location
        state_space = UnifiedState(np.array([(x, y, p, d) 
                                    for x in range(width) 
                                    for y in range(height) 
                                    for p in range(-1, len(passenger_locs)) 
                                    for d in range(len(destination_locs))]), 
                                   is_discrete=True)
        
        action_space = ActionSpace(["up", "down", "left", "right", "pickup", "dropoff"])
        
        super().__init__(state_space, action_space, self._transition_func, self._reward_func, gamma)

    def _transition_func(self, state, action):
        taxi_x, taxi_y, passenger_idx, destination_idx = state.data
        
        new_taxi_x, new_taxi_y = taxi_x, taxi_y
        new_passenger_idx = passenger_idx

        if action == "up" and taxi_y < self.height - 1:
            new_taxi_y += 1
        elif action == "down" and taxi_y > 0:
            new_taxi_y -= 1
        elif action == "left" and taxi_x > 0:
            new_taxi_x -= 1
        elif action == "right" and taxi_x < self.width - 1:
            new_taxi_x += 1
        elif action == "pickup" and passenger_idx != -1:
            if (taxi_x, taxi_y) == self.passenger_locs[passenger_idx]:
                new_passenger_idx = -1  # Passenger is in the taxi
        elif action == "dropoff" and passenger_idx == -1:
            if (taxi_x, taxi_y) == self.destination_locs[destination_idx]:
                new_passenger_idx = destination_idx  # Drop off successful
        
        new_state = np.array([new_taxi_x, new_taxi_y, new_passenger_idx, destination_idx])
        return UnifiedState(new_state, is_discrete=True)

    def _reward_func(self, state, action, next_state):
        if next_state is None:
            return 0  # Return 0 reward if next_state is None
        
        _, _, passenger_idx, destination_idx = state.data
        next_taxi_x, next_taxi_y, next_passenger_idx, next_destination_idx = next_state.data
        
        if action == "dropoff" and passenger_idx == -1 and next_passenger_idx == destination_idx:
            return 20  # Successful dropoff
        elif action == "pickup" and passenger_idx != -1 and next_passenger_idx == -1:
            return 0  # Successful pickup
        else:
            return -1  # Time penalty

    def is_terminal(self, state):
        _, _, passenger_idx, destination_idx = state.data
        return passenger_idx == destination_idx and passenger_idx != -1

    def reset(self):
        # Start with taxi at a random location and passenger at a random pickup location
        taxi_x = np.random.randint(0, self.width)
        taxi_y = np.random.randint(0, self.height)
        passenger_idx = np.random.randint(0, len(self.passenger_locs))
        destination_idx = np.random.randint(0, len(self.destination_locs))
        self.current_state = UnifiedState(np.array([taxi_x, taxi_y, passenger_idx, destination_idx]), is_discrete=True)
        return self.current_state
    

class PuddleUnifiedMDP(UnifiedMDP):
    def __init__(self, puddle_rects, goal_locs, gamma=0.99, step_size=0.05, grid_size=100):
        self.puddle_rects = puddle_rects
        self.goal_locs = goal_locs
        self.step_size = step_size
        self.grid_size = grid_size
        
        # Create a grid of states
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        self.states = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        
        state_space = UnifiedState(self.states, is_discrete=False)
        action_space = ActionSpace(["up", "down", "left", "right"])
        
        super().__init__(state_space, action_space, self._transition_func, self._reward_func, gamma)

        print(f"Initialized PuddleUnifiedMDP with {len(self.states)} states")
        print(f"Puddle rectangles: {self.puddle_rects}")
        print(f"Goal locations: {self.goal_locs}")

    def _transition_func(self, state, action):
        if state is None:
            print("Warning: state is None in _transition_func")
            return self.reset()  # Return to a valid initial state

        x, y = state.data
        
        if action == "up":
            y = min(y + self.step_size, 1.0)
        elif action == "down":
            y = max(y - self.step_size, 0.0)
        elif action == "left":
            x = max(x - self.step_size, 0.0)
        elif action == "right":
            x = min(x + self.step_size, 1.0)
        
        # Add some noise to the movement
        x += np.random.normal(0, 0.01)
        y += np.random.normal(0, 0.01)
        
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        # Find the nearest state in the grid
        next_state = self._find_nearest_state(x, y)
        
        return UnifiedState(next_state, is_discrete=False)

    def _find_nearest_state(self, x, y):
        distances = np.sqrt(np.sum((self.states - np.array([x, y]))**2, axis=1))
        nearest_index = np.argmin(distances)
        return self.states[nearest_index]

    def _reward_func(self, state, action, next_state):
        if state is None:
            print("Warning: state is None in _reward_func")
            return 0

        if next_state is None:
            print("Warning: next_state is None in _reward_func")
            next_state = state  # Use current state if next_state is None

        x, y = next_state.data
        
        # Check if in puddle
        in_puddle = any(rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3] 
                        for rect in self.puddle_rects)
        
        if in_puddle:
            return -1  # Penalty for being in a puddle
        elif any(abs(x - goal[0]) < self.step_size and abs(y - goal[1]) < self.step_size 
                for goal in self.goal_locs):
            return 1  # Reward for reaching the goal
        else:
            return -0.1  # Small penalty for each step

    def is_terminal(self, state):
        if state is None:
            print("Warning: state is None in is_terminal")
            return False

        x, y = state.data
        return any(abs(x - goal[0]) < self.step_size and abs(y - goal[1]) < self.step_size 
                   for goal in self.goal_locs)

    def reset(self):
        # Start in a random location that's not in a puddle
        while True:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            if not any(rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3] 
                       for rect in self.puddle_rects):
                self.current_state = UnifiedState(self._find_nearest_state(x, y), is_discrete=False)
                return self.current_state

    def step(self, action):
        if self.current_state is None:
            print("Warning: current_state is None in step")
            self.reset()

        next_state = self._transition_func(self.current_state, action)
        reward = self._reward_func(self.current_state, action, next_state)
        done = self.is_terminal(next_state)
        info = {}

        self.current_state = next_state
        return next_state, reward, done, info
            

# Example usage:
grid_world_mdp = GridWorldUnifiedMDP(5, 5, (0, 0), [(4, 4)], walls=[(1, 1), (2, 2), (3, 3)])
taxi_mdp = TaxiUnifiedMDP(5, 5, [(0, 0), (4, 4)], [(0, 4), (4, 0)])
puddle_mdp = PuddleUnifiedMDP([(0.1, 0.8, 0.5, 0.7), (0.4, 0.7, 0.5, 0.4)], [(1.0, 1.0)])