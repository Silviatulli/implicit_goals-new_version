import numpy as np
import random

class UnifiedState:
    def __init__(self, state_data, is_discrete=True):
        self.data = state_data  # Store as-is, without converting to numpy array
        self.is_discrete = is_discrete

    def __hash__(self):
        if self.is_discrete:
            return hash(tuple(self.data)) if isinstance(self.data, list) else hash(self.data)
        return None

    def __eq__(self, other):
        return self.data == other.data
    
class ActionSpace:
    def __init__(self, actions):
        self.actions = actions

    def sample(self):
        return random.choice(self.actions)

class UnifiedMDP:
    def __init__(self, state_space, action_space, transition_func, reward_func, gamma):
        self.state_space = state_space
        self.action_space = action_space
        self.transition_func = transition_func
        self.reward_func = reward_func
        self.gamma = gamma
        self.current_state = None

    def reset(self):
        if hasattr(self.state_space, 'sample'):
            self.current_state = self.state_space.sample()
        else:
            self.current_state = UnifiedState(np.random.choice(self.state_space.data), is_discrete=True)
        return self.current_state

    def step(self, action):
        if self.current_state is None:
            raise ValueError("Call reset() before calling step()")

        next_state = self.transition_func(self.current_state, action)
        reward = self.reward_func(self.current_state, action, next_state)
        done = self.is_terminal(next_state)
        info = {}

        self.current_state = next_state
        return next_state, reward, done, info

    def is_terminal(self, state):
        if hasattr(state, 'is_terminal'):
            return state.is_terminal()
        elif hasattr(self, 'goal_locs'):
            return state.data in self.goal_locs
        else:
            return False

    def vectorized_transition(self, states, action):
        n_states = len(states)
        transition_matrix = np.zeros((n_states, n_states))
        
        for i, state in enumerate(states):
            next_state = self.transition_func(state, action)
            j = states.index(next_state) if next_state in states else i
            transition_matrix[i, j] = 1.0
        
        return transition_matrix

    def vectorized_reward(self, states, action, next_states):
        return np.array([self.reward_func(state, action, next_state) 
                         for state, next_state in zip(states, next_states)])

class GridWorldUnifiedMDP(UnifiedMDP):
    def __init__(self, width, height, init_loc, goal_locs, walls=[], gamma=0.99):
        self.width = width
        self.height = height
        self.init_loc = init_loc
        self.goal_locs = goal_locs
        self.walls = walls
        
        state_space = UnifiedState([(x, y) for x in range(width) for y in range(height) if (x, y) not in walls], is_discrete=True)
        action_space = ActionSpace(["up", "down", "left", "right"])
        
        super().__init__(state_space, action_space, self._transition_func, self._reward_func, gamma)
        
        self.goal_state = UnifiedState(goal_locs[0], is_discrete=True)

    def _transition_func(self, state, action):
        x, y = state.data
        new_x, new_y = x, y
        if action == "up":
            new_y = min(y + 1, self.height - 1)
        elif action == "down":
            new_y = max(y - 1, 0)
        elif action == "left":
            new_x = max(x - 1, 0)
        elif action == "right":
            new_x = min(x + 1, self.width - 1)
        
        if (new_x, new_y) in self.walls:
            new_x, new_y = x, y
        
        return UnifiedState((new_x, new_y), is_discrete=True)

    def _reward_func(self, state, action, next_state):
        if next_state.data in self.goal_locs:
            return 1.0
        else:
            return 0.0

    def is_terminal(self, state):
        return state.data in self.goal_locs

class TaxiUnifiedMDP(UnifiedMDP):
    def __init__(self, width, height, passenger_locs, destination_locs, gamma=0.99):
        self.width = width
        self.height = height
        self.passenger_locs = passenger_locs
        self.destination_locs = destination_locs
        
        state_space = UnifiedState([(x, y, p, d) 
                                    for x in range(width) 
                                    for y in range(height) 
                                    for p in range(-1, len(passenger_locs)) 
                                    for d in range(len(destination_locs))], 
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
        
        new_state = (new_taxi_x, new_taxi_y, new_passenger_idx, destination_idx)
        return UnifiedState(new_state, is_discrete=True)

    def _reward_func(self, state, action, next_state):
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

class PuddleUnifiedMDP(UnifiedMDP):
    def __init__(self, puddle_rects, goal_locs, gamma=0.99, step_size=0.05):
        self.puddle_rects = puddle_rects
        self.goal_locs = goal_locs
        self.step_size = step_size
        
        state_space = UnifiedState([(x, y) for x in np.arange(0, 1.01, 0.01) 
                                    for y in np.arange(0, 1.01, 0.01)], 
                                   is_discrete=False)
        
        action_space = ActionSpace(["up", "down", "left", "right"])
        
        super().__init__(state_space, action_space, self._transition_func, self._reward_func, gamma)

    def _transition_func(self, state, action):
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
        
        return UnifiedState((x, y), is_discrete=False)

    def _reward_func(self, state, action, next_state):
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
        x, y = state.data
        return any(abs(x - goal[0]) < self.step_size and abs(y - goal[1]) < self.step_size 
                   for goal in self.goal_locs)