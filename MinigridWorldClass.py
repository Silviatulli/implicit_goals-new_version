import numpy as np
from typing import List, Tuple, Dict
from MDP import MDP
from Utils import *



class UnlockEnv(MDP):
    def __init__(self, grid_size=5, max_steps=100, slip_prob=0.1, 
                 init_state=None, goal_states=None):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.slip_prob = slip_prob
        self.step_count = 0
        self.discount = 0.99
        
        self.door_pos = (grid_size - 1, grid_size - 1)
        self.key_pos = (0, grid_size - 1)
        
        self.init_state = init_state if init_state else ((0, 0), (False, 0), (2,))
        self.current_state = self.init_state
        
        self.goal_states = goal_states if goal_states else [
            ((self.door_pos[0], self.door_pos[1]), (True, d), (0,)) 
            for d in range(4)
        ]
        
        self.actions = ["left", "right", "forward", "toggle", "pickup"]
        self.map = np.zeros((grid_size, grid_size))
        self.reward_func = self.get_reward
        self.create_state_space()

    def create_state_space(self):
        """Generate complete state space"""
        self.state_space = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for has_key in [False, True]:
                    for direction in range(4):
                        for door_state in [0, 1, 2]:
                            pos = (x, y)
                            if pos == self.door_pos and door_state > 0 and not has_key:
                                continue  # Can't be at closed/locked door without key
                            self.state_space.append((pos, (has_key, direction), (door_state,)))

    def get_next_states(self, state, action):
        """Helper function to get all possible next states"""
        next_states = []
        (x, y), (has_key, direction), (door_state,) = state
        
        if action == "pickup" and (x, y) == self.key_pos and not has_key:
            # If at key position and don't have key, can pick it up
            next_state = ((x, y), (True, direction), (door_state,))
            next_states.append((next_state, 1.0))
        elif action == "toggle" and (x, y) == self.door_pos and has_key and door_state > 0:
            # If at door with key and door not open, can change door state
            next_state = ((x, y), (has_key, direction), (door_state - 1,))
            next_states.append((next_state, 1.0))
        elif action == "forward":
            dx, dy = [(0,-1), (1,0), (0,1), (-1,0)][direction]
            intended_pos = (x + dx, y + dy)
            # Check if movement is valid
            if (0 <= intended_pos[0] < self.grid_size and 
                0 <= intended_pos[1] < self.grid_size and 
                not (intended_pos == self.door_pos and door_state > 0 and not has_key)):
                # Can move to intended position
                next_state = (intended_pos, (has_key, direction), (door_state,))
                next_states.append((next_state, 1.0 - self.slip_prob))
                # Or stay in place due to slip
                next_states.append((state, self.slip_prob))
            else:
                # Invalid move, stay in place
                next_states.append((state, 1.0))
        elif action in ["left", "right"]:
            # Rotation
            next_dir = (direction - 1) % 4 if action == "left" else (direction + 1) % 4
            next_state = ((x, y), (has_key, next_dir), (door_state,))
            next_states.append((next_state, 1.0))
        else:
            # Stay in place for invalid actions
            next_states.append((state, 1.0))
            
        return next_states

    def get_transition_probability(self, state, action, next_state):
        """Calculate transition probability using get_next_states"""
        # Goal state is absorbing
        if state in self.goal_states:
            return 1.0 if state == next_state else 0.0
            
        # Get all possible next states and their probabilities
        next_states = self.get_next_states(state, action)
        
        # Find probability for the specific next_state
        for possible_next, prob in next_states:
            if possible_next == next_state:
                return prob
                
        return 0.0

    def get_reward(self, state, action, next_state):
        """Reward function with better incentives"""
        (x, y), (has_key, _), (door_state,) = state
        next_pos, (next_has_key, _), (next_door_state,) = next_state
        
        # Goal state reward
        if next_state in self.goal_states:
            return 1000.0
            
        # Key pickup reward
        if next_has_key and not has_key:
            return 100.0
            
        # Door progress reward
        if next_door_state < door_state:
            return 50.0
            
        # Movement rewards
        reward = 0.0
        if not has_key:
            # Distance to key if we don't have it
            curr_key_dist = abs(x - self.key_pos[0]) + abs(y - self.key_pos[1])
            next_key_dist = abs(next_pos[0] - self.key_pos[0]) + abs(next_pos[1] - self.key_pos[1])
            reward += 10.0 * (curr_key_dist - next_key_dist)
        else:
            # Distance to door if we have key
            curr_door_dist = abs(x - self.door_pos[0]) + abs(y - self.door_pos[1])
            next_door_dist = abs(next_pos[0] - self.door_pos[0]) + abs(next_pos[1] - self.door_pos[1])
            reward += 10.0 * (curr_door_dist - next_door_dist)
        
        # Small step penalty
        reward -= 1.0
        
        return reward

    def render(self):
        """Visualize current state"""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        pos, (has_key, direction), (door_state,) = self.current_state
        
        # Place agent
        grid[pos[1]][pos[0]] = '^>v<'[direction]
        
        # Place door
        door_x, door_y = self.door_pos
        grid[door_y][door_x] = 'D' if door_state > 0 else 'O'
        
        # Place key if not collected
        if not has_key:
            key_x, key_y = self.key_pos
            grid[key_y][key_x] = 'K'
        
        print("\nGrid:")
        for row in grid:
            print(' '.join(row))
        print(f"Has key: {has_key}")
        print(f"Door state: {door_state}")

    def step(self, action):
        """Take a step in the environment"""
        next_states = self.get_next_states(self.current_state, action)
        next_state = max(next_states, key=lambda x: x[1])[0]
        reward = self.get_reward(self.current_state, action, next_state)
        self.current_state = next_state
        done = next_state in self.goal_states
        return next_state, reward, done

    def get_actions(self):
        return self.actions

    def get_init_state(self):
        return self.init_state

    def get_state_hash(self, state):
        return str(state)

    def get_goal_states(self):
        return self.goal_states

    def check_goal_reached(self, pos):
        return pos == self.door_pos







def visualize_path(env, path):
    print("\nPath visualization:")
    for i, state in enumerate(path):
        pos, (has_key, direction), (door_state,) = state
        
        # Debug info
        print(f"\nStep {i} Debug Info:")
        print(f"Position: {pos}")
        print(f"Grid size: {env.grid_size}")
        print(f"Direction value: {direction}")
        
        # Validate position
        if not (0 <= pos[0] < env.grid_size and 0 <= pos[1] < env.grid_size):
            print(f"ERROR: Position {pos} is out of bounds for grid size {env.grid_size}")
            continue
            
        # Validate direction
        if not (0 <= direction < 4):
            print(f"ERROR: Invalid direction value: {direction}")
            continue
            
        # Create grid
        grid = [['.' for _ in range(env.grid_size)] for _ in range(env.grid_size)]
        
        # Place agent with direction
        direction_symbols = '^>v<'
        try:
            grid[pos[1]][pos[0]] = direction_symbols[direction]
        except IndexError as e:
            print(f"ERROR placing agent: {e}")
            print(f"Position: {pos}")
            print(f"Direction: {direction}")
            continue
        
        # Place door
        door_x, door_y = env.door_pos
        try:
            grid[door_y][door_x] = 'D' if door_state > 0 else 'O'
        except IndexError as e:
            print(f"ERROR placing door: {e}")
            continue
        
        # Place key if not collected
        if not has_key:
            key_x, key_y = env.key_pos
            try:
                grid[key_y][key_x] = 'K'
            except IndexError as e:
                print(f"ERROR placing key: {e}")
                continue
        
        print(f"\nStep {i}:")
        print(f"Has key: {'Yes' if has_key else 'No'}")
        print(f"Facing: {direction_symbols[direction]}")
        print(f"Door state: {'Locked' if door_state == 2 else 'Closed' if door_state == 1 else 'Open'}")
        
        # Print grid
        try:
            for row in grid:
                print(' '.join(row))
        except Exception as e:
            print(f"ERROR printing grid: {e}")




def test_solution_path(env):
    print("\nTesting solution path...")
    

    V = robust_vectorized_value_iteration(env)
    policy = get_robust_policy(env, V)
    
    # Debug initial values
    init_state = env.get_init_state()
    init_hash = env.get_state_hash(init_state)
    print(f"\nInitial state: {init_state}")
    print(f"Initial state value: {V[init_hash]}")
    print(f"Initial action: {policy[init_hash]}")
    
    state = init_state
    steps = 0
    max_steps = 100
    path = [state]
    total_reward = 0
    
    while steps < max_steps:
        print(f"\nStep {steps}:")
        print(f"Current state: {state}")
        
        # Debug key-related info
        pos, (has_key, direction), (door_state,) = state
        print(f"At key position: {pos == env.key_pos}")
        print(f"Has key: {has_key}")
        
        if state in env.get_goal_states():
            print("✓ Goal reached!")
            break
        
        state_hash = env.get_state_hash(state)
        action = policy[state_hash]
        print(f"Chosen action: {action}")
        
        # Find next state with debugging
        next_states = []
        for possible_next in env.get_state_space():
            prob = env.get_transition_probability(state, action, possible_next)
            if prob > 0:
                next_states.append((possible_next, prob))
                if action == "pickup" and prob > 0:
                    print(f"Possible next state: {possible_next}")
                    print(f"Probability: {prob}")
        
        if not next_states:
            print("✗ No valid next states found!")
            break
            
        next_state = max(next_states, key=lambda x: x[1])[0]
        reward = env.get_reward(state, action, next_state)
        
        print(f"Next state: {next_state}")
        print(f"Reward: {reward}")
        
        state = next_state
        path.append(state)
        steps += 1
        total_reward += reward
    
    success = state in env.get_goal_states()
    print(f"\nFinal result: {'Success' if success else 'Failure'}")
    print(f"Steps taken: {steps}")
    print(f"Total reward: {total_reward}")
    
    return success, path

def run_test():
    env = UnlockEnv(grid_size=5, slip_prob=0.1)
    success, path = test_solution_path(env)
    if success:
        visualize_path(env, path)



if __name__ == "__main__":
    run_test()
