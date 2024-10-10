import numpy as np
from typing import List, Tuple, Dict
from MDP import MDP

class UnlockEnv(MDP):
    def __init__(self, grid_size: int = 5, max_steps: int = 100, slip_prob: float = 0.1, 
                 init_state: Tuple = None, goal_states: List[Tuple] = None):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.actions = ["left", "right", "forward", "toggle"]
        self.door_states = [0, 1, 2]  # 0=open, 1=closed, 2=locked
        self.step_count = 0
        self.slip_prob = slip_prob

        # Initialize the environment
        self.door_pos = (grid_size - 1, grid_size - 1)
        
        self.state_space = self.get_state_space()
        
        # Set initial state
        if init_state is None:
            self.init_state = ((0, 0), 0, 2)  # Default: top-left, facing North, door locked
        else:
            self.init_state = init_state
        self.current_state = self.init_state
        
        # Set goal states
        if goal_states is None:
            self.goal_states = [((x, y), direction, 0) for x in range(self.grid_size) 
                                for y in range(self.grid_size) for direction in range(4)]
        else:
            self.goal_states = goal_states
        
        self.discount = 0.99
        self.reward_func = self.get_reward

    def get_state_space(self) -> List[Tuple]:
        state_space = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for direction in range(4):
                    for door_state in self.door_states:
                        state_space.append(((x, y), direction, door_state))
        return state_space

    def get_actions(self) -> List[str]:
        return self.actions

    def get_transition_probability(self, state: Tuple, action: str, next_state: Tuple) -> float:
        transitions = self.return_transition_probabilities(state, action)
        return transitions.get(next_state, 0.0)

    def return_transition_probabilities(self, state: Tuple, action: str) -> Dict[Tuple, float]:
        transitions = {}
        (x, y), direction, door_state = state

        if action == "left" or action == "right":
            intended_direction = (direction - 1) % 4 if action == "left" else (direction + 1) % 4
            transitions[((x, y), intended_direction, door_state)] = 1 - self.slip_prob
            transitions[((x, y), direction, door_state)] = self.slip_prob / 3
            transitions[((x, y), (direction + 1) % 4, door_state)] = self.slip_prob / 3
            transitions[((x, y), (direction - 1) % 4, door_state)] = self.slip_prob / 3

        elif action == "forward":
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][direction]
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                transitions[((new_x, new_y), direction, door_state)] = 1 - self.slip_prob
            else:
                transitions[((x, y), direction, door_state)] = 1 - self.slip_prob

            # Slip probabilities
            transitions[((x, y), (direction + 1) % 4, door_state)] = self.slip_prob / 3
            transitions[((x, y), (direction - 1) % 4, door_state)] = self.slip_prob / 3
            transitions[((x, y), direction, door_state)] = transitions.get(((x, y), direction, door_state), 0) + self.slip_prob / 3

        elif action == "toggle":
            if (x, y) == self.door_pos:
                if door_state == 2:  # locked
                    new_door_state = 1  # closed
                elif door_state == 1:  # closed
                    new_door_state = 0  # open
                else:
                    new_door_state = door_state  # already open
                transitions[((x, y), direction, new_door_state)] = 1 - self.slip_prob
                transitions[((x, y), direction, door_state)] = self.slip_prob
            else:
                transitions[((x, y), direction, door_state)] = 1.0

        # Ensure probabilities sum to 1
        total_prob = sum(transitions.values())
        if abs(total_prob - 1.0) > 1e-10:  # Allow for small floating-point errors
            for state in transitions:
                transitions[state] /= total_prob

        return transitions

    def get_reward(self, state: Tuple, action: str, next_state: Tuple) -> float:
        if next_state in self.goal_states and state not in self.goal_states:
            return 1 - 0.9 * (self.step_count / self.max_steps)
        return 0

    def get_init_state(self) -> Tuple:
        return self.init_state

    def get_state_hash(self, state: Tuple) -> str:
        return str(state)

    def get_goal_states(self) -> List[Tuple]:
        return self.goal_states

    def is_terminal(self, state: Tuple) -> bool:
        return state in self.goal_states or self.step_count >= self.max_steps

    def reset(self) -> Tuple:
        self.current_state = self.init_state
        self.step_count = 0
        return self.current_state

    def step(self, action: str) -> Tuple[Tuple, float, bool]:
        self.step_count += 1
        transitions = self.return_transition_probabilities(self.current_state, action)
        next_state = max(transitions, key=transitions.get)
        reward = self.get_reward(self.current_state, action, next_state)
        done = self.is_terminal(next_state)
        
        self.current_state = next_state
        return next_state, reward, done

    def render(self) -> None:
        (x, y), direction, door_state = self.current_state
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[y][x] = '^>v<'[direction]
        door_x, door_y = self.door_pos
        grid[door_y][door_x] = 'D' if door_state > 0 else 'O'
        
        for row in grid:
            print(' '.join(row))
        print(f"Door state: {'Locked' if door_state == 2 else 'Closed' if door_state == 1 else 'Open'}")
        print(f"Step: {self.step_count}/{self.max_steps}")




class UnlockPickupEnv(MDP):
    def __init__(self, grid_size: int = 5, max_steps: int = 100, slip_prob: float = 0.1):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.actions = ["left", "right", "forward", "toggle", "pickup"]
        self.door_states = [0, 1, 2]  # 0=open, 1=closed, 2=locked
        self.step_count = 0
        self.slip_prob = slip_prob

        # Initialize the environment
        self.agent_pos = (0, 0)
        self.agent_dir = 0  # 0: North, 1: East, 2: South, 3: West
        self.door_pos = (grid_size // 2, grid_size - 1)
        self.ball_pos = (self.door_pos[0] - 1, self.door_pos[1])
        self.key_pos = (0, grid_size - 1)
        self.box_pos = (grid_size - 1, grid_size - 1)
        
        self.door_state = 2  # Start with locked door
        self.has_key = False
        self.has_box = False
        
        self.state_space = self.get_state_space()
        self.init_state = self.get_init_state()
        self.discount = 0.99
        self.reward_func = self.get_reward

    def get_state_space(self) -> List[Tuple]:
        state_space = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for direction in range(4):
                    for door_state in self.door_states:
                        for has_key in [False, True]:
                            for has_box in [False, True]:
                                for ball_x in range(self.grid_size):
                                    for ball_y in range(self.grid_size):
                                        state_space.append(((x, y), direction, door_state, has_key, has_box, (ball_x, ball_y)))
        return state_space

    def get_actions(self) -> List[str]:
        return self.actions

    def get_transition_probability(self, state: Tuple, action: str, next_state: Tuple) -> float:
        transitions = self.return_transition_probabilities(state, action)
        return transitions.get(next_state, 0.0)

    def return_transition_probabilities(self, state: Tuple, action: str) -> Dict[Tuple, float]:
        transitions = {}
        (x, y), direction, door_state, has_key, has_box, ball_pos = state

        if action in ["left", "right"]:
            intended_direction = (direction - 1) % 4 if action == "left" else (direction + 1) % 4
            transitions[((x, y), intended_direction, door_state, has_key, has_box, ball_pos)] = 1 - self.slip_prob
            for slip_dir in range(4):
                if slip_dir != intended_direction:
                    transitions[((x, y), slip_dir, door_state, has_key, has_box, ball_pos)] = self.slip_prob / 3

        elif action == "forward":
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][direction]
            new_x, new_y = x + dx, y + dy
            if self.is_valid_position(new_x, new_y) and (new_x, new_y) != self.ball_pos:
                transitions[((new_x, new_y), direction, door_state, has_key, has_box, ball_pos)] = 1 - self.slip_prob
            else:
                transitions[((x, y), direction, door_state, has_key, has_box, ball_pos)] = 1 - self.slip_prob

            # Slip probabilities
            for slip_dir in range(4):
                if slip_dir != direction:
                    transitions[((x, y), slip_dir, door_state, has_key, has_box, ball_pos)] = self.slip_prob / 3

        elif action == "toggle":
            if (x, y) == self.door_pos and has_key:
                new_door_state = max(0, door_state - 1)
                transitions[((x, y), direction, new_door_state, has_key, has_box, ball_pos)] = 1
            else:
                transitions[((x, y), direction, door_state, has_key, has_box, ball_pos)] = 1

        elif action == "pickup":
            if (x, y) == self.key_pos and not has_key:
                transitions[((x, y), direction, door_state, True, has_box, ball_pos)] = 1
            elif (x, y) == self.box_pos and not has_box and door_state == 0:
                transitions[((x, y), direction, door_state, has_key, True, ball_pos)] = 1
            elif (x, y) == self.ball_pos:
                dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][direction]
                new_ball_x, new_ball_y = ball_pos[0] + dx, ball_pos[1] + dy
                if self.is_valid_position(new_ball_x, new_ball_y) and (new_ball_x, new_ball_y) != (x, y):
                    transitions[((x, y), direction, door_state, has_key, has_box, (new_ball_x, new_ball_y))] = 1
                else:
                    transitions[((x, y), direction, door_state, has_key, has_box, ball_pos)] = 1
            else:
                transitions[((x, y), direction, door_state, has_key, has_box, ball_pos)] = 1

        return transitions

    def is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def get_reward(self, state: Tuple, action: str, next_state: Tuple) -> float:
        _, _, _, _, has_box, _ = next_state
        if has_box:
            return 1 - 0.9 * (self.step_count / self.max_steps)
        return 0

    def get_init_state(self) -> Tuple:
        return ((0, 0), 0, 2, False, False, self.ball_pos)

    def get_state_hash(self, state: Tuple) -> str:
        return str(state)

    def get_goal_states(self) -> List[Tuple]:
        return [((x, y), direction, door_state, has_key, True, ball_pos)
                for x in range(self.grid_size)
                for y in range(self.grid_size)
                for direction in range(4)
                for door_state in self.door_states
                for has_key in [False, True]
                for ball_pos in [(bx, by) for bx in range(self.grid_size) for by in range(self.grid_size)]]

    def is_terminal(self, state: Tuple) -> bool:
        _, _, _, _, has_box, _ = state
        return has_box or self.step_count >= self.max_steps

    def reset(self) -> Tuple:
        self.agent_pos = (0, 0)
        self.agent_dir = 0
        self.door_state = 2
        self.has_key = False
        self.has_box = False
        self.ball_pos = (self.door_pos[0] - 1, self.door_pos[1])
        self.step_count = 0
        return self.get_init_state()

    def step(self, action: str) -> Tuple[Tuple, float, bool]:
        self.step_count += 1
        current_state = self.get_init_state()
        transitions = self.return_transition_probabilities(current_state, action)
        next_state = max(transitions, key=transitions.get)
        reward = self.get_reward(current_state, action, next_state)
        done = self.is_terminal(next_state)
        
        self.agent_pos, self.agent_dir, self.door_state, self.has_key, self.has_box, self.ball_pos = next_state
        return next_state, reward, done

    def render(self) -> None:
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.agent_pos[1]][self.agent_pos[0]] = '^>v<'[self.agent_dir]
        grid[self.door_pos[1]][self.door_pos[0]] = 'D' if self.door_state > 0 else 'O'
        grid[self.key_pos[1]][self.key_pos[0]] = 'K' if not self.has_key else '.'
        grid[self.box_pos[1]][self.box_pos[0]] = 'B' if not self.has_box else '.'
        grid[self.ball_pos[1]][self.ball_pos[0]] = 'o'
        
        for row in grid:
            print(' '.join(row))
        print(f"Door state: {'Locked' if self.door_state == 2 else 'Closed' if self.door_state == 1 else 'Open'}")
        print(f"Has key: {'Yes' if self.has_key else 'No'}")
        print(f"Has box: {'Yes' if self.has_box else 'No'}")
        print(f"Step: {self.step_count}/{self.max_steps}")










