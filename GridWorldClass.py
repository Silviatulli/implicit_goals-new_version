from MDP import MDP
import numpy as np
from Search import BFSearch
from Utils import powerset, ValueIteration
class GridWorld(MDP):
    def __init__(self, size=5, start=None, goal=None,  obstacles_percent=0.1,
                 divide_rooms=False, room_count=4, agent_features=[], locatables=[],
                 locatable_locations=[], slip_prob=0.1, discount=0.99, max_tries=100, obstacle_seed=1,
                 starting_features=[]):
        self.size = size
        self.start_pos = start
        self.goal_pos = goal
        self.obstacles_percent = obstacles_percent
        self.divide_rooms = divide_rooms
        self.room_count = room_count
        # All agent features are assumed to be binary
        self.agent_features = agent_features
        # All locatables are either in the position or being held by the agent
        # 0 - in position, 1 - being held
        self.locatables = locatables
        self.locatable_locations = locatable_locations
        self.slip_prob = slip_prob
        self.reward_func = self.goal_reward_func
        #self.stay_prob = stay_prob

        # Prepare map
        # Map is a 2D numpy array with a matrix style indexing
        # 0 is empty, -1 is obstacle
        # rooted at top left corner - i.e. (0,0) is top left corner
        self.map = np.zeros((size, size))
        self.state_space = None
        self.discount = discount
        self.obstacle_seed = obstacle_seed

        valid_config_found = False
        max_tries = 100
        curr_tries = 0
        while not valid_config_found and curr_tries < max_tries:
            self.place_random_obstacles()
            if self.divide_rooms:
                self.divide_into_rooms()
            self.place_start_and_goal()
            path_found = self.check_for_path()
            if path_found:
                valid_config_found = True
            else:
                curr_tries += 1

        if not valid_config_found:
            print("Could not find a valid configuration after ", max_tries, " tries.")
            print("Creating a default empty grid world.")
            self.map = np.zeros((size, size))
            self.place_start_and_goal()

        self.create_state_space()
        self.start_features = starting_features
        assert slip_prob >= 0 and slip_prob*3 <= 1, "Slip probability should be between 0 and shouldn't add upto more than one."

    def place_random_obstacles(self):
        self.state_space = None
        np.random.seed(self.obstacle_seed)
        total_obstacles = int(self.size * self.size * self.obstacles_percent)
        for i in range(total_obstacles):
            x = np.random.randint(self.size)
            y = np.random.randint(self.size)
            self.map[x, y] = -1

    def divide_into_rooms(self):
        self.state_space = None
        assert self.room_count == 4, "Currently only supports 4 rooms."

        room_divider = self.size // 2

        print("Room divider: ", room_divider)
        self.map[room_divider, :] = -1
        self.map[:, room_divider] = -1
        # Place doors
        x1 = np.random.randint(room_divider)
        self.map[x1, room_divider] = 0
        x2 = np.random.randint(room_divider+1, self.size)
        self.map[x2, room_divider] = 0
        y1 = np.random.randint(room_divider)
        self.map[room_divider, y1] = 0
        y2 = np.random.randint(room_divider+1, self.size)
        self.map[room_divider, y2] = 0

    def place_start_and_goal(self):
        if self.start_pos is None:
            self.start_pos = (np.random.randint(self.size), np.random.randint(self.size))
        if self.goal_pos is None:
            self.goal_pos = (np.random.randint(self.size), np.random.randint(self.size))


    def get_all_neighbors(self, node):
        x, y = node[0]
        neighbors = []
        if x > 0:
            if self.map[x-1, y] != -1:
                neighbors.append(((x-1, y), "up"))
        if x < self.size-1:
            if self.map[x+1, y] != -1:
                neighbors.append(((x+1, y), "down"))
        if y > 0:
            if self.map[x, y-1] != -1:
                neighbors.append(((x, y-1), "left"))
        if y < self.size-1:
            if self.map[x, y+1] != -1:
                neighbors.append(((x, y+1), "right"))
        return neighbors

    def check_goal_reached(self, node):
        x, y = node[0]
        return x == self.goal_pos[0] and y == self.goal_pos[1]

    def check_for_path(self):
        if self.start_pos is None or self.goal_pos is None:
           print("Start or goal not set.")
           return False
        path = BFSearch(self.start_pos, self.check_goal_reached, self.get_all_neighbors)
        if path is None:
            print("No path found.")
            return False
        return True


    def get_actions(self):
        return ["up", "down", "left", "right"]

    def create_state_space(self):
        if self.state_space is not None:
            return None
        self.state_space = []
        for i in range(self.size):
            for j in range(self.size):
                current_state = [(i,j)]
                for agent_feature_set in powerset(self.agent_features):
                    for locatable_set in powerset(self.locatables):
                                current_state.append(agent_feature_set)
                                current_state.append(locatable_set)
                self.state_space.append(current_state)
    def get_state_space(self):
        if self.state_space is None:
            self.create_state_space()
        return self.state_space

    def get_transition_probability_for_move(self, state, action, state_prime):
        # If current state is an obstacle, then the agent stays in the same state
        if self.map[state[0]] == -1:
            if state == state_prime:
                return 1
            else:
                return 0
        # If the next state is an obstacle, then the agent can't move there
        if self.map[state_prime[0]] == -1:
            return 0

        # Make goal state absorbing state
        if self.check_goal_reached(state):
            if state == state_prime:
                return 1
            else:
                return 0

        x, y = state[0]
        x_prime, y_prime = state_prime[0]
        # It can't move more than one step in one of the cardinal directions or is at the same cell
        if (x_prime, y_prime) not in [(x-1, y), (x+1, y), (x, y-1), (x, y+1), (x, y)]:
            return 0

        # count free spaces
        up_free = False
        down_free = False
        left_free = False
        right_free = False
        if x-1 >= 0 and self.map[x-1, y] != -1:
            up_free = True
        if x+1 < self.size and self.map[x+1, y] != -1:
            down_free = True
        if  y-1 >= 0 and self.map[x, y-1] != -1:
            left_free = True
        if y+1 < self.size and self.map[x, y+1] != -1:
            right_free = True
        if action == "up":
            total_prob = 1
            if down_free:
                total_prob += 1
            if left_free:
                total_prob += 1
            if right_free:
                total_prob += 1
            if up_free:
                # Remaining probability is distributed among the other actions
                if x_prime == x-1 and y_prime == y:
                    return (1- total_prob * self.slip_prob)
                else:
                    return self.slip_prob
            else:
                # distribute the probability among the other actions
                # with the highest probability to staying at the same state
                if x_prime == x and y_prime == y:
                    return (1- ((total_prob-1) * self.slip_prob))
                else:
                    return self.slip_prob
        elif action == "down":
            total_prob = 1
            if up_free:
                total_prob += 1
            if left_free:
                total_prob += 1
            if right_free:
                total_prob += 1

            if down_free:
                if x_prime == x+1 and y_prime == y:
                    return (1- total_prob * self.slip_prob)
                else:
                    return self.slip_prob
            else:
                if x_prime == x and y_prime == y:
                    return (1- ((total_prob-1) * self.slip_prob))
                else:
                    return self.slip_prob
        elif action == "left":
            total_prob = 1
            if up_free:
                total_prob += 1
            if down_free:
                total_prob += 1
            if right_free:
                total_prob += 1
            if left_free:
                if x_prime == x and y_prime == y-1:
                    return (1- total_prob * self.slip_prob)
                else:
                    return self.slip_prob
            else:
                if x_prime == x and y_prime == y:
                    return (1- ((total_prob-1) * self.slip_prob))
                else:
                    return self.slip_prob
        elif action == "right":
            total_prob = 1
            if up_free:
                total_prob += 1
            if down_free:
                total_prob += 1
            if left_free:
                total_prob += 1
            if right_free:
                if x_prime == x and y_prime == y+1:
                    return (1- total_prob * self.slip_prob)
                else:
                    return self.slip_prob
            else:
                if x_prime == x and y_prime == y:
                    return (1- ((total_prob-1) * self.slip_prob))
                else:
                    return self.slip_prob
        assert False, "Should never reach here."

    def get_transition_probability(self, state, action, state_prime):
        return self.get_transition_probability_for_move(state, action, state_prime)

    def goal_reward_func(self, state, action, next_state):
        #print("next_state: ", next_state, type(next_state))
        # Only reward for reaching the goal state from a non-goal state
        if self.check_goal_reached(next_state) and not self.check_goal_reached(state):
            return 1
        return 0

    def get_state_hash(self, state):
        return str(state)

    def get_reward(self, state, action, next_state):
        return self.reward_func(state, action, next_state)

    def get_init_state(self):
        start_state = [self.start_pos, tuple(self.start_features), tuple(self.locatables)]
        return start_state

    def visualize(self):
        print(self.map)

    def get_goal_states(self):
        return [[self.goal_pos, tuple(self.start_features), tuple(self.locatables)]]






if __name__ == "__main__":
    grid = GridWorld(start=(0,0), goal=(1, 1), size=2, obstacles_percent=0, discount=0.99)
    grid.visualize()
    print(grid.get_init_state())
    #print(grid.get_transition_probability([(1,1),(),()], 'down', [(2,1),(),()]))
    V = ValueIteration(grid)
    print(V)






