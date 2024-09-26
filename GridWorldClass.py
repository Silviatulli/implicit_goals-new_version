import numpy as np
from Search import BFSearch
from collections import combinations
from Utils import powerset
class GridWorld(object):
    def __init__(self, size=5, start=None, goal=None,  obstacles_percent=0.1,
                 divide_rooms=False, room_count=4, agent_features=None, locatables=None,
                 locatable_locations=None, slip_prob=0.1):
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
        #self.stay_prob = stay_prob

        # Prepare map
        # Map is a 2D numpy array with a matrix style indexing
        # 0 is empty, -1 is obstacle
        # rooted at top left corner - i.e. (0,0) is top left corner
        self.map = np.zeros((size, size))
        self.state_space = None

    def place_random_obstacles(self):
        self.state_space = None
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


    def return_all_neighbors(self, node):
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
        path = BFSearch(self.start_pos, self.check_goal_reached, self.return_all_neighbors)
        if path is None:
            print("No path found.")
            return False
        return True


    def return_actions(self):
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
    def return_state_space(self):
        if self.state_space is None:
            self.create_state_space()
        return self.state_space

    def return_transition_probabilities_for_move(self, state, action, state_prime):
        # If current state is an obstacle, then the agent stays in the same state
        if self.map[state[0]] == -1:
            if state == state_prime:
                return 1
            else:
                return 0
        # If the next state is an obstacle, then the agent can't move there
        if self.map[state_prime[0]] == -1:
            return 0

        x, y = state[0]
        x_prime, y_prime = state_prime[0]
        # It can't move more than one step
        if x_prime != x-1 or x_prime != x or x_prime != x+1:
            return 0
        if y_prime != y-1 or y_prime != y or y_prime != y+1:
            return 0
        if action == "up":
            if x_prime == x-1 and y_prime == y:
                return (1-self.slip_prob)
            else:
                return self.slip_prob
        elif action == "down":
            if x_prime == x+1 and y_prime == y:
                return (1-self.slip_prob)
            else:
                return self.slip_prob
        elif action == "left":
            if x_prime == x and y_prime == y-1:
                return (1-self.slip_prob)
            else:
                return self.slip_prob
        elif action == "right":
            if x_prime == x and y_prime == y+1:
                return (1-self.slip_prob)
            else:
                return self.slip_prob
        assert False, "Should never reach here."

    def return_transition_probabilities(self, state, action, state_prime):
        return self.return_transition_probabilities_for_move(state, action, state_prime)



    def visualize(self):
        print(self.map)




if __name__ == "__main__":
    grid = GridWorld(start=(0,0), goal=(4, 4))
    grid.place_random_obstacles()
    grid.visualize()
    grid.divide_into_rooms()
    grid.visualize()
    grid.place_start_and_goal()
    path_found = grid.check_for_path()
    print("Path found: ", path_found)






