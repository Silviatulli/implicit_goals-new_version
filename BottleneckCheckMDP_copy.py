from Utils import powerset, ValueIteration, vectorized_value_iteration, get_policy
from DeterminizedMDP import DeterminizedMDP, identify_bottlenecks
import random

class BottleneckMDP(object):
    def __init__(self, mdp, bottlenecks):
        self.original_mdp = mdp
        self.bottlenecks = bottlenecks
        self.create_state_space()
        self.actions = mdp.get_actions()
        self.init_state = (mdp.get_init_state(),set())
        self.discount = mdp.discount

    def create_state_space(self):
        possible_bottleneck_sets = powerset(self.bottlenecks)
        self.state_space = []
        for I in possible_bottleneck_sets:
            #if len(I)>0:
            #    print(list(I[0]))
            #    print(type(I))
            #I_set = set(I)
            for J in self.original_mdp.get_state_space():
                #print(tuple(J))
                #print(type(J))
                #exit(0)
                self.state_space.append((J, set(I)))
        #exit(0)

    def get_state_space(self):
        return self.state_space

    def get_actions(self):
        return self.actions

    def get_transition_probability(self, state_tuple, action, next_state_tuple):
        state, subgoals = state_tuple
        next_state, next_subgoals = next_state_tuple

        current_transition_prob = self.original_mdp.get_transition_probability(state, action, next_state)
        if current_transition_prob == 0:
            return 0
        # You can't add more than one subgoal at a time
        if len(next_subgoals - subgoals) > 1:
            return 0
        # You can't ever remove a subgoal
        if len(subgoals - next_subgoals) > 0:
            return 0

        added_subgoal = list(next_subgoals - subgoals)
        if len(added_subgoal) > 0 and added_subgoal[0] != tuple(next_state):
            return 0

        if tuple(next_state) in self.bottlenecks and tuple(next_state) not in subgoals:
            return 0

        # # Add absorbing states for goal states
        # if self.original_mdp.check_goal_reached(state[0]):
        #     if state == next_state:
        #         return 1
        #     else:
        #         return 0

        return current_transition_prob

    def get_init_state(self):
        return self.init_state

    def get_state_hash(self, state):
        return str(state)

    def get_goal_states(self):
        return self.original_mdp.get_goal_states()

    def get_reward(self, state_tuple, action, next_state_tuple):
        next_state, next_subgoals = next_state_tuple
        if not self.original_mdp.check_goal_reached(next_state[0]):
            return 0

        if len(next_subgoals) == len(self.bottlenecks):
            return 1000
        #print("Missed bottleneck", state_tuple, next_state_tuple)
        return -1000


def check_bottleneck_achievability(robot_mdp, robot_bottlenecks, human_bottlenecks):
    achievable_bottlenecks = []

    for human_bottleneck in human_bottlenecks:
        print(f"\nChecking achievability of human bottleneck: {human_bottleneck}")

        # Check if the human bottleneck is an obstacle in the robot's model
        if robot_mdp.map[human_bottleneck[0]] == -1:
            print(f"Bottleneck {human_bottleneck} is an obstacle in the robot's model")
            continue

        def bottleneck_reward(state, action, next_state):
            if next_state[0] == human_bottleneck[0]:
                return 1000
            elif robot_mdp.check_goal_reached(next_state[0]):
                return 500  # Smaller reward for reaching the goal
            return 0

        robot_mdp.reward_func = bottleneck_reward

        V = vectorized_value_iteration(robot_mdp)
        initial_state = robot_mdp.get_init_state()
        initial_state_hash = robot_mdp.get_state_hash(initial_state)

        print(f"Initial state: {initial_state}")
        print(f"Initial state hash: {initial_state_hash}")
        print(f"Value of initial state: {V[initial_state_hash]}")

        # Check if there's a path with significant positive value
        if V[initial_state_hash] > 10:  # Adjust this threshold as needed
            achievable_bottlenecks.append(human_bottleneck)
            print(f"Bottleneck {human_bottleneck} is achievable")
        else:
            print(f"Bottleneck {human_bottleneck} is NOT achievable")

        # Print some path information
        path = extract_path(robot_mdp, V, initial_state, human_bottleneck[0])
        print(f"Path to bottleneck: {path}")
        print(f"Path length: {len(path)}")

    return achievable_bottlenecks

def extract_path(mdp, V, start_state, target_state):
    current_state = start_state
    path = [current_state]
    while current_state[0] != target_state:  # Compare only the position part of the state
        best_action = None
        best_value = float('-inf')
        best_next_state = None
        for action in mdp.get_actions():
            for next_state in mdp.get_state_space():
                prob = mdp.get_transition_probability(current_state, action, next_state)
                if prob > 0:
                    value = V[mdp.get_state_hash(next_state)]
                    if value > best_value:
                        best_value = value
                        best_action = action
                        best_next_state = next_state
        if best_next_state is None:
            print(f"No valid next state found from {current_state}")
            break
        current_state = best_next_state
        path.append(current_state)
        if len(path) > 100:  # Prevent infinite loops
            print("Path extraction stopped due to length limit")
            break
    return path

class HumanBottleneckMDP(object):
    def __init__(self, robot_mdp, human_bottlenecks):
        self.original_mdp = robot_mdp
        self.human_bottlenecks = human_bottlenecks
        self.create_state_space()
        self.actions = robot_mdp.get_actions()
        self.init_state = (robot_mdp.get_init_state(), frozenset())
        self.discount = robot_mdp.discount

    def create_state_space(self):
        possible_bottleneck_sets = powerset(self.human_bottlenecks)
        self.state_space = []
        for I in possible_bottleneck_sets:
            for J in self.original_mdp.get_state_space():
                self.state_space.append((J, frozenset(I)))

    def get_state_space(self):
        return self.state_space

    def get_actions(self):
        return self.actions

    def get_transition_probability(self, state_tuple, action, next_state_tuple):
        state, visited_bottlenecks = state_tuple
        next_state, next_visited_bottlenecks = next_state_tuple

        current_transition_prob = self.original_mdp.get_transition_probability(state, action, next_state)
        if current_transition_prob == 0:
            return 0

        # Can only add bottlenecks, never remove
        if not visited_bottlenecks.issubset(next_visited_bottlenecks):
            return 0

        # Can only add the current bottleneck if it's being visited
        if len(next_visited_bottlenecks) - len(visited_bottlenecks) > 1:
            return 0

        if len(next_visited_bottlenecks) > len(visited_bottlenecks):
            added_bottleneck = list(next_visited_bottlenecks - visited_bottlenecks)[0]
            if added_bottleneck != tuple(next_state[0]):
                return 0

        return current_transition_prob

    def get_init_state(self):
        return self.init_state

    def get_state_hash(self, state):
        return str(state)

    def get_goal_states(self):
        return self.original_mdp.get_goal_states()

    def get_reward(self, state_tuple, action, next_state_tuple):
        _, next_visited_bottlenecks = next_state_tuple
        if self.original_mdp.check_goal_reached(next_state_tuple[0][0]):
            return len(next_visited_bottlenecks) * 100  # Reward based on number of bottlenecks visited
        return 0  # No reward for non-goal states
    
    
if __name__ == "__main__":
    from GridWorldClass import generate_and_visualize_gridworld, visualize_grids_with_bottlenecks
    from DeterminizedMDP import identify_bottlenecks
    
    # Generate Robot Model
    M_R = generate_and_visualize_gridworld(size=5, start=(0,0), goal=(4,4), obstacles_percent=0.1, divide_rooms=True, model_type="Robot Model", obstacle_seed=42)
    bottleneck_states_robot = identify_bottlenecks(M_R)
    # Filter out goal state from bottlenecks
    bottleneck_states_robot = [b for b in bottleneck_states_robot if b[0] != M_R.goal_pos]

    # Generate multiple Human Models
    num_human_models = 3
    human_models = []
    human_bottlenecks_list = []
    achievable_bottlenecks_list = []

    for i in range(num_human_models):
        M_H = generate_and_visualize_gridworld(size=5, start=(0,0), goal=(4,4), obstacles_percent=0.1, divide_rooms=True, model_type=f"Human Model {i+1}", obstacle_seed=random.randint(1, 10000))
        bottleneck_states_human = identify_bottlenecks(M_H)
        # Filter out goal state from bottlenecks
        bottleneck_states_human = [b for b in bottleneck_states_human if b[0] != M_H.goal_pos]
        achievable_human_bottlenecks = check_bottleneck_achievability(M_R, bottleneck_states_robot, bottleneck_states_human)
        
        human_models.append(M_H)
        human_bottlenecks_list.append(bottleneck_states_human)
        achievable_bottlenecks_list.append(achievable_human_bottlenecks)

    # Visualize the comparison
    visualize_grids_with_bottlenecks(M_R, human_models, bottleneck_states_robot, human_bottlenecks_list, achievable_bottlenecks_list)

    print("\nRobot's bottleneck states:", bottleneck_states_robot)
    for i, (bottlenecks, achievable) in enumerate(zip(human_bottlenecks_list, achievable_bottlenecks_list)):
        print(f"\nHuman Model {i+1}:")
        print("Bottleneck states:", bottlenecks)
        print("Achievable bottlenecks:", achievable)
        print("Unachievable bottlenecks:", [b for b in bottlenecks if b not in achievable])

    
    M = BottleneckMDP(M_R, achievable)
    V = vectorized_value_iteration(M)
    #for s in M.state_space:
    #    print(s, V[M.get_state_hash(s)])
    #exit(0)
    policy = get_policy(M, V)

    M.reward_func = None
    # Determinized for the policy
    det_mdp_for_policy = DeterminizedMDP(M, policy)
    det_mdp_for_policy.bottleneck_MDP = M

    det_mdp_for_policy.reward_func = det_mdp_for_policy.reward_function_for_avoiding_all_bottleneck

    V = vectorized_value_iteration(det_mdp_for_policy)
    initial_state_hash = det_mdp_for_policy.get_state_hash(det_mdp_for_policy.get_init_state())
    print(policy)
