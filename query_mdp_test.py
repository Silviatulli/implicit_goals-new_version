from PuddleWorldClass import PuddleWorld, generate_and_visualize_puddleworld
from GridWorldClass import GridWorld
from DeterminizedMDP import identify_bottlenecks
from BottleneckCheckMDP import BottleneckMDP, visualize_puddleworld_policy
from GridWorldClass import generate_and_visualize_gridworld, visualize_grids_with_bottlenecks
from QueryMDP import QueryMDP
from Utils import vectorized_value_iteration
import random

def check_bottleneck_achievability(robot_mdp, robot_bottlenecks, human_bottlenecks):
    achievable_bottlenecks = []
    for human_bottleneck in human_bottlenecks:
        if robot_mdp.map[human_bottleneck[0]] == -1:
            continue
        
        def bottleneck_reward(state, action, next_state):
            if next_state[0] == human_bottleneck[0]:
                return 1000
            elif robot_mdp.check_goal_reached(next_state[0]):
                return 500
            return 0
        
        robot_mdp.reward_func = bottleneck_reward
        V = vectorized_value_iteration(robot_mdp)
        initial_state = robot_mdp.get_init_state()
        initial_state_hash = robot_mdp.get_state_hash(initial_state)
        
        if V[initial_state_hash] > 10:
            achievable_bottlenecks.append(human_bottleneck)
    
    return achievable_bottlenecks

# generate robot model
print("generating robot model...")
M_R = generate_and_visualize_gridworld(size=5, start=(0,0), goal=(4,4), obstacles_percent=0.1, divide_rooms=True, model_type="Robot Model", obstacle_seed=42)
bottleneck_states_robot = identify_bottlenecks(M_R)
# remove goal state from bottlenecks
bottleneck_states_robot = [b for b in bottleneck_states_robot if b[0] != M_R.goal_pos]

# generate multiple human models
num_human_models = 3
human_models = []
human_bottlenecks_list = []
achievable_bottlenecks_list = []

print(f"generating {num_human_models} human models...")
for i in range(num_human_models):
    M_H = generate_and_visualize_gridworld(size=5, start=(0,0), goal=(4,4), obstacles_percent=0.1, divide_rooms=True, model_type=f"Human Model {i+1}", obstacle_seed=random.randint(1, 10000))
    bottleneck_states_human = identify_bottlenecks(M_H)
    # remove goal state from bottlenecks
    bottleneck_states_human = [b for b in bottleneck_states_human if b[0] != M_H.goal_pos]
    achievable_human_bottlenecks = check_bottleneck_achievability(M_R, bottleneck_states_robot, bottleneck_states_human)
        
    human_models.append(M_H)
    human_bottlenecks_list.append(bottleneck_states_human)
    achievable_bottlenecks_list.append(achievable_human_bottlenecks)
    non_achievable_bottlenecks = [b for b in human_bottlenecks_list if b not in achievable_bottlenecks_list]

print("achievable bottlenecks:", achievable_bottlenecks_list)
print("non-achievable bottlenecks:", non_achievable_bottlenecks)

visualize_grids_with_bottlenecks(M_R, human_models, bottleneck_states_robot, human_bottlenecks_list, achievable_bottlenecks_list)

# create and run querymdp
print("creating and running querymdp...")
query_mdp = QueryMDP(M_R, achievable_bottlenecks_list, non_achievable_bottlenecks)
policy, bottleneck_mdp, explanations = query_mdp.run()

# display results
print("\nfinal policy:")
for i, state in enumerate(bottleneck_mdp.get_state_space()):
    if i < 10:  # print first 10 state-action pairs
        print(f"state: {state}, action: {policy[bottleneck_mdp.get_state_hash(state)]}")
    else:
        break

print("\nexplanations for unreachable necessary bottlenecks:")
for explanation in explanations:
    print(explanation)

# compare robot's policy with each human model
for i, human_model in enumerate(human_models):
    print(f"\ncomparing robot's policy with human model {i+1}:")
    human_bottlenecks = identify_bottlenecks(human_model)
    matching_bottlenecks = [b for b in human_bottlenecks if b in achievable_bottlenecks_list]
    missing_bottlenecks = [b for b in human_bottlenecks if b not in non_achievable_bottlenecks]
    
    print(f"matching bottlenecks: {matching_bottlenecks}")
    print(f"missing bottlenecks: {missing_bottlenecks}")
    
    # you could add more detailed comparison here

# summarize overall performance
print("\noverall performance summary:")
print(f"total unique human bottlenecks: {len(human_bottlenecks_list)}")
print(f"achievable bottlenecks: {len(achievable_bottlenecks_list)}")
print(f"non-achievable bottlenecks: {len(non_achievable_bottlenecks)}")
print(f"necessary but unreachable bottlenecks: {len(query_mdp.necessary_bottlenecks)}")
