from MDP import MDP
from GridWorldClass import GridWorld, visualize_grid
from Utils import ValueIteration, vectorized_value_iteration, get_policy
from BottleneckCheckMDP import BottleneckMDP
from collections import defaultdict
import numpy as np
from DeterminizedMDP import DeterminizedMDP
from collections import deque



def find_maximally_achievable_subsets(M_R, M_H_list):
    B = set()  # Set of all bottleneck states
    for M in M_H_list:
        bottlenecks = identify_bottlenecks(M)
        B.update(tuple(b) for b in bottlenecks)  # Convert each bottleneck to a tuple
    
    print(f"Total bottleneck states: {len(B)}")
    print("Bottleneck states:", B)

    I = set()  # Set of maximal achievable subgoal sets
    fringe = [frozenset(B)]  # Start with the set of all bottleneck states
    
    while fringe:
        I_prime = fringe.pop(0)
        print(f"\nChecking subset of size {len(I_prime)}: {I_prime}")
        if check_achievability(I_prime, M_R):
            print("Subset is achievable")
            # Check if I_prime is maximal
            if all(not check_achievability(I_prime | {s}, M_R) for s in B - I_prime):
                I.add(I_prime)
                print(f"Added maximal achievable subset: {I_prime}")
            else:
                print("Subset is not maximal")
        else:
            print("Subset is not achievable")
            for s in I_prime:
                new_subset = frozenset(I_prime - {s})
                if new_subset not in fringe and not any(new_subset.issubset(i) for i in I):
                    fringe.append(new_subset)
                    print(f"Added new subset to fringe: {new_subset}")
    
    return I, B

def identify_bottlenecks(M):
    det_mdp = DeterminizedMDP(M)
    init_state_hash = det_mdp.get_state_hash(det_mdp.get_init_state())
    det_mdp.reward_func = det_mdp.bottleneck_reward
    bottlenecks = []
    
    for state in det_mdp.get_state_space():
        if state != M.get_init_state() and state != M.get_goal_states()[0]:
            det_mdp.bottleneck_state = det_mdp.get_state_hash(state)
            V = vectorized_value_iteration(det_mdp)
            if V[init_state_hash] <= 0:
                bottlenecks.append(tuple(state))  # Convert state to tuple
                print(f"Identified bottleneck state: {tuple(state)}")
    
    return bottlenecks

def check_achievability(I_prime, M_R):
    M = BottleneckMDP(M_R, I_prime)
    V = vectorized_value_iteration(M)
    # for s in M.state_space:
    #    print(s, V[M.get_state_hash(s)])
    # exit(0)
    policy = get_policy(M, V)

    M.reward_func = None
    # Determinized for the policy
    det_mdp_for_policy = DeterminizedMDP(M, policy)
    det_mdp_for_policy.bottleneck_MDP = M

    det_mdp_for_policy.reward_func = det_mdp_for_policy.reward_function_for_avoiding_all_bottleneck

    V = vectorized_value_iteration(det_mdp_for_policy)
    initial_state_hash = det_mdp_for_policy.get_state_hash(det_mdp_for_policy.get_init_state())
    # print(policy)
    is_achievable = False
    if V[initial_state_hash] <= 0:
        is_achievable = True
        print("Achievable")
    return is_achievable


# if __name__ == "__main__":
#     # Generate a simple robot model
#     M_R = GridWorld(size=5, start=(0, 0), goal=(4, 4), obstacles_percent=0.1, divide_rooms=False, slip_prob=0.1)
#     print("Robot Model:")
#     visualize_grid(M_R)
#
#     # Generate a simple human model
#     M_H = GridWorld(size=5, start=(0, 0), goal=(4, 4), obstacles_percent=0.1, divide_rooms=True, slip_prob=0.1)
#     #M_H = generate_and_visualize_gridworld(size=5, start=(0, 0), goal=(4, 4), obstacles_percent=0.1, divide_rooms=True,
#                                            #model_type=f"Human Model 1")
#     print("\nHuman Model:")
#     visualize_grid(M_H)
#
#
#     # Find maximally achievable subsets
#     I, B = find_maximally_achievable_subsets(M_R, [M_H])
#
#     print("\nMaximally achievable subsets of bottleneck states:")
#     for subset in I:
#         print(subset)
#     print("\nAll bottleneck states:", B)


if __name__ == "__main__":
    from GridWorldClass import generate_and_visualize_gridworld

    # Generate robot model
    M_R = generate_and_visualize_gridworld(size=5, start=(0,0), goal=(4,4), obstacles_percent=0.1, divide_rooms=True, model_type="Robot Model")

    # Generate human models with different configurations
    M_H_list = []
    human_configs = [
        {"obstacles_percent": 0.1, "divide_rooms": True},
        {"obstacles_percent": 0.1, "divide_rooms": True},
        {"obstacles_percent": 0.1, "divide_rooms": False}
    ]


    for i, config in enumerate(human_configs):
        M_H = generate_and_visualize_gridworld(size=5, start=(0,0), goal=(4,4),
                                               obstacles_percent=config["obstacles_percent"],
                                               divide_rooms=config["divide_rooms"],
                                               model_type=f"Human Model {i+1}")
        if M_H:
            M_H_list.append(M_H)

    # Proceed with analysis only if we have valid models
    if M_R and M_H_list:
        print("\nRobot Model Initial State:", M_R.get_init_state())
        print("\nActions available in Human Model 1:", M_H_list[0].get_actions())

        I, B = find_maximally_achievable_subsets(M_R, M_H_list)
        print("\nMaximally achievable subsets of bottleneck states:")
        for subset in I:
            print(subset)
        print("\nAll bottleneck states:", B)
    else:
        print("Could not generate valid models for analysis.")


