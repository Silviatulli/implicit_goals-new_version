from MDP import MDP
from GridWorldClass import GridWorld
from Utils import ValueIteration
from collections import defaultdict
import numpy as np
from DeterminizedMDP import DeterminizedMDP



def find_maximally_achievable_subsets(M_R, M_H_list):
    B = set()  # Set of all bottleneck states
    for M in M_H_list:
        bottlenecks = identify_bottlenecks(M)
        B.update(tuple(b) if isinstance(b, list) else b for b in bottlenecks)
    
    I = set()  # Set of potential implicit subgoal sets
    fringe = [frozenset(B)]  # Queue for BFS
    
    while fringe:
        I_prime = fringe.pop(0)
        if check_achievability(I_prime, M_R):
            I.add(I_prime)
        else:
            for s in I_prime:
                new_subset = frozenset(I_prime - {s})
                if new_subset not in fringe and new_subset not in I:
                    fringe.append(new_subset)
    
    return I, B

def identify_bottlenecks(M):
    det_mdp = DeterminizedMDP(M)
    init_state_hash = det_mdp.get_state_hash(det_mdp.get_init_state())
    det_mdp.reward_func = det_mdp.bottleneck_reward
    bottlenecks = []
    
    for state in det_mdp.get_state_space():
        det_mdp.bottleneck_state = det_mdp.get_state_hash(state)
        V = ValueIteration(det_mdp)
        if V[init_state_hash] <= 0:
            bottlenecks.append(state)
    
    return bottlenecks

def check_achievability(I_prime, M_R):
    det_mdp = DeterminizedMDP(M_R)
    init_state_hash = det_mdp.get_state_hash(det_mdp.get_init_state())
    
    def achievability_reward(state, action, next_state):
        if det_mdp.get_state_hash(next_state) in I_prime:
            return 1
        return 0
    
    det_mdp.reward_func = achievability_reward
    V = ValueIteration(det_mdp)
    
    return V[init_state_hash] > 0


# Usage example
if __name__ == "__main__":
    from GridWorldClass import generate_and_visualize_gridworld
    
    # Generate robot model
    M_R = generate_and_visualize_gridworld(size=5, start=(0,0), goal=(4,4), obstacles_percent=0.1, divide_rooms=True, model_type="Robot Model")

    # Generate human models with different configurations
    M_H_list = []
    human_configs = [
        {"obstacles_percent": 0.1, "divide_rooms": True},
        {"obstacles_percent": 0.2, "divide_rooms": True},
        {"obstacles_percent": 0.15, "divide_rooms": False}
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