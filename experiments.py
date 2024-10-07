import random
from GridWorldClass import *
from maximal_achievable_subsets import *
import unittest
from RockWorldClass import *

def generate_human_models(base_model, num_models=3):
    human_models = []
    
    for _ in range(num_models):
        new_model = GridWorld(
            size=base_model.size,
            start=base_model.start_pos,
            goal=base_model.goal_pos,
            obstacles_percent=random.uniform(0.1, 0.3),
            # divide_rooms=random.choice([True, False]),
            slip_prob=random.uniform(0.05, 0.3)
        )
        
        human_models.append(new_model)
    
    return human_models


if __name__ == "__main__":
    # Generate robot model
    M_R = GridWorld(size=6, start=(0,0), goal=(5,2), obstacles_percent=0.09, divide_rooms=True, slip_prob=0.1)
    
    print("Robot Model:")
    visualize_grid(M_R)

    print(f"Obstacle Percentage: {M_R.obstacles_percent:.2f}")
    print(f"Divided into rooms: {M_R.divide_rooms}")
    print(f"Slip Probability: {M_R.slip_prob:.2f}")


    # Generate human models
    M_H_list = generate_human_models(M_R, num_models=3)
    
    for i, M_H in enumerate(M_H_list):
        print(f"\nHuman Model {i+1}:")
        visualize_grid(M_H)

        print(f"Obstacle Percentage: {M_H.obstacles_percent:.2f}")
        print(f"Divided into rooms: {M_H.divide_rooms}")
        print(f"Slip Probability: {M_H.slip_prob:.2f}")

    # Find maximally achievable subsets
    I, B = find_maximally_achievable_subsets(M_R, M_H_list)
    
    print("\nMaximally achievable subsets of bottleneck states:")
    for subset in I:
        print(subset)
    print("\nAll bottleneck states:", B)

   
