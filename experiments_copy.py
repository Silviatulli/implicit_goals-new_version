from PuddleWorldClass import PuddleWorld
from RockWorldClass import RockWorld
from TaxiWorldClass import TaxiWorld
from MinigridWorldClass import UnlockEnv, UnlockPickupEnv
from GridWorldClass import generate_and_visualize_gridworld
from DeterminizedMDP import identify_bottlenecks
from BottleneckCheckMDP import BottleneckMDP
from QueryMDP import QueryMDP, simulate_policy
from Utils import vectorized_value_iteration, get_policy, sparse_value_iteration
import numpy as np
import time
import random
from maximal_achievable_subsets import optimized_find_maximally_achievable_subsets

# def generate_and_visualize_puddleworld(size, start, goal, obstacles_percent, puddle_percent, model_type, obstacle_seed):
#     return PuddleWorld(size=size, start=start, goal=goal, 
#                        obstacles_percent=obstacles_percent,
#                        puddle_percent=puddle_percent,
#                        obstacle_seed=obstacle_seed)

# def generate_and_visualize_rockworld(size, start, goal, obstacles_percent, rock_percent, model_type, obstacle_seed):
#     return RockWorld(size=size, start=start, goal=goal, 
#                      obstacles_percent=obstacles_percent,
#                      rock_percent=rock_percent,
#                      obstacle_seed=obstacle_seed)

# def generate_and_visualize_taxiworld(size, start, goal, obstacles_percent, model_type, obstacle_seed):
#     passenger_loc = (random.randint(0, size-1), random.randint(0, size-1))
#     return TaxiWorld(size=size, start=start, passenger_loc=passenger_loc, destination=goal, 
#                      obstacles_percent=obstacles_percent,
#                      obstacle_seed=obstacle_seed)

def run_experiments(num_runs, num_models, grid_size, world_type, query_threshold):
    results = {
        "query_counts": [],
        "human_bottlenecks": [],
        "maximal_subset_times": [],
        "query_times": []
    }

    for run in range(num_runs):
        print(f"Starting run {run + 1}/{num_runs}")
        start_time_total = time.time()

        # Generate robot model
        print("Generating robot model...")
        M_R = generate_and_visualize_gridworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                               obstacles_percent=0.1, divide_rooms=True, 
                                               model_type="Robot Model", obstacle_seed=random.randint(1, 10000))

        # Generate human models
        print("Generating human models...")
        M_H_list = []
        for i in range(num_models):
            M_H = generate_and_visualize_gridworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                   obstacles_percent=0.1, divide_rooms=True, 
                                                   model_type=f"Human Model {i+1}", 
                                                   obstacle_seed=random.randint(1, 10000))
            M_H_list.append(M_H)

        # Find maximally achievable subsets
        print("Finding maximally achievable subsets...")
        start_time_maximal = time.time()
        I, B = optimized_find_maximally_achievable_subsets(M_R, M_H_list)
        maximal_subset_time = time.time() - start_time_maximal
        results["maximal_subset_times"].append(maximal_subset_time)
        print(f"Maximally achievable subsets found in {maximal_subset_time:.2f} seconds")

        results["human_bottlenecks"].append(len(B))

        print("Running query MDP...")
        start_time = time.time()
        query_mdp = QueryMDP(M_R, list(B), list(I))
    
        print("Starting value iteration...")
        V = vectorized_value_iteration(query_mdp)
        print("Value iteration completed. Extracting policy...")
        policy = get_policy(query_mdp, V)
        print("Policy extracted. Simulating policy...")
    
        query_count = simulate_policy(query_mdp, list(B), query_threshold)
        query_time = time.time() - start_time
        results["query_times"].append(query_time)
        print(f"Query MDP completed in {query_time:.2f} seconds")

        results["query_counts"].append(query_count)

        print(f"Run {run + 1} completed in {time.time() - start_time_total:.2f} seconds")

    return results


def print_results(results, num_runs, num_models, grid_size, world_type):
    print("\nExperiment Results:")
    print(f"World Type: {world_type}")
    print(f"Number of runs: {num_runs}")
    print(f"Number of human models: {num_models}")
    print(f"Grid size: {grid_size}x{grid_size}")
    
    for metric, values in results.items():
        print(f"{metric}: {np.mean(values):.2f} ± {np.std(values):.2f}")

if __name__ == "__main__":
    num_runs = 1
    num_models = 3
    grid_size = 4
    query_threshold = 1000 
    
    #world_types = ['grid', 'puddle', 'rock', 'taxi']
    world_types = ['grid']
    
    for world_type in world_types:
        print(f"\nStarting experiments for {world_type.capitalize()}World...")
        results = run_experiments(num_runs, num_models, grid_size, world_type, query_threshold)
        print_results(results, num_runs, num_models, grid_size, world_type)
