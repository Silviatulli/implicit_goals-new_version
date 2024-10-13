from PuddleWorldClass import PuddleWorld
from RockWorldClass import RockWorld
from TaxiWorldClass import TaxiWorld
from GridWorldClass_copy import generate_and_visualize_gridworld
from DeterminizedMDP import identify_bottlenecks
from BottleneckCheckMDP import BottleneckMDP
from QueryMDP import QueryMDP, simulate_policy
from Utils import vectorized_value_iteration, get_policy
import numpy as np
import time
import random

def generate_and_visualize_puddleworld(size, start, goal, obstacles_percent, puddle_percent, model_type, obstacle_seed):
    return PuddleWorld(size=size, start=start, goal=goal, 
                       obstacles_percent=obstacles_percent,
                       puddle_percent=puddle_percent,
                       obstacle_seed=obstacle_seed)

def generate_and_visualize_rockworld(size, start, goal, obstacles_percent, rock_percent, model_type, obstacle_seed):
    return RockWorld(size=size, start=start, goal=goal, 
                     obstacles_percent=obstacles_percent,
                     rock_percent=rock_percent,
                     obstacle_seed=obstacle_seed)

def generate_and_visualize_taxiworld(size, start, goal, obstacles_percent, model_type, obstacle_seed):
    passenger_loc = (random.randint(0, size-1), random.randint(0, size-1))
    return TaxiWorld(size=size, start=start, passenger_loc=passenger_loc, destination=goal, 
                     obstacles_percent=obstacles_percent,
                     obstacle_seed=obstacle_seed)

def run_experiments(num_runs=5, num_models=3, grid_size=5, world_type='grid', query_threshold=1000):
    results = {
        "query_counts": [],
        "query_costs": [],
        "human_bottlenecks": [],
        "total_times": [],
        "generation_times": [],
        "bottleneck_times": [],
        "query_times": []
    }

    for run in range(num_runs):
        print(f"Starting run {run + 1}/{num_runs}")
        start_time_total = time.time()

        # Generate robot model
        print("Generating robot model...")
        start_time = time.time()
        if world_type == 'grid':
            M_R = generate_and_visualize_gridworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                   obstacles_percent=0.1, divide_rooms=True, 
                                                   model_type="Robot Model", obstacle_seed=random.randint(1, 10000))
        elif world_type == 'puddle':
            M_R = generate_and_visualize_puddleworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                     obstacles_percent=0.1, puddle_percent=0.2,
                                                     model_type="Robot Model", obstacle_seed=random.randint(1, 10000))
        elif world_type == 'rock':
            M_R = generate_and_visualize_rockworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                   obstacles_percent=0.1, rock_percent=0.3,
                                                   model_type="Robot Model", obstacle_seed=random.randint(1, 10000))
        elif world_type == 'taxi':
            M_R = generate_and_visualize_taxiworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                   obstacles_percent=0.1,
                                                   model_type="Robot Model", obstacle_seed=random.randint(1, 10000))
        generation_time = time.time() - start_time
        results["generation_times"].append(generation_time)
        print(f"Robot model generated in {generation_time:.2f} seconds")

        # Identify robot bottlenecks
        print("Identifying robot bottlenecks...")
        start_time = time.time()
        bottleneck_states_robot = identify_bottlenecks(M_R)
        bottleneck_states_robot = [b for b in bottleneck_states_robot if b[0] != M_R.goal_pos]
        bottleneck_time = time.time() - start_time
        results["bottleneck_times"].append(bottleneck_time)
        print(f"Robot bottlenecks identified in {bottleneck_time:.2f} seconds")

        # Generate human models and identify their bottlenecks
        print("Generating human models and identifying their bottlenecks...")
        all_human_bottlenecks = set()
        start_time_human = time.time()
        for i in range(num_models):
            if world_type == 'grid':
                M_H = generate_and_visualize_gridworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                       obstacles_percent=0.1, divide_rooms=True, 
                                                       model_type=f"Human Model {i+1}", 
                                                       obstacle_seed=random.randint(1, 10000))
            elif world_type == 'puddle':
                M_H = generate_and_visualize_puddleworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                         obstacles_percent=0.1, puddle_percent=0.2,
                                                         model_type=f"Human Model {i+1}", 
                                                         obstacle_seed=random.randint(1, 10000))
            elif world_type == 'rock':
                M_H = generate_and_visualize_rockworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                       obstacles_percent=0.1, rock_percent=0.3,
                                                       model_type=f"Human Model {i+1}", 
                                                       obstacle_seed=random.randint(1, 10000))
            elif world_type == 'taxi':
                M_H = generate_and_visualize_taxiworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                       obstacles_percent=0.1,
                                                       model_type=f"Human Model {i+1}", 
                                                       obstacle_seed=random.randint(1, 10000))
            bottleneck_states_human = identify_bottlenecks(M_H)
            bottleneck_states_human = [b for b in bottleneck_states_human if b[0] != M_H.goal_pos]
            all_human_bottlenecks.update(bottleneck_states_human)
        bottleneck_time = time.time() - start_time_human
        results["bottleneck_times"].append(bottleneck_time)
        print(f"Human models generated and bottlenecks identified in {bottleneck_time:.2f} seconds")

        results["human_bottlenecks"].append(len(all_human_bottlenecks))

        print("Running query MDP...")
        start_time = time.time()
        achievable_bottlenecks = [b for b in all_human_bottlenecks if M_R.map[b[0][0]][b[0][1]] != -1]
        non_achievable_bottlenecks = [b for b in all_human_bottlenecks if b not in achievable_bottlenecks]
        query_mdp = QueryMDP(M_R, achievable_bottlenecks, non_achievable_bottlenecks)
        
        print("Starting value iteration...")
        V = vectorized_value_iteration(query_mdp)
        print("Value iteration completed. Extracting policy...")
        policy = get_policy(query_mdp, V)
        print("Policy extracted. Simulating policy...")
        
        # Pass the achievable_bottlenecks as the true bottlenecks
        query_count = simulate_policy(query_mdp, achievable_bottlenecks)
        
        query_time = time.time() - start_time
        results["query_times"].append(query_time)
        print(f"Query MDP completed in {query_time:.2f} seconds")

        results["query_counts"].append(query_count)
        results["query_costs"].append(query_count)  # Assuming each query has a cost of 1
        results["total_times"].append(time.time() - start_time_total)

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
    num_runs = 5
    num_models = 3
    grid_size = 5
    query_threshold = 1000  # You can adjust this value as needed
    
    world_types = ['grid', 'puddle', 'rock', 'taxi']
    
    for world_type in world_types:
        print(f"\nStarting experiments for {world_type.capitalize()}World...")
        results = run_experiments(num_runs, num_models, grid_size, world_type, query_threshold)
        print_results(results, num_runs, num_models, grid_size, world_type)
