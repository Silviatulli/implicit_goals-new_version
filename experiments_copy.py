from PuddleWorldClass import PuddleWorld, generate_and_visualize_puddleworld
from GridWorldClass import GridWorld, generate_and_visualize_gridworld
from DeterminizedMDP import identify_bottlenecks
from BottleneckCheckMDP import BottleneckMDP
from QueryMDP import QueryMDP
from Utils import vectorized_value_iteration
import numpy as np
import time
import random

def run_experiments(num_runs=5, num_models=3, grid_size=5):
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
        print(f"starting run {run + 1}/{num_runs}")
        start_time_total = time.time()

        # generate robot model
        print("generating robot model...")
        start_time = time.time()
        M_R = generate_and_visualize_gridworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                               obstacles_percent=0.1, divide_rooms=True, 
                                               model_type="Robot Model", obstacle_seed=random.randint(1, 10000))
        generation_time = time.time() - start_time
        results["generation_times"].append(generation_time)
        print(f"robot model generated in {generation_time:.2f} seconds")

        # identify robot bottlenecks
        print("identifying robot bottlenecks...")
        start_time = time.time()
        bottleneck_states_robot = identify_bottlenecks(M_R)
        bottleneck_states_robot = [b for b in bottleneck_states_robot if b[0] != M_R.goal_pos]
        bottleneck_time = time.time() - start_time
        results["bottleneck_times"].append(bottleneck_time)
        print(f"robot bottlenecks identified in {bottleneck_time:.2f} seconds")

        # generate human models and identify their bottlenecks
        print("generating human models and identifying their bottlenecks...")
        all_human_bottlenecks = set()
        start_time_human = time.time()
        for i in range(num_models):
            M_H = generate_and_visualize_gridworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                   obstacles_percent=0.1, divide_rooms=True, 
                                                   model_type=f"Human Model {i+1}", 
                                                   obstacle_seed=random.randint(1, 10000))
            bottleneck_states_human = identify_bottlenecks(M_H)
            bottleneck_states_human = [b for b in bottleneck_states_human if b[0] != M_H.goal_pos]
            all_human_bottlenecks.update(bottleneck_states_human)
        bottleneck_time = time.time() - start_time_human
        results["bottleneck_times"].append(bottleneck_time)
        print(f"human models generated and bottlenecks identified in {bottleneck_time:.2f} seconds")

        results["human_bottlenecks"].append(len(all_human_bottlenecks))

        # run query mdp
        print("running query mdp...")
        start_time = time.time()
        achievable_bottlenecks = [b for b in all_human_bottlenecks if M_R.map[b[0]] != -1]
        non_achievable_bottlenecks = [b for b in all_human_bottlenecks if b not in achievable_bottlenecks]
        query_mdp = QueryMDP(M_R, achievable_bottlenecks, non_achievable_bottlenecks)
        policy, bottleneck_mdp, explanations = query_mdp.run()
        query_time = time.time() - start_time
        results["query_times"].append(query_time)
        print(f"query mdp completed in {query_time:.2f} seconds")

        results["query_counts"].append(len(query_mdp.necessary_bottlenecks))
        results["query_costs"].append(query_mdp.query_cost)
        results["total_times"].append(time.time() - start_time_total)

        print(f"run {run + 1} completed in {time.time() - start_time_total:.2f} seconds")

    return results

def print_results(results):
    print("\nexperiment results:")
    print(f"number of runs: {num_runs}")
    print(f"number of human models: {num_models}")
    print(f"grid size: {grid_size}x{grid_size}")
    
    for metric, values in results.items():
        print(f"{metric}: {np.mean(values):.2f} ± {np.std(values):.2f}")

if __name__ == "__main__":
    num_runs = 5
    num_models = 3
    grid_size = 5
    
    print("starting experiments...")
    results = run_experiments(num_runs, num_models, grid_size)
    print_results(results)
