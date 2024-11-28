from typing import List, Any, Set, Tuple, FrozenSet
from PuddleWorldClass import PuddleWorld
from RockWorldClass import RockWorld 
from TaxiWorldClass import TaxiWorld
from MinigridWorldClass import UnlockEnv, UnlockPickupEnv
from GridWorldClass import generate_and_visualize_gridworld
from BottleneckCheckMDP import BottleneckMDP
from QueryMDP import QueryMDP, simulate_policy, simulate_policy_unachievable
from Utils import vectorized_value_iteration, get_policy, sparse_value_iteration, get_sparse_policy
from maximal_achievable_subsets import find_maximally_achievable_subsets, identify_bottlenecks, check_achievability, find_maximally_achievable_subsets_no_pruning
from DeterminizedMDP import DeterminizedMDP
import numpy as np
import time
import random
import pandas as pd
from tqdm import tqdm
import os

def visualize_taxiworld(taxi_world):
    print(f"Grid Size: {taxi_world.size}x{taxi_world.size}")
    print(f"Start: {taxi_world.start_pos}")
    print(f"Passengers: {taxi_world.passenger_locs}")
    print(f"Destinations: {taxi_world.destinations}")
    
    for i in range(taxi_world.size):
        for j in range(taxi_world.size):
            if (i, j) == taxi_world.start_pos:
                print("S", end=" ")
            elif (i, j) in taxi_world.passenger_locs:
                print(f"P{taxi_world.passenger_locs.index((i,j))+1}", end=" ")
            elif (i, j) in taxi_world.destinations:
                print(f"D{taxi_world.destinations.index((i,j))+1}", end=" ")
            elif taxi_world.map[i, j] == -1:
                print("█", end=" ")
            else:
                print(".", end=" ")
        print()

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
    num_passengers = 2
    passenger_locs = [(random.randint(0, size-1), random.randint(0, size-1)) for _ in range(num_passengers)]
    destinations = [(random.randint(0, size-1), random.randint(0, size-1)) for _ in range(num_passengers)]
    
    taxi_world = TaxiWorld(
        size=size,
        num_passengers=num_passengers,
        start=start,
        passenger_locs=passenger_locs,
        destinations=destinations,
        obstacles_percent=obstacles_percent,
        obstacle_seed=obstacle_seed
    )
    
    print(f"\n{model_type}:")
    visualize_taxiworld(taxi_world)
    
    return taxi_world

def generate_and_visualize_minigridworld(env_type, size, model_type):
    if env_type == 'unlock':
        return UnlockEnv(size=size)
    elif env_type == 'unlock_pickup':
        return UnlockPickupEnv(size=size)
    else:
        raise ValueError(f"Unknown Minigrid environment type: {env_type}")

def simulate_policy_query_all(query_mdp: QueryMDP, human_bottlenecks: List[Any], query_threshold: int = 1000) -> int:
    """
    Simple baseline strategy that queries all bottleneck states sequentially.
    """
    human_bottleneck_hash = frozenset(query_mdp.robot_mdp.get_state_hash(state) 
                                    for state in human_bottlenecks)
    query_count = 0
    confirmed_subgoals = set()
    confirmed_non_subgoals = set()
    all_bottlenecks = query_mdp.bottleneck_hash
    
    for bottleneck in all_bottlenecks:
        if query_count >= query_threshold:
            return query_threshold
            
        query_count += 1
        is_subgoal = bottleneck in human_bottleneck_hash
        
        if is_subgoal:
            confirmed_subgoals.add(bottleneck)
        else:
            confirmed_non_subgoals.add(bottleneck)
            
        current_state = (frozenset(confirmed_subgoals), frozenset(confirmed_non_subgoals))
        if query_mdp.check_terminal_state(current_state):
            print(f"Found all bottlenecks after {query_count} queries")
            return query_count
    
    return query_threshold

def run_experiments_with_ablations(num_runs, num_models, grid_size, world_type, query_threshold):
    """
    Enhanced experiment runner that includes both ablation studies:
    1. Pruning vs No-pruning for maximal achievable subsets
    2. Strategic querying vs Query-all approach
    """
    results = {
        "baseline": {
            "query_counts": [],
            "human_bottlenecks": [],
            "maximal_subset_times": [],
            "query_times": [],
            "query_mdp_state_space_sizes": [],
            "query_mdp_action_space_sizes": [],
            "initial_mdp_state_space_sizes": [],
            "initial_mdp_action_space_sizes": []
        },
        "no_pruning": {
            "maximal_subset_times": [],
            "num_maximal_subsets": []
        },
        "query_all": {
            "query_counts": [],
            "query_times": []
        }
    }

    for run in tqdm(range(num_runs), desc="Running experiments"):
        print(f"\nStarting run {run + 1}/{num_runs}")
        
        # Generate robot model
        M_R = None
        if world_type == 'grid':
            M_R = generate_and_visualize_gridworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                               obstacles_percent=0.1, divide_rooms=False, 
                                               model_type="Robot Model", obstacle_seed=random.randint(1, 10000))
        elif world_type == 'four_rooms':
            M_R = generate_and_visualize_gridworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                               obstacles_percent=0, divide_rooms=True, 
                                               model_type="Robot Model", obstacle_seed=random.randint(1, 10000))
        elif world_type == 'puddle':
            M_R = generate_and_visualize_puddleworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                 obstacles_percent=0.1, puddle_percent=0.1, 
                                                 model_type="Robot Model", obstacle_seed=random.randint(1, 10000))
        elif world_type == 'rock':
            M_R = generate_and_visualize_rockworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                               obstacles_percent=0.1, rock_percent=0.1, 
                                               model_type="Robot Model", obstacle_seed=random.randint(1, 10000))

        if not M_R:
            print(f"Failed to generate robot model for {world_type}")
            continue

        results["baseline"]["initial_mdp_state_space_sizes"].append(len(M_R.state_space))
        results["baseline"]["initial_mdp_action_space_sizes"].append(len(M_R.get_actions()))

        # Generate human models
        M_H_list = []
        for i in range(num_models):
            M_H = None
            if world_type == 'grid':
                M_H = generate_and_visualize_gridworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                   obstacles_percent=0.1, divide_rooms=False, 
                                                   model_type=f"Human Model {i+1}", 
                                                   obstacle_seed=random.randint(1, 10000))
            elif world_type == 'four_rooms':
                M_H = generate_and_visualize_gridworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                   obstacles_percent=0, divide_rooms=True, 
                                                   model_type=f"Human Model {i+1}", 
                                                   obstacle_seed=random.randint(1, 10000))
            elif world_type == 'puddle':
                M_H = generate_and_visualize_puddleworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                     obstacles_percent=0.1, puddle_percent=0.1, 
                                                     model_type=f"Human Model {i+1}", 
                                                     obstacle_seed=random.randint(1, 10000))
            elif world_type == 'rock':
                M_H = generate_and_visualize_rockworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                   obstacles_percent=0.1, rock_percent=0.1, 
                                                   model_type=f"Human Model {i+1}", 
                                                   obstacle_seed=random.randint(1, 10000))
            if M_H:
                M_H_list.append(M_H)

        # Run original maximal achievable subsets algorithm
        start_time = time.time()
        I, B = find_maximally_achievable_subsets(M_R, M_H_list)
        original_time = time.time() - start_time
        results["baseline"]["maximal_subset_times"].append(original_time)
        results["baseline"]["human_bottlenecks"].append(len(B))

        # Run no-pruning version
        if len(B) <= 15:  # Only run exhaustive search for smaller state spaces
            start_time = time.time()
            I_no_pruning, _ = find_maximally_achievable_subsets_no_pruning(M_R, M_H_list)
            no_pruning_time = time.time() - start_time
            results["no_pruning"]["maximal_subset_times"].append(no_pruning_time)
            results["no_pruning"]["num_maximal_subsets"].append(len(I_no_pruning))
        
        # Skip if no bottlenecks found
        if len(B) == 0:
            print("No bottlenecks found - skipping query experiments")
            continue

        # Create QueryMDP
        query_mdp = QueryMDP(M_R, list(B), list(I))
        results["baseline"]["query_mdp_state_space_sizes"].append(len(query_mdp.state_space))
        results["baseline"]["query_mdp_action_space_sizes"].append(len(query_mdp.get_actions()))

        # Run strategic querying
        start_time = time.time()
        query_count = simulate_policy_unachievable(query_mdp, list(B), query_threshold)
        query_time = time.time() - start_time
        results["baseline"]["query_counts"].append(query_count)
        results["baseline"]["query_times"].append(query_time)

        # Run query-all baseline
        start_time = time.time()
        query_all_count = simulate_policy_query_all(query_mdp, list(B), query_threshold)
        query_all_time = time.time() - start_time
        results["query_all"]["query_counts"].append(query_all_count)
        results["query_all"]["query_times"].append(query_all_time)

    return results

def print_and_save_results_with_ablations(all_results, num_runs, num_models, grid_size, output_prefix="experiment"):
    """Enhanced results printer that includes ablation study results"""
    
    # Original metrics
    main_df = pd.DataFrame({
        'Metric': [
            'Average Query Count',
            'Average Human Bottlenecks',
            'Average Maximal Subset Time (s)',
            'Average Query Time (s)',
            'Average Initial State Space Size',
            'Average Initial Action Space Size',
            'Average Query MDP State Space Size',
            'Average Query MDP Action Space Size'
        ],
        'Value': [
            np.mean(all_results["baseline"]["query_counts"]),
            np.mean(all_results["baseline"]["human_bottlenecks"]),
            np.mean(all_results["baseline"]["maximal_subset_times"]),
            np.mean(all_results["baseline"]["query_times"]),
            np.mean(all_results["baseline"]["initial_mdp_state_space_sizes"]),
            np.mean(all_results["baseline"]["initial_mdp_action_space_sizes"]),
            np.mean(all_results["baseline"]["query_mdp_state_space_sizes"]),
            np.mean(all_results["baseline"]["query_mdp_action_space_sizes"])
        ]
    })

    # Ablation results
    ablation_df = pd.DataFrame({
        'Metric': [
            'Pruning vs No-Pruning Time Ratio',
            'Strategic vs Query-All Query Count Ratio',
            'Strategic vs Query-All Time Ratio'
        ],
        'Value': [
            np.mean(all_results["no_pruning"]["maximal_subset_times"]) / 
            np.mean(all_results["baseline"]["maximal_subset_times"]) if all_results["no_pruning"]["maximal_subset_times"] else "N/A",
            
            np.mean(all_results["query_all"]["query_counts"]) / 
            np.mean(all_results["baseline"]["query_counts"]) if all_results["baseline"]["query_counts"] else "N/A",
            
            np.mean(all_results["query_all"]["query_times"]) / 
            np.mean(all_results["baseline"]["query_times"]) if all_results["baseline"]["query_times"] else "N/A"
        ]
    })

    # Print results
    print("\nMain Results:")
    print(main_df)
    print("\nAblation Study Results:")
    print(ablation_df)

    # Save to CSV
    main_df.to_csv(f'{output_prefix}_main_results.csv', index=False)
    ablation_df.to_csv(f'{output_prefix}_ablation_results.csv', index=False)
    print(f"\nResults saved to {output_prefix}_main_results.csv and {output_prefix}_ablation_results.csv")


def create_combined_results_table(all_environments_results, output_file="experiment_results/combined_comparison.csv"):
    """Create table showing both subset finding and query strategy ablations"""
    combined_data = {
        'Environment': [],
        # Subset Finding Ablation
        'Pruning Time (s)': [],
        'No Pruning Time (s)': [],
        'Pruning Speedup': [],
        # Query Strategy Ablation
        'Strategic Query Count': [],
        'Query-All Count': [],
        'Query Reduction (%)': [],
        'Strategic Query Time (s)': [],
        'Query-All Time (s)': [],
        'Query Speedup': [],
        # Key MDP metrics
        'Human Bottlenecks': [],
        'Initial State Space': [],
        'Initial Actions': []
    }
    
    for env_type, results in all_environments_results.items():
        combined_data['Environment'].append(env_type)
        
        # Subset Finding Ablation
        pruning_times = results["pruning"]["times"]
        no_pruning_times = results["no_pruning"]["times"]
        
        combined_data['Pruning Time (s)'].append(f"{np.mean(pruning_times):.3f}")
        if no_pruning_times:
            combined_data['No Pruning Time (s)'].append(f"{np.mean(no_pruning_times):.3f}")
            speedup = np.mean(no_pruning_times) / np.mean(pruning_times)
            combined_data['Pruning Speedup'].append(f"{speedup:.2f}x")
        else:
            combined_data['No Pruning Time (s)'].append("N/A")
            combined_data['Pruning Speedup'].append("N/A")
        
        # Query Strategy Ablation
        strategic_queries = results.get('query_counts', [])
        query_all_counts = results.get('query_all_counts', [])
        strategic_times = results.get('query_times', [])
        query_all_times = results.get('query_all_times', [])
        
        if strategic_queries and query_all_counts:
            combined_data['Strategic Query Count'].append(f"{np.mean(strategic_queries):.1f}")
            combined_data['Query-All Count'].append(f"{np.mean(query_all_counts):.1f}")
            reduction = (1 - np.mean(strategic_queries) / np.mean(query_all_counts)) * 100
            combined_data['Query Reduction (%)'].append(f"{reduction:.1f}")
            
            combined_data['Strategic Query Time (s)'].append(f"{np.mean(strategic_times):.3f}")
            combined_data['Query-All Time (s)'].append(f"{np.mean(query_all_times):.3f}")
            query_speedup = np.mean(query_all_times) / np.mean(strategic_times)
            combined_data['Query Speedup'].append(f"{query_speedup:.2f}x")
        else:
            combined_data['Strategic Query Count'].append("N/A")
            combined_data['Query-All Count'].append("N/A")
            combined_data['Query Reduction (%)'].append("N/A")
            combined_data['Strategic Query Time (s)'].append("N/A")
            combined_data['Query-All Time (s)'].append("N/A")
            combined_data['Query Speedup'].append("N/A")
        
        # MDP metrics
        combined_data['Human Bottlenecks'].append(f"{np.mean(results['human_bottlenecks']):.1f}")
        combined_data['Initial State Space'].append(f"{np.mean(results['initial_mdp_state_space_sizes']):.0f}")
        combined_data['Initial Actions'].append(f"{np.mean(results['initial_mdp_action_space_sizes']):.0f}")
    
    # Create DataFrame
    df = pd.DataFrame(combined_data)
    
    # Print formatted tables by section
    print("\nSubset Finding Ablation:")
    print("=" * 80)
    subset_columns = ['Environment', 'Pruning Time (s)', 'No Pruning Time (s)', 'Pruning Speedup']
    print(df[subset_columns].to_string(index=False))
    
    print("\nQuery Strategy Ablation:")
    print("=" * 80)
    query_columns = ['Environment', 'Strategic Query Count', 'Query-All Count', 'Query Reduction (%)',
                    'Strategic Query Time (s)', 'Query-All Time (s)', 'Query Speedup']
    print(df[query_columns].to_string(index=False))
    
    print("\nMDP Metrics:")
    print("=" * 80)
    mdp_columns = ['Environment', 'Human Bottlenecks', 'Initial State Space', 'Initial Actions']
    print(df[mdp_columns].to_string(index=False))
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    
    # Subset Finding summary
    valid_pruning_speedups = [float(x[:-1]) for x in combined_data['Pruning Speedup'] if x != "N/A"]
    if valid_pruning_speedups:
        print(f"Average subset finding speedup: {np.mean(valid_pruning_speedups):.2f}x")
    
    # Query Strategy summary
    valid_query_reductions = [float(x) for x in combined_data['Query Reduction (%)'] if x != "N/A"]
    valid_query_speedups = [float(x[:-1]) for x in combined_data['Query Speedup'] if x != "N/A"]
    if valid_query_reductions:
        print(f"Average query reduction: {np.mean(valid_query_reductions):.1f}%")
        print(f"Average query speedup: {np.mean(valid_query_speedups):.2f}x")
    
    # MDP metrics summary
    bottlenecks = [float(x) for x in combined_data['Human Bottlenecks']]
    print(f"Average bottlenecks: {np.mean(bottlenecks):.1f}")
    
    return df

if __name__ == "__main__":
    # experiment parameters
    num_runs = 5
    num_models = 100
    grid_size = 5
    query_threshold = 1000
    
    # Define all world types
    world_types = ['grid', 'four_rooms', 'puddle', 'rock']
    
    # Create results directory
    os.makedirs("experiment_results", exist_ok=True)
    
    # Dictionary to store results for all environments
    all_environments_results = {}
    
    for world_type in world_types:
        print(f"\nRunning ablation study for {world_type}...")
        
        # Results dictionary for this world type
        results = {
            "pruning": {
                "times": [],
                "checks": [],
                "subsets": []
            },
            "no_pruning": {
                "times": [],
                "checks": [],
                "subsets": []
            },
            # General MDP metrics
            "query_counts": [],
            "query_times": [],
            # Adding query-all metrics
            "query_all_counts": [],
            "query_all_times": [],
            "human_bottlenecks": [],
            "initial_mdp_state_space_sizes": [],
            "initial_mdp_action_space_sizes": [],
            "query_mdp_state_space_sizes": [],
            "query_mdp_action_space_sizes": []
        }

        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs}")
            
            # Generate robot model based on world type
            M_R = None
            if world_type == 'grid':
                M_R = generate_and_visualize_gridworld(
                    size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                    obstacles_percent=0.1, divide_rooms=False, 
                    model_type="Robot Model", obstacle_seed=random.randint(1, 10000)
                )
            elif world_type == 'four_rooms':
                M_R = generate_and_visualize_gridworld(
                    size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                    obstacles_percent=0, divide_rooms=True, 
                    model_type="Robot Model", obstacle_seed=random.randint(1, 10000)
                )
            elif world_type == 'puddle':
                M_R = generate_and_visualize_puddleworld(
                    size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                    obstacles_percent=0.1, puddle_percent=0.1, 
                    model_type="Robot Model", obstacle_seed=random.randint(1, 10000)
                )
            elif world_type == 'rock':
                M_R = generate_and_visualize_rockworld(
                    size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                    obstacles_percent=0.1, rock_percent=0.1, 
                    model_type="Robot Model", obstacle_seed=random.randint(1, 10000)
                )
            
            if not M_R:
                print(f"Failed to generate robot model for {world_type}")
                continue
                
            # Store initial MDP sizes
            results["initial_mdp_state_space_sizes"].append(len(M_R.state_space))
            results["initial_mdp_action_space_sizes"].append(len(M_R.get_actions()))
            
            # Generate human models
            M_H_list = []
            for i in range(num_models):
                M_H = None
                if world_type == 'grid':
                    M_H = generate_and_visualize_gridworld(
                        size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                        obstacles_percent=0.1, divide_rooms=False, 
                        model_type=f"Human Model {i+1}", 
                        obstacle_seed=random.randint(1, 10000)
                    )
                elif world_type == 'four_rooms':
                    M_H = generate_and_visualize_gridworld(
                        size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                        obstacles_percent=0, divide_rooms=True, 
                        model_type=f"Human Model {i+1}", 
                        obstacle_seed=random.randint(1, 10000)
                    )
                elif world_type == 'puddle':
                    M_H = generate_and_visualize_puddleworld(
                        size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                        obstacles_percent=0.1, puddle_percent=0.1, 
                        model_type=f"Human Model {i+1}", 
                        obstacle_seed=random.randint(1, 10000)
                    )
                elif world_type == 'rock':
                    M_H = generate_and_visualize_rockworld(
                        size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                        obstacles_percent=0.1, rock_percent=0.1, 
                        model_type=f"Human Model {i+1}", 
                        obstacle_seed=random.randint(1, 10000)
                    )
                if M_H:
                    M_H_list.append(M_H)
            
            # Run pruning version
            start_time = time.time()
            I_pruning, B = find_maximally_achievable_subsets(M_R, M_H_list)
            pruning_time = time.time() - start_time
            
            results["pruning"]["times"].append(pruning_time)
            results["pruning"]["checks"].append(achievability_check_count_pruning if 'achievability_check_count_pruning' in globals() else 0)
            results["pruning"]["subsets"].append(len(I_pruning))
            
            # Store number of bottlenecks
            results["human_bottlenecks"].append(len(B))
            
            # Run no-pruning version if state space is small enough
            if len(B) <= 15:
                start_time = time.time()
                I_no_pruning, _ = find_maximally_achievable_subsets_no_pruning(M_R, M_H_list)
                no_pruning_time = time.time() - start_time
                
                results["no_pruning"]["times"].append(no_pruning_time)
                results["no_pruning"]["checks"].append(achievability_check_count_no_pruning if 'achievability_check_count_no_pruning' in globals() else 0)
                results["no_pruning"]["subsets"].append(len(I_no_pruning))
            
            # Inside the main run loop, after creating QueryMDP:
            if len(B) > 0:
                query_mdp = QueryMDP(M_R, list(B), list(I_pruning))
                
                # Strategic querying
                start_time = time.time()
                strategic_count = simulate_policy_unachievable(query_mdp, list(B), query_threshold)
                strategic_time = time.time() - start_time
                results["query_counts"].append(strategic_count)
                results["query_times"].append(strategic_time)
                
                # Query-all approach
                start_time = time.time()
                query_all_count = simulate_policy_query_all(query_mdp, list(B), query_threshold)
                query_all_time = time.time() - start_time
                results["query_all_counts"].append(query_all_count)
                results["query_all_times"].append(query_all_time)
        
        all_environments_results[world_type] = results
    
    # Create and save combined results table
    combined_df = create_combined_results_table(all_environments_results)
