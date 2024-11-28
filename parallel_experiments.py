from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Value
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set, FrozenSet
import time
import os
import random
import signal
import sys

from experiments import generate_and_visualize_gridworld, generate_and_visualize_puddleworld, generate_and_visualize_rockworld
from maximal_achievable_subsets import find_maximally_achievable_subsets, find_maximally_achievable_subsets_no_pruning, improved_find_maximally_achievable_subsets
from QueryMDP import QueryMDP, simulate_policy_unachievable


pruning_counter = Value('i', 0)
no_pruning_counter = Value('i', 0)

def simulate_policy_query_all(query_mdp: QueryMDP, human_bottlenecks: List[Any], query_threshold: int = 1000) -> int:
    """Simple baseline strategy that queries all bottleneck states sequentially."""
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

def run_single_experiment(params: Dict[str, Any]):
    """Run a single experiment with given parameters."""
    world_type = params['world_type']
    grid_size = params['grid_size']
    num_models = params['num_models']
    query_threshold = params['query_threshold']
    obstacle_percent = params['obstacle_percent']
    puddle_percent = params['puddle_percent']
    rock_percent = params['rock_percent']
    
    # Generate robot model
    M_R = None
    if world_type == 'grid':
        M_R = generate_and_visualize_gridworld(
            size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
            obstacles_percent=obstacle_percent, divide_rooms=False, 
            model_type="Robot Model", obstacle_seed=random.randint(1, 10000)
        )
    elif world_type == 'four_rooms':
        M_R = generate_and_visualize_gridworld(
            size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
            obstacles_percent=obstacle_percent, divide_rooms=True, 
            model_type="Robot Model", obstacle_seed=random.randint(1, 10000)
        )
    elif world_type == 'puddle':
        M_R = generate_and_visualize_puddleworld(
            size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
            obstacles_percent=obstacle_percent, puddle_percent=puddle_percent, 
            model_type="Robot Model", obstacle_seed=random.randint(1, 10000)
        )
    elif world_type == 'rock':
        M_R = generate_and_visualize_rockworld(
            size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
            obstacles_percent=obstacle_percent, rock_percent=rock_percent, 
            model_type="Robot Model", obstacle_seed=random.randint(1, 10000)
        )
    
    if not M_R:
        return None
        
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
        "query_counts": [],
        "query_times": [],
        "query_all_counts": [],
        "query_all_times": [],
        "human_bottlenecks": [],
        "initial_mdp_state_space_sizes": [],
        "initial_mdp_action_space_sizes": [],
        "query_mdp_state_space_sizes": [],
        "query_mdp_action_space_sizes": []
    }
    
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
                obstacles_percent=obstacle_percent, divide_rooms=False, 
                model_type=f"Human Model {i+1}", 
                obstacle_seed=random.randint(1, 10000)
            )
        elif world_type == 'four_rooms':
            M_H = generate_and_visualize_gridworld(
                size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                obstacles_percent=obstacle_percent, divide_rooms=True, 
                model_type=f"Human Model {i+1}", 
                obstacle_seed=random.randint(1, 10000)
            )
        elif world_type == 'puddle':
            M_H = generate_and_visualize_puddleworld(
                size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                obstacles_percent=obstacle_percent, puddle_percent=puddle_percent, 
                model_type=f"Human Model {i+1}", 
                obstacle_seed=random.randint(1, 10000)
            )
        elif world_type == 'rock':
            M_H = generate_and_visualize_rockworld(
                size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                obstacles_percent=obstacle_percent, rock_percent=rock_percent, 
                model_type=f"Human Model {i+1}", 
                obstacle_seed=random.randint(1, 10000)
            )
        if M_H:
            M_H_list.append(M_H)

    # Run pruning version
    with pruning_counter.get_lock():
        start_time = time.time()
        I_pruning, B = improved_find_maximally_achievable_subsets(M_R, M_H_list)
        pruning_time = time.time() - start_time
        check_count = pruning_counter.value
        pruning_counter.value = 0  # Reset for next run
    
    results["pruning"]["times"].append(pruning_time)
    results["pruning"]["checks"].append(check_count)
    results["pruning"]["subsets"].append(len(I_pruning))
    results["human_bottlenecks"].append(len(B))
    
    # Run no-pruning version if state space is small enough
    if len(B) <= 15:
        with no_pruning_counter.get_lock():
            start_time = time.time()
            I_no_pruning, _ = find_maximally_achievable_subsets_no_pruning(M_R, M_H_list)
            no_pruning_time = time.time() - start_time
            check_count = no_pruning_counter.value
            no_pruning_counter.value = 0  # Reset for next run
        
        results["no_pruning"]["times"].append(no_pruning_time)
        results["no_pruning"]["checks"].append(check_count)
        results["no_pruning"]["subsets"].append(len(I_no_pruning))
    
    # Run query experiments if bottlenecks exist
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
    
    return results

def run_parallel_experiments(num_runs: int, num_models: int, grid_size: int, world_types: list, 
                           query_threshold: int, max_workers: int = None):
    """Run experiments in parallel using ProcessPoolExecutor."""
    all_environments_results = {}
    
    # Create experiment parameters
    experiment_params = []
    for world_type in world_types:
        for _ in range(num_runs):
            params = {
                'world_type': world_type,
                'grid_size': grid_size,
                'num_models': num_models,
                'query_threshold': query_threshold
            }
            experiment_params.append(params)
    
    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_experiment, params) for params in experiment_params]
        
        # Collect results
        for i, future in enumerate(futures):
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(run_single_experiment, params) for _, params in experiment_params]
                result = future.result()
                if result:
                    world_type = experiment_params[i]['world_type']
                    if world_type not in all_environments_results:
                        all_environments_results[world_type] = {
                            "pruning": {"times": [], "checks": [], "subsets": []},
                            "no_pruning": {"times": [], "checks": [], "subsets": []},
                            "query_counts": [],
                            "query_times": [],
                            "query_all_counts": [],
                            "query_all_times": [],
                            "human_bottlenecks": [],
                            "initial_mdp_state_space_sizes": [],
                            "initial_mdp_action_space_sizes": [],
                            "query_mdp_state_space_sizes": [],
                            "query_mdp_action_space_sizes": []
                        }
                    
                    # Aggregate results
                    for key in result:
                        if isinstance(result[key], dict):
                            for subkey in result[key]:
                                all_environments_results[world_type][key][subkey].extend(result[key][subkey])
                        else:
                            all_environments_results[world_type][key].extend(result[key])
            except KeyboardInterrupt:
                print("\nStopping all processes...")
                executor.shutdown(wait=False)
                sys.exit(0)
    
    return all_environments_results

# Copy the create_combined_results_table function from the original experiments.py
def create_combined_results_table(all_environments_results, output_file="experiment_results/combined_comparison.csv"):
    """Create table showing both subset finding and query strategy ablations"""
    combined_data = {
        'Environment': [],
        'Pruning Time (s)': [],
        'No Pruning Time (s)': [],
        'Pruning Speedup': [],
        'Strategic Query Count': [],
        'Query-All Count': [],
        'Query Reduction (%)': [],
        'Strategic Query Time (s)': [],
        'Query-All Time (s)': [],
        'Query Speedup': [],
        'Human Bottlenecks': [],
        'Initial State Space': [],
        'Initial Actions': []
    }
    
    for env_type, results in all_environments_results.items():
        combined_data['Environment'].append(env_type)
        
        # Subset Finding Ablation metrics
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
            
        # Query Strategy Ablation metrics
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
    
    # Create DataFrame and save results
    df = pd.DataFrame(combined_data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return df

def run_parallel_experiments_with_obstacles(num_runs: int, num_models: int, grid_size: int, 
                                    world_types: list, query_threshold: int, 
                                    obstacle_percentages: list, max_workers: int = None):
    """Run experiments in parallel for different obstacle percentages."""
    all_environments_results = {}
    
    # Create experiment parameters
    experiment_params = []
    for world_type in world_types:
        if world_type == 'four_rooms':
            # Four rooms environment uses fixed 0% obstacles
            world_config = f"{world_type}_0.0"
            for _ in range(num_runs):
                params = {
                    'world_type': world_type,
                    'grid_size': grid_size,
                    'num_models': num_models,
                    'query_threshold': query_threshold,
                    'obstacle_percent': 0.0,
                    'puddle_percent': 0.0,
                    'rock_percent': 0.0
                }
                experiment_params.append((world_config, params))
        else:
            # Other environments test different obstacle percentages
            for obstacle_percent in obstacle_percentages:
                world_config = f"{world_type}_{obstacle_percent}"
                for _ in range(num_runs):
                    params = {
                        'world_type': world_type,
                        'grid_size': grid_size,
                        'num_models': num_models,
                        'query_threshold': query_threshold,
                        'obstacle_percent': obstacle_percent,
                        'puddle_percent': obstacle_percent,
                        'rock_percent': obstacle_percent
                    }
                    experiment_params.append((world_config, params))
    
    # Rest of the function remains the same...
    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_experiment, params) for _, params in experiment_params]
        
        # Collect results as before...
        for i, future in enumerate(futures):
            try:
                result = future.result()
                if result:
                    world_config = experiment_params[i][0]
                    if world_config not in all_environments_results:
                        all_environments_results[world_config] = {
                            "pruning": {"times": [], "checks": [], "subsets": []},
                            "no_pruning": {"times": [], "checks": [], "subsets": []},
                            "query_counts": [],
                            "query_times": [],
                            "query_all_counts": [],
                            "query_all_times": [],
                            "human_bottlenecks": [],
                            "initial_mdp_state_space_sizes": [],
                            "initial_mdp_action_space_sizes": [],
                            "query_mdp_state_space_sizes": [],
                            "query_mdp_action_space_sizes": []
                        }
                    
                    # Aggregate results
                    for key in result:
                        if isinstance(result[key], dict):
                            for subkey in result[key]:
                                all_environments_results[world_config][key][subkey].extend(result[key][subkey])
                        else:
                            all_environments_results[world_config][key].extend(result[key])
            except Exception as e:
                print(f"Error in experiment {i}: {e}")
    
    return all_environments_results


if __name__ == "__main__":
    def signal_handler(signum, frame):
        print("\nStopping all processes...")
        if 'executor' in globals():
            executor.shutdown(wait=False)
        sys.exit(0)
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    # Set experiment parameters
    num_runs = 5
    num_models = 20
    grid_size = 5
    query_threshold = 1000
    world_types = ['grid', 'four_rooms', 'puddle', 'rock']
    obstacle_percentages = [0.1, 0.2, 0.3]  # Only used for non-four_rooms environments
    max_workers = 4
    
    # Create results directory
    os.makedirs("experiment_results", exist_ok=True)
    
    # Run parallel experiments with different obstacle percentages
    results = run_parallel_experiments_with_obstacles(
        num_runs=num_runs,
        num_models=num_models,
        grid_size=grid_size,
        world_types=world_types,
        query_threshold=query_threshold,
        obstacle_percentages=obstacle_percentages,
        max_workers=max_workers
    )
    
    # Create and save combined results table
    combined_df = create_combined_results_table(results)
    
    # Print summary for each environment and obstacle percentage
    for env_type in world_types:
        print(f"\nResults for {env_type}:")
        print("=" * 80)
        env_data = combined_df[combined_df['Environment'] == env_type]
        print(env_data.to_string(index=False))
