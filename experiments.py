import numpy as np
from MDPgrid import GridWorldMDP
from determinizeMDP import DeterminizedMDP, find_bottleneck_states
from algorithm1 import find_maximal_achievable_subsets
from human_models import generate_human_models
from queryMDP import optimal_query_strategy, update_robot_policy
from robot_models import generate_diverse_gridworlds
import time

num_models = 5
num_runs = 5

def run_experiments(num_runs=num_runs, num_models=num_models):
    grid_sizes = [(5, 5)]
    results = {}

    environments = [("GridWorldMDP", generate_human_models)]
    for env_name, env_generator in environments:
        env_results = {}
        for grid_size in grid_sizes:
            query_costs, query_counts, times, human_bottlenecks, query_time_ab, query_counts_ab = [], [], [], [], [], []
            generation_times, bottleneck_times, query_times = [], [], []
            for _ in range(num_runs):
                start_time_total = time.time()
                
                start_time = time.time()
                robot_mdps = generate_diverse_gridworlds([grid_size])
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                
                for mdp in robot_mdps:
                    robot_det_mdp = DeterminizedMDP(mdp)
                    
                    start_time = time.time()
                    robot_bottlenecks = find_bottleneck_states(robot_det_mdp)
                    bottleneck_time = time.time() - start_time
                    bottleneck_times.append(bottleneck_time)
                    
                    start_time = time.time()
                    human_models = env_generator(mdp, num_models=num_models, wall_change_prob=0.5, prob_noise=0.4)
                    generation_time = time.time() - start_time
                    generation_times.append(generation_time)
                    
                    print(mdp.grid)

                    # find all human bottlenecks
                    all_human_bottlenecks = set()
                    start_time_human = time.time()
                    for human_model in human_models:
                        human_det_mdp = DeterminizedMDP(human_model)
                        human_bottlenecks = find_bottleneck_states(human_det_mdp)
                        all_human_bottlenecks.update(human_bottlenecks)
                    bottleneck_time = time.time() - start_time_human
                    bottleneck_times.append(bottleneck_time)
                    
                    human_bottlenecks.append(len(all_human_bottlenecks))

                    # run optimal query strategy
                    start_time = time.time()
                    queried_bottlenecks, total_cost = optimal_query_strategy(robot_det_mdp, list(all_human_bottlenecks), robot_bottlenecks)
                    query_time = time.time() - start_time
                    query_times.append(query_time)

                    query_costs.append(total_cost)
                    query_counts.append(len(queried_bottlenecks))
                    times.append(time.time() - start_time_total)

                    # Ablation study: remove robot bottlenecks
                    start_time_ab = time.time()
                    ablation_queried_bottlenecks, _ = optimal_query_strategy(robot_det_mdp, 
                                                                             [b for b in all_human_bottlenecks], 
                                                                             [])
                    query_counts_ab.append(len(ablation_queried_bottlenecks))
                    query_time_ab.append(time.time() - start_time_ab)

            env_results[grid_size] = {
                "Query Count": f"{np.mean(query_counts):.2f} ± {np.std(query_counts):.2f}",
                "Query Cost": f"{np.mean(query_costs):.2f} ± {np.std(query_costs):.2f}",
                "Human Bottlenecks": f"{np.mean(human_bottlenecks):.2f} ± {np.std(human_bottlenecks):.2f}",
                "Total Time (secs)": f"{np.mean(times):.2f} ± {np.std(times):.2f}",
                "MDP Generation Time (secs)": f"{np.mean(generation_times):.2f} ± {np.std(generation_times):.2f}",
                "Bottleneck Finding Time (secs)": f"{np.mean(bottleneck_times):.2f} ± {np.std(bottleneck_times):.2f}",
                "Query Strategy Time (secs)": f"{np.mean(query_times):.2f} ± {np.std(query_times):.2f}",
                "Query Count (Ablation)": f"{np.mean(query_counts_ab):.2f} ± {np.std(query_counts_ab):.2f}",
                "Time (Ablation) (secs)": f"{np.mean(query_time_ab):.2f} ± {np.std(query_time_ab):.2f}",
            }
        
        results[env_name] = env_results

    return results

def print_results(results):
    for env_name, env_results in results.items():
        print(f"\n{env_name}:")
        for grid_size, metrics in env_results.items():
            print(f"  Grid size: {grid_size}")
            print(f"  Number of runs: {num_runs}")
            print(f"  Number of human models: {num_models}")
            for metric, value in metrics.items():
                print(f"    {metric}: {value}")

if __name__ == "__main__":
    results = run_experiments()
    print_results(results)