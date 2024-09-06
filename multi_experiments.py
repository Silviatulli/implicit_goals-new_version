import numpy as np
import time
from simple_rl_envs import GridWorldUnifiedMDP, TaxiUnifiedMDP, PuddleUnifiedMDP
from determinizeMDP import DeterminizedMDP, find_bottleneck_states
from human_models import generate_human_models
from queryMDP import optimal_query_strategy
from robot_models import generate_diverse_gridworlds, generate_diverse_taxis, generate_diverse_puddles

num_models = 2
num_runs = 2

def run_experiments(num_runs=num_runs, num_models=num_models):
    environments = [
        ("GridWorldMDP", GridWorldUnifiedMDP, (4, 4)),
        ("GridWorldMDP", GridWorldUnifiedMDP, (5, 5)),
        ("TaxiOOMDP", TaxiUnifiedMDP, (5, 5)),
        ("PuddleMDP", PuddleUnifiedMDP, None)
    ]
    results = {}

    for env_name, env_class, env_size in environments:
        env_results = {}
        query_costs, query_counts, times, human_bottlenecks = [], [], [], []
        generation_times, bottleneck_times, query_times = [], [], []

        for _ in range(num_runs):
            start_time_total = time.time()
            
            # Generate robot MDPs
            start_time = time.time()
            if env_class == GridWorldUnifiedMDP:
                robot_mdps = generate_diverse_gridworlds([env_size])
            elif env_class == TaxiUnifiedMDP:
                robot_mdps = generate_diverse_taxis(env_size)
            elif env_class == PuddleUnifiedMDP:
                robot_mdps = generate_diverse_puddles()
            else:
                raise ValueError(f"Unsupported environment class: {env_class}")
            
            generation_time = time.time() - start_time
            generation_times.append(generation_time)
            
            for mdp in robot_mdps:
                robot_det_mdp = DeterminizedMDP(mdp)
                
                # Find robot bottlenecks
                start_time = time.time()
                robot_bottlenecks = find_bottleneck_states(robot_det_mdp)
                bottleneck_time = time.time() - start_time
                bottleneck_times.append(bottleneck_time)
                
                # Generate human models
                start_time = time.time()
                human_models = generate_human_models(mdp, env_class, num_models=num_models)
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                
                print(f"Environment: {env_name}, Size: {env_size}")
                
                # Find all human bottlenecks
                all_human_bottlenecks = set()
                start_time_human = time.time()
                for human_model in human_models:
                    human_det_mdp = DeterminizedMDP(human_model)
                    human_bottlenecks = find_bottleneck_states(human_det_mdp)
                    all_human_bottlenecks.update(human_bottlenecks)
                bottleneck_time = time.time() - start_time_human
                bottleneck_times.append(bottleneck_time)
                
                human_bottlenecks.append(len(all_human_bottlenecks))

                # Run optimal query strategy
                start_time = time.time()
                queried_bottlenecks, total_cost = optimal_query_strategy(robot_det_mdp, list(all_human_bottlenecks), robot_bottlenecks)
                query_time = time.time() - start_time
                query_times.append(query_time)

                query_costs.append(total_cost)
                query_counts.append(len(queried_bottlenecks))
                times.append(time.time() - start_time_total)

        env_results[env_size] = {
            "Query Count": f"{np.mean(query_counts):.2f} ± {np.std(query_counts):.2f}",
            "Query Cost": f"{np.mean(query_costs):.2f} ± {np.std(query_costs):.2f}",
            "Human Bottlenecks": f"{np.mean(human_bottlenecks):.2f} ± {np.std(human_bottlenecks):.2f}",
            "Total Time (secs)": f"{np.mean(times):.2f} ± {np.std(times):.2f}",
            "MDP Generation Time (secs)": f"{np.mean(generation_times):.2f} ± {np.std(generation_times):.2f}",
            "Bottleneck Finding Time (secs)": f"{np.mean(bottleneck_times):.2f} ± {np.std(bottleneck_times):.2f}",
            "Query Strategy Time (secs)": f"{np.mean(query_times):.2f} ± {np.std(query_times):.2f}",
        }
        
        results[env_name] = env_results

    return results

def print_results(results):
    for env_name, env_results in results.items():
        print(f"\n{env_name}:")
        for env_size, metrics in env_results.items():
            print(f"  Environment size: {env_size}")
            print(f"  Number of runs: {num_runs}")
            print(f"  Number of human models: {num_models}")
            for metric, value in metrics.items():
                print(f"    {metric}: {value}")

if __name__ == "__main__":
    results = run_experiments()
    print_results(results)