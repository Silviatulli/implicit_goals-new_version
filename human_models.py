import random
from MDPgrid import GridWorldMDP
from collections import deque
from copy import deepcopy
import numpy as np
from determinizeMDP import DeterminizedMDP, find_bottleneck_states


def generate_human_models(robot_mdp, num_models, wall_change_prob=0.5, prob_noise=0.4):
    def modify_grid(grid):
        new_grid = [row[:] for row in grid]
        counter = robot_mdp.n_states + 1
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if random.random() < wall_change_prob:
                    if cell == 0:
                        new_grid[i][j] = counter
                        counter += 1
                    elif cell not in (robot_mdp.initial_state + 1, robot_mdp.goal_state + 1):
                        new_grid[i][j] = 0
        return new_grid

    def ensure_connectivity(grid):
        def find_state(state_num):
            return next((i, j) for i, row in enumerate(grid) for j, val in enumerate(row) if val == state_num)

        start = find_state(robot_mdp.initial_state + 1)
        goal = find_state(robot_mdp.goal_state + 1)

        def bfs(start, goal):
            queue = [(start, [])]
            visited = set([start])
            
            while queue:
                (x, y), path = queue.pop(0)
                if (x, y) == goal:
                    return path
                
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] != 0 and (nx, ny) not in visited:
                        queue.append(((nx, ny), path + [(nx, ny)]))
                        visited.add((nx, ny))
            return None

        if not bfs(start, goal):
            for i, row in enumerate(grid):
                for j, cell in enumerate(row):
                    if cell == 0 and robot_mdp.grid[i][j] != 0:
                        grid[i][j] = robot_mdp.grid[i][j]
        return grid

    def modify_probabilities(mdp):
        move_prob = np.clip(mdp.move_prob + np.random.normal(0, prob_noise), 0, 1)
        stay_prob = np.clip(mdp.stay_prob + np.random.normal(0, prob_noise), 0, 1)
        
        total_prob = move_prob + stay_prob
        if total_prob > 0:
            mdp.move_prob = move_prob / total_prob
            mdp.stay_prob = stay_prob / total_prob
        else:
            mdp.move_prob = 0.8  
            mdp.stay_prob = 0.2 
        mdp._initialize_transitions_and_rewards(mdp.R[mdp.goal_state])

    human_models = []
    while len(human_models) < num_models:
        new_grid = ensure_connectivity(modify_grid(robot_mdp.grid))
        
        try:
            human_mdp = GridWorldMDP(new_grid, initial_state=robot_mdp.initial_state + 1, goal_state=robot_mdp.goal_state + 1)
            modify_probabilities(human_mdp)
            
            human_det_mdp = DeterminizedMDP(human_mdp)
            bottlenecks = find_bottleneck_states(human_det_mdp)
            
            if bottlenecks:
                human_models.append(human_mdp)
                print(f"human model {len(human_models)} created with bottlenecks: {[s + 1 for s in bottlenecks]}")
            #else:
                #print("model created but no bottlenecks found, retrying...")
        except Exception as e:
            print(f"error creating MDP: {e}. Retrying...")

    return human_models

def print_grid_with_policy(mdp, policy):
    action_symbols = ['↑', '↓', '←', '→']
    grid = mdp.grid
    for i in range(len(grid)):
        row = []
        for j in range(len(grid[0])):
            state = grid[i][j]
            if state == 0:
                row.append(' # ')
            elif state - 1 == mdp.goal_state:
                row.append(' G ')
            elif state - 1 == mdp.initial_state:
                row.append(' S ')
            else:
                action = policy[state - 1]
                row.append(f' {action_symbols[action]} ')
        print(''.join(row))


if __name__ == "__main__":

    from MDPgrid import GridWorldMDP

    # define the robot grid
    robot_grid = [
        [1, 2, 3, 4],
        [5, 6, 0, 0],
        [0, 7, 8, 0]
    ]


    # create the robot MDP
    robot_mdp = GridWorldMDP(robot_grid, initial_state=1, goal_state=8)

    det_policy, stoch_policy, V = robot_mdp.get_optimal_policies(temperature=0.1)

    robot_mdp.print_policy(det_policy, is_deterministic=True)
        

    human_models = generate_human_models(robot_mdp, num_models=2, wall_change_prob=0.5, prob_noise=0.4)


    print("robot grid:")
    for row in robot_grid:
        print(' '.join(f'{cell:2}' for cell in row))
    det_robot_mdp = DeterminizedMDP(robot_mdp)
    bottlenecks = find_bottleneck_states(det_robot_mdp)
    print("robot bottlenecks:", [s + 1 for s in bottlenecks])
       
    for i, human_model in enumerate(human_models):
        print(f"\nHuman Model {i+1} Grid:")
        for row in human_model.grid:
            print(' '.join(f'{cell:2}' for cell in row))
        print(f"move probability: {human_model.move_prob:.2f}")
        print(f"stay probability: {human_model.stay_prob:.2f}")

        human_det_mdp = DeterminizedMDP(human_model)
        print(f"human Model {i+1} Bottlenecks:")
        bottlenecks = find_bottleneck_states(human_det_mdp)
        if bottlenecks:
            print([s + 1 for s in bottlenecks])
        else:
            print("no bottlenecks found")

        # Visualize the policy
        Q, V = human_model.value_iteration()
        policy = human_model.compute_deterministic_policy(Q)
        print("\npolicy:")
        print_grid_with_policy(human_model, policy)
