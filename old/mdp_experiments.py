import numpy as np
import random
#from determinized_mdp import DeterminizedMDP
from mdp import MDP
from old.bottleneck_analysis import find_bottleneck_states, find_max_bottleneck_policy
from old.algorithm2 import optimal_query
from visualization import visualize_grid, visualize_grid_with_values
from human_models import generate_human_models, determinize_model

def generate_complex_grid(n_rows, n_cols, wall_density=0.2):
    grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
    state_counter = 1  
    for i in range(n_rows):
        for j in range(n_cols):
            if random.random() > wall_density or state_counter <= n_rows * n_cols * 0.5:  # Ensure at least 50% non-wall states
                grid[i][j] = state_counter
                state_counter += 1
            else:
                grid[i][j] = 0 
    

    start_state = 1  # S1 is always the first state
    start_i, start_j = None, None

    # find the position of S1 in the grid
    for i in range(n_rows):
        for j in range(n_cols):
            if grid[i][j] == start_state:
                start_i, start_j = i, j
                break
        if start_i is not None:
            break
    
    # set goal state
    while True:
        goal_i, goal_j = random.randint(0, n_rows-1), random.randint(0, n_cols-1)
        if grid[goal_i][goal_j] > 0 and grid[goal_i][goal_j] != start_state:
            goal_state = grid[goal_i][goal_j]
            break
    
    return grid, start_state - 1, goal_state - 1  

def generate_robot_walls(grid, num_walls):
    n_rows, n_cols = len(grid), len(grid[0])
    available_cells = [(i, j) for i in range(n_rows) for j in range(n_cols) if grid[i][j] != 0 and grid[i][j] != 1 and grid[i][j] != len(grid) * len(grid[0])]
    robot_walls = random.sample(available_cells, min(num_walls, len(available_cells)))
    return set(grid[i][j] - 1 for i, j in robot_walls)

def create_mdp_from_grid(grid, start_state, goal_state, robot_walls=None, human_walls=None):
    n_rows, n_cols = len(grid), len(grid[0])
    S = [s-1 for row in grid for s in row if s != 0] 
    A = ['up', 'down', 'left', 'right']
    
    if robot_walls is None:
        robot_walls = set()
    if human_walls is None:
        human_walls = set()
    
    robot_walls = set(robot_walls) - {start_state, goal_state}
    human_walls = set(human_walls) - {start_state, goal_state}
    
    mdp = MDP(S, A, start_state, goal_state, human_walls)
    
    for i in range(n_rows):
        for j in range(n_cols):
            if grid[i][j] == 0:  # common wall
                continue
            s = grid[i][j] - 1  # convert to 0-based indexing
            if s in human_walls or s in robot_walls:
                continue  # skip transitions for human and robot walls
            
            valid_actions = []
            if i > 0 and grid[i-1][j] != 0 and grid[i-1][j] - 1 not in human_walls and grid[i-1][j] - 1 not in robot_walls:
                valid_actions.append(('up', grid[i-1][j] - 1))
            if i < n_rows - 1 and grid[i+1][j] != 0 and grid[i+1][j] - 1 not in human_walls and grid[i+1][j] - 1 not in robot_walls:
                valid_actions.append(('down', grid[i+1][j] - 1))
            if j > 0 and grid[i][j-1] != 0 and grid[i][j-1] - 1 not in human_walls and grid[i][j-1] - 1 not in robot_walls:
                valid_actions.append(('left', grid[i][j-1] - 1))
            if j < n_cols - 1 and grid[i][j+1] != 0 and grid[i][j+1] - 1 not in human_walls and grid[i][j+1] - 1 not in robot_walls:
                valid_actions.append(('right', grid[i][j+1] - 1))
            
            # Add transitions with equal probability for all valid actions
            n_actions = len(valid_actions)
            if n_actions > 0:
                prob = 1.0 / n_actions
                for action, next_state in valid_actions:
                    mdp.add_transition(s, action, next_state, prob)
    
    return mdp

def generate_human_walls(grid, num_walls):
    empty_cells = [(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j] != 0 and grid[i][j] != 1 and grid[i][j] != len(grid) * len(grid[0])]
    human_walls = random.sample(empty_cells, min(num_walls, len(empty_cells)))
    return set(grid[i][j] - 1 for i, j in human_walls)  

def run_experiment(n_rows, n_cols, num_human_walls, num_robot_walls):
    grid, start_state, goal_state = generate_complex_grid(n_rows, n_cols)
    
    # create robot and human grids
    robot_grid = [row[:] for row in grid]
    human_grid = [row[:] for row in grid]
    
    # generate robot-only walls
    robot_walls = generate_robot_walls(robot_grid, num_robot_walls)
    
    # generate human-only walls
    human_walls = generate_human_walls(human_grid, num_human_walls)
    
    # create MDPs
    robot_mdp = create_mdp_from_grid(robot_grid, start_state, goal_state, robot_walls=robot_walls)
    human_mdp = create_mdp_from_grid(human_grid, start_state, goal_state, human_walls=human_walls)
    
    print(f"initial state: S{start_state + 1}")
    print(f"goal state: S{goal_state + 1}")

    # Generate human models
    n_states = len(robot_mdp.S)
    n_actions = len(robot_mdp.A)
    human_models = generate_human_models(n_models=3, n_states=n_states, n_actions=n_actions, 
                                         human_walls=human_walls, start_state=start_state, 
                                         goal_state=goal_state)
    
    print("Generated human models:")
    for i, model in enumerate(human_models):
        print(f"Human Model {i + 1}:")
        print(model)
        print()

    # Determinize all human models and compute bottlenecks
    all_human_bottlenecks = set()
    for i, human_model in enumerate(human_models):
        determinized_models = determinize_model(human_model)
        #print(f"Number of determinized models for Human Model {i + 1}: {len(determinized_models)}")
        
        for j, det_model in enumerate(determinized_models):
            #print(f"Processing Determinized Model {j + 1} of Human Model {i + 1}")
            #print(f"Shape of det_model: {det_model.shape}")
            bottlenecks = find_bottleneck_states(det_model, robot_mdp.S, robot_mdp.A, start_state, goal_state)
            all_human_bottlenecks.update(bottlenecks)
            
            #print(f"Bottlenecks for Determinized Model {j + 1} of Human Model {i + 1}:")
            #print([s + 1 for s in bottlenecks])
        #print()

    print("All human bottlenecks:", [s + 1 for s in all_human_bottlenecks])

    print("Finding robot bottlenecks:")
    robot_bottlenecks = find_bottleneck_states(robot_mdp, robot_mdp.S, robot_mdp.A, robot_mdp.s0, robot_mdp.G)
    print("Robot's bottleneck states:", [s + 1 for s in robot_bottlenecks])

    # Combine robot and human bottlenecks
    all_bottlenecks = list(set(robot_bottlenecks) | all_human_bottlenecks)
    print("Combined bottleneck states:", [s + 1 for s in all_bottlenecks])

    best_policy, traversed_bottlenecks, num_traversed = find_max_bottleneck_policy(robot_mdp, all_bottlenecks)
    
    print("Traversed bottleneck states:", [s + 1 for s in traversed_bottlenecks])
    print(f"Number of traversed bottlenecks: {num_traversed}")

    non_traversed_bottlenecks = set(all_bottlenecks) - set(traversed_bottlenecks)
    print("Non-traversed bottleneck states:", [s + 1 for s in non_traversed_bottlenecks])

    visualize_grid(robot_grid, all_bottlenecks, robot_mdp.s0, robot_mdp.G, best_policy, human_walls, robot_walls, traversed_bottlenecks, non_traversed_bottlenecks)

    V = robot_mdp.get_value_function()
    visualize_grid_with_values(robot_grid, V, all_bottlenecks, robot_mdp.s0, robot_mdp.G, best_policy, human_walls, robot_walls, traversed_bottlenecks, non_traversed_bottlenecks)

def run_multiple_experiments(n_experiments, n_rows, n_cols, num_human_walls, num_robot_walls):
    for i in range(n_experiments):
        print(f"\n\nExperiment {i+1}:")
        run_experiment(n_rows, n_cols, num_human_walls, num_robot_walls)
