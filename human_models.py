import random
import numpy as np
from simple_rl_envs import GridWorldUnifiedMDP, PuddleUnifiedMDP, TaxiUnifiedMDP
from determinizeMDP import DeterminizedMDP, find_bottleneck_states

def generate_human_models(robot_mdp, mdp_class, num_models, wall_change_prob=0.5, prob_noise=0.4):
    if mdp_class == GridWorldUnifiedMDP:
        return generate_grid_world_human_models(robot_mdp, num_models, wall_change_prob, prob_noise)
    elif mdp_class == TaxiUnifiedMDP:
        return generate_taxi_human_models(robot_mdp, num_models, prob_noise)
    elif mdp_class == PuddleUnifiedMDP:
        return generate_puddle_human_models(robot_mdp, num_models, prob_noise)
    else:
        raise ValueError(f"Unsupported MDP class: {mdp_class}")

def generate_grid_world_human_models(robot_mdp, num_models, wall_change_prob=0.5, prob_noise=0.4):
    def modify_grid(walls):
        new_walls = set(walls)
        all_positions = set((x, y) for x in range(robot_mdp.width) for y in range(robot_mdp.height))
        non_walls = all_positions - new_walls - {robot_mdp.init_loc} - set(robot_mdp.goal_locs)
        
        for pos in list(new_walls):
            if random.random() < wall_change_prob:
                new_walls.remove(pos)
        
        for pos in list(non_walls):
            if random.random() < wall_change_prob:
                new_walls.add(pos)
        
        return list(new_walls)

    def ensure_connectivity(walls):
        def bfs(start, goal, width, height, walls):
            queue = [(start, [])]
            visited = set([start])
            
            while queue:
                (x, y), path = queue.pop(0)
                if (x, y) == goal:
                    return path
                
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls and (nx, ny) not in visited:
                        queue.append(((nx, ny), path + [(nx, ny)]))
                        visited.add((nx, ny))
            return None

        if not bfs(robot_mdp.init_loc, robot_mdp.goal_locs[0], robot_mdp.width, robot_mdp.height, walls):
            return robot_mdp.walls
        return walls

    human_models = []
    while len(human_models) < num_models:
        new_walls = ensure_connectivity(modify_grid(robot_mdp.walls))
        
        try:
            # modify slip probability
            slip_prob = np.clip(robot_mdp.slip_prob + np.random.normal(0, prob_noise), 0, 1)
            
            human_mdp = GridWorldUnifiedMDP(robot_mdp.width, robot_mdp.height, 
                                            robot_mdp.init_loc, robot_mdp.goal_locs, 
                                            walls=new_walls,
                                            slip_prob=slip_prob)
            
            human_det_mdp = DeterminizedMDP(human_mdp)
            bottlenecks = find_bottleneck_states(human_det_mdp)
            
            if bottlenecks:
                human_models.append(human_mdp)
                print(f"GridWorld human model {len(human_models)} created with bottlenecks: {bottlenecks}")
        except Exception as e:
            print(f"Error creating GridWorld MDP: {e}. Retrying...")

    return human_models

def generate_taxi_human_models(robot_mdp, num_models, prob_noise=0.4):
    human_models = []
    while len(human_models) < num_models:
        try:
            # Randomly modify passenger and destination locations
            passenger_locs = [(x + np.random.randint(-1, 2), y + np.random.randint(-1, 2)) 
                              for x, y in robot_mdp.passenger_locs]
            passenger_locs = [(max(0, min(x, robot_mdp.width - 1)), max(0, min(y, robot_mdp.height - 1))) 
                              for x, y in passenger_locs]
            
            destination_locs = [(x + np.random.randint(-1, 2), y + np.random.randint(-1, 2)) 
                                for x, y in robot_mdp.destination_locs]
            destination_locs = [(max(0, min(x, robot_mdp.width - 1)), max(0, min(y, robot_mdp.height - 1))) 
                                for x, y in destination_locs]
            
            human_mdp = TaxiUnifiedMDP(robot_mdp.width, robot_mdp.height, 
                                       passenger_locs, destination_locs)
            
            human_det_mdp = DeterminizedMDP(human_mdp)
            bottlenecks = find_bottleneck_states(human_det_mdp)
            
            if bottlenecks:
                human_models.append(human_mdp)
                print(f"Taxi human model {len(human_models)} created with bottlenecks: {bottlenecks}")
        except Exception as e:
            print(f"Error creating Taxi MDP: {e}. Retrying...")

    return human_models

def generate_puddle_human_models(robot_mdp, num_models, prob_noise=0.4):
    human_models = []
    while len(human_models) < num_models:
        try:
            # randomly modify puddle locations and sizes
            puddle_rects = []
            for rect in robot_mdp.puddle_rects:
                x1, y1, x2, y2 = rect
                x1 += np.random.normal(0, 0.05)
                y1 += np.random.normal(0, 0.05)
                x2 += np.random.normal(0, 0.05)
                y2 += np.random.normal(0, 0.05)
                puddle_rects.append((max(0, min(x1, 1)), max(0, min(y1, 1)), 
                                     max(0, min(x2, 1)), max(0, min(y2, 1))))
            
            # randomly modify goal location
            goal_loc = [(x + np.random.normal(0, 0.05), y + np.random.normal(0, 0.05)) 
                        for x, y in robot_mdp.goal_locs]
            goal_loc = [(max(0, min(x, 1)), max(0, min(y, 1))) for x, y in goal_loc]
            
            human_mdp = PuddleUnifiedMDP(puddle_rects, goal_loc)
            
            # Modify step size
            human_mdp.step_size = max(0.01, min(0.1, robot_mdp.step_size + np.random.normal(0, 0.01)))
            
            print(f"Created PuddleUnifiedMDP with step_size: {human_mdp.step_size}")
            print(f"Puddle rects: {human_mdp.puddle_rects}")
            print(f"Goal locations: {human_mdp.goal_locs}")
            
            human_det_mdp = DeterminizedMDP(human_mdp)
            bottlenecks = find_bottleneck_states(human_det_mdp)
            
            if bottlenecks:
                human_models.append(human_mdp)
                print(f"Puddle human model {len(human_models)} created with bottlenecks: {bottlenecks}")
            else:
                print("No bottlenecks found for this model, discarding")
        except Exception as e:
            print(f"Error creating Puddle MDP: {e}. Retrying...")

    return human_models

def print_mdp(mdp):
    if isinstance(mdp, GridWorldUnifiedMDP):
        for y in range(mdp.height):
            row = []
            for x in range(mdp.width):
                if (x, y) in mdp.walls:
                    row.append(' # ')
                elif (x, y) == mdp.init_loc:
                    row.append(' S ')
                elif (x, y) in mdp.goal_locs:
                    row.append(' G ')
                else:
                    row.append(' . ')
            print(''.join(row))
        print(f"Slip probability: {mdp.slip_prob:.2f}")
    elif isinstance(mdp, TaxiUnifiedMDP):
        for y in range(mdp.height):
            row = []
            for x in range(mdp.width):
                if (x, y) in mdp.passenger_locs:
                    row.append(' P ')
                elif (x, y) in mdp.destination_locs:
                    row.append(' D ')
                else:
                    row.append(' . ')
            print(''.join(row))
    elif isinstance(mdp, PuddleUnifiedMDP):
        resolution = 20
        grid = [[' . ' for _ in range(resolution)] for _ in range(resolution)]
        
        for rect in mdp.puddle_rects:
            x1, y1, x2, y2 = rect
            for i in range(int(y1 * resolution), int(y2 * resolution) + 1):
                for j in range(int(x1 * resolution), int(x2 * resolution) + 1):
                    grid[i][j] = ' # '
        
        for goal in mdp.goal_locs:
            x, y = goal
            grid[int(y * resolution)][int(x * resolution)] = ' G '
        
        for row in grid:
            print(''.join(row))
        print(f"Step size: {mdp.step_size:.2f}")
    else:
        print("Unsupported MDP type for printing")

if __name__ == "__main__":
    # Test GridWorldUnifiedMDP
    robot_grid_mdp = GridWorldUnifiedMDP(5, 5, (0, 0), [(4, 4)], walls=[(1, 1), (2, 2), (3, 3)])
    grid_human_models = generate_human_models(robot_grid_mdp, GridWorldUnifiedMDP, num_models=2)
    
    print("Robot GridWorld MDP:")
    print_mdp(robot_grid_mdp)
    for i, model in enumerate(grid_human_models):
        print(f"\nHuman GridWorld Model {i+1}:")
        print_mdp(model)
    
    # Test TaxiUnifiedMDP
    robot_taxi_mdp = TaxiUnifiedMDP(5, 5, [(0, 0), (4, 4)], [(0, 4), (4, 0)])
    taxi_human_models = generate_human_models(robot_taxi_mdp, TaxiUnifiedMDP, num_models=2)
    
    print("\nRobot Taxi MDP:")
    print_mdp(robot_taxi_mdp)
    for i, model in enumerate(taxi_human_models):
        print(f"\nHuman Taxi Model {i+1}:")
        print_mdp(model)
    
    # Test PuddleUnifiedMDP
    robot_puddle_mdp = PuddleUnifiedMDP([(0.1, 0.8, 0.5, 0.7), (0.4, 0.7, 0.5, 0.4)], [(1.0, 1.0)])
    puddle_human_models = generate_human_models(robot_puddle_mdp, PuddleUnifiedMDP, num_models=2)
    
    print("\nRobot Puddle MDP:")
    print_mdp(robot_puddle_mdp)
    for i, model in enumerate(puddle_human_models):
        print(f"\nHuman Puddle Model {i+1}:")
        print_mdp(model)