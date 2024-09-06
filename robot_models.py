import random
import numpy as np
from unified_mdp import GridWorldUnifiedMDP, TaxiUnifiedMDP, PuddleUnifiedMDP
from simple_rl_envs import GridWorldUnifiedMDP, TaxiUnifiedMDP, PuddleUnifiedMDP

def generate_diverse_mdps(mdp_class, env_size, num_mdps=1, wall_density=0.3):
    if mdp_class == GridWorldUnifiedMDP:
        return generate_diverse_gridworlds(env_size, num_mdps, wall_density)
    elif mdp_class == TaxiUnifiedMDP:
        return generate_diverse_taxis(env_size, num_mdps)
    elif mdp_class == PuddleUnifiedMDP:
        return generate_diverse_puddles(num_mdps)
    else:
        raise ValueError(f"Unsupported MDP class: {mdp_class}")

def generate_diverse_gridworlds(grid_sizes, num_mdps=1, wall_density=0.3):
    diverse_mdps = []

    # convert grid_sizes to a list if it's a single tuple
    if isinstance(grid_sizes, tuple):
        grid_sizes = [grid_sizes]

    for grid_size in grid_sizes:
        width, height = grid_size
        for _ in range(num_mdps):
            walls = generate_grid(width, height, wall_density)
            attempts = 0
            max_attempts = 100
            
            while not ensure_path_exists(width, height, walls) and attempts < max_attempts:
                walls = generate_grid(width, height, wall_density)
                attempts += 1
            
            if attempts == max_attempts:
                print(f"Warning: Could not generate a valid grid for size {width}x{height} after {max_attempts} attempts.")
                continue

            mdp = GridWorldUnifiedMDP(width, height, (0, 0), [(height-1, width-1)], walls=list(walls))
            diverse_mdps.append(mdp)

    return diverse_mdps

def generate_grid(width, height, wall_density):
    total_cells = width * height
    num_walls = int(total_cells * wall_density)
    
    all_positions = [(i, j) for i in range(height) for j in range(width)]
    all_positions.remove((0, 0))
    all_positions.remove((height-1, width-1))
    
    wall_positions = set(random.sample(all_positions, num_walls))
    
    return wall_positions

def ensure_path_exists(width, height, walls):
    start = (0, 0)
    goal = (height-1, width-1)

    def get_neighbors(x, y):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and (nx, ny) not in walls:
                neighbors.append((nx, ny))
        return neighbors

    queue = [start]
    visited = set([start])

    while queue:
        current = queue.pop(0)
        if current == goal:
            return True

        for neighbor in get_neighbors(*current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return False


def generate_diverse_taxis(env_size, num_mdps=1):
    width, height = env_size
    diverse_mdps = []

    for _ in range(num_mdps):
        num_passengers = random.randint(1, 3)
        passenger_locs = random.sample([(x, y) for x in range(width) for y in range(height)], num_passengers)
        destination_locs = random.sample([(x, y) for x in range(width) for y in range(height)], num_passengers)

        mdp = TaxiUnifiedMDP(width, height, passenger_locs, destination_locs)
        diverse_mdps.append(mdp)

    return diverse_mdps

def generate_diverse_puddles(num_mdps=1):
    diverse_mdps = []

    for _ in range(num_mdps):
        num_puddles = random.randint(1, 3)
        puddle_rects = []
        for _ in range(num_puddles):
            x1, x2 = sorted(np.random.uniform(0, 1, 2))
            y1, y2 = sorted(np.random.uniform(0, 1, 2))
            puddle_rects.append((x1, y1, x2, y2))

        goal_loc = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        mdp = PuddleUnifiedMDP(puddle_rects, [goal_loc])
        diverse_mdps.append(mdp)

    return diverse_mdps

def print_grid(mdp):
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
    else:
        print("Unsupported MDP type for printing")

if __name__ == "__main__":
    grid_sizes = [(4, 4), (5, 5)]
    wall_density = 0.3

    print("GridWorld MDPs:")
    for size in grid_sizes:
        robot_mdps = generate_diverse_mdps(GridWorldUnifiedMDP, size, num_mdps=1, wall_density=wall_density)   
        for i, mdp in enumerate(robot_mdps):
            print(f"\nGridWorld MDP {i+1} (size: {size}):")
            print_grid(mdp)

    print("\nTaxi MDPs:")
    for size in grid_sizes:
        robot_mdps = generate_diverse_mdps(TaxiUnifiedMDP, size, num_mdps=1)
        for i, mdp in enumerate(robot_mdps):
            print(f"\nTaxi MDP {i+1} (size: {size}):")
            print_grid(mdp)

    print("\nPuddle MDPs:")
    robot_mdps = generate_diverse_mdps(PuddleUnifiedMDP, None, num_mdps=1)
    for i, mdp in enumerate(robot_mdps):
        print(f"\nPuddle MDP {i+1}:")
        print_grid(mdp)