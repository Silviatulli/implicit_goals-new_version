from MDPgrid import GridWorldMDP
import random
import numpy as np

def generate_diverse_gridworlds(grid_sizes, wall_density=0.3):
    def generate_grid(width, height):
        total_cells = width * height
        num_walls = int(total_cells * wall_density)
        
        # create a flat list of all possible positions
        all_positions = [(i, j) for i in range(height) for j in range(width)]
        
        # remove start and goal positions
        all_positions.remove((0, 0))
        all_positions.remove((height-1, width-1))
        
        # randomly select wall positions
        wall_positions = set(random.sample(all_positions, num_walls))
        
        grid = [[0 for _ in range(width)] for _ in range(height)]
        state_number = 1
        
        for i in range(height):
            for j in range(width):
                if (i, j) == (0, 0) or (i, j) == (height-1, width-1) or (i, j) not in wall_positions:
                    grid[i][j] = state_number
                    state_number += 1
        
        return grid

    def ensure_path_exists(grid):
        height, width = len(grid), len(grid[0])
        start = (0, 0)
        goal = (height-1, width-1)

        def get_neighbors(x, y):
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and grid[nx][ny] != 0:
                    neighbors.append((nx, ny))
            return neighbors

        queue = [start]
        visited = set([start])
        parent = {start: None}

        while queue:
            current = queue.pop(0)
            if current == goal:
                return True

            for neighbor in get_neighbors(*current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        return False

    diverse_mdps = []

    for width, height in grid_sizes:
        grid = generate_grid(width, height)
        attempts = 0
        max_attempts = 100  # limit the number of attempts to avoid infinite loops
        
        while not ensure_path_exists(grid) and attempts < max_attempts:
            grid = generate_grid(width, height)
            attempts += 1
        
        if attempts == max_attempts:
            print(f"Warning: Could not generate a valid grid for size {width}x{height} after {max_attempts} attempts.")
            continue

        initial_state = 1
        goal_state = max(max(row) for row in grid)
        mdp = GridWorldMDP(grid, initial_state=initial_state, goal_state=goal_state)
        diverse_mdps.append(mdp)

    return diverse_mdps

def print_grid(grid):
    for row in grid:
        print(' '.join(f'{cell:2}' if cell != 0 else ' #' for cell in row))


if __name__ == "__main__":
    grid_sizes = [(4, 4), (6, 6), (8, 8), (10, 10), (12, 12)]
    wall_density = 0.3  # you can adjust this value as needed

    robot_mdps = generate_diverse_gridworlds(grid_sizes, wall_density)

    for i, mdp in enumerate(robot_mdps):
        print(f"\nRobot MDP {i+1} (size: {len(mdp.grid[0])}x{len(mdp.grid)}):")
        print_grid(mdp.grid)
        print(f"Initial state: {mdp.initial_state}, Goal state: {mdp.goal_state}")