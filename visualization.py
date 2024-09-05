import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from matplotlib.colors import LinearSegmentedColormap

def visualize_grid(grid, bottlenecks, initial_state, goal_state, policy, human_walls, robot_walls, traversed_bottlenecks, non_traversed_bottlenecks):
    n_rows = len(grid)
    n_cols = len(grid[0])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect('equal')

    cmap = LinearSegmentedColormap.from_list("CustomMap", ["white", "lightblue"])
    
    for i in range(n_rows):
        for j in range(n_cols):
            state = grid[i][j]
            if state == 0:
                rect = Rectangle((j, n_rows-1-i), 1, 1, facecolor='black', edgecolor='black')
                ax.add_patch(rect)
            else:
                state_idx = state - 1
                if state_idx == goal_state:
                    color = 'gold'
                elif state_idx in human_walls:
                    color = 'purple'
                elif state_idx in robot_walls:
                    color = 'darkgray'
                elif state_idx in traversed_bottlenecks:
                    color = 'lightgreen'
                elif state_idx in non_traversed_bottlenecks:
                    color = 'orange'
                else:
                    color = cmap(0.5)
                rect = Rectangle((j, n_rows-1-i), 1, 1, facecolor=color, edgecolor='black')
                ax.add_patch(rect)
                
                ax.text(j+0.5, n_rows-1-i+0.5, f"S{state}", ha='center', va='center', fontsize=10, color='black')

                if state_idx == initial_state:
                    ax.text(j+0.5, n_rows-1-i+0.8, "Initial", ha='center', va='center', fontsize=12, color='blue', fontweight='bold')
                
                if state_idx == goal_state:
                    ax.text(j+0.5, n_rows-1-i+0.8, "Goal", ha='center', va='center', fontsize=12, color='black', fontweight='bold')

                if policy and state_idx in policy:
                    action = policy[state_idx]
                    dx, dy = 0, 0
                    if action == 'up': dx, dy = 0.0, 0.2
                    elif action == 'down': dx, dy = 0, -0.2
                    elif action == 'left': dx, dy = -0.2, 0
                    elif action == 'right': dx, dy = 0.2, 0
                    arrow = Arrow(j+0.5, n_rows-1-i+0.5, dx, dy, width=0.3, color='red')
                    ax.add_patch(arrow)

    ax.set_title("Grid World MDP: Bottleneck States and Policy")
    ax.set_xticks([])
    ax.set_yticks([])
    
    legend_elements = [
        plt.Rectangle((0,0),1,1,facecolor='lightgreen',edgecolor='black',label='Traversed Bottleneck'),
        plt.Rectangle((0,0),1,1,facecolor='orange',edgecolor='black',label='Non-traversed Human Bottleneck'),
        plt.Rectangle((0,0),1,1,facecolor='gold',edgecolor='black',label='Goal'),
        plt.Rectangle((0,0),1,1,facecolor=cmap(0.5),edgecolor='black',label='Normal'),
        plt.Rectangle((0,0),1,1,facecolor='black',edgecolor='black',label='Wall'),
        plt.Rectangle((0,0),1,1,facecolor='darkgray',edgecolor='black',label='Robot Wall'),
        plt.Rectangle((0,0),1,1,facecolor='purple',edgecolor='black',label='Human Wall'),
        Arrow(0,0,1,0,width=0.3,color='red',label='Policy Action')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

    plt.tight_layout()
    plt.show()

def visualize_grid_with_values(grid, V, bottlenecks, initial_state, goal_state, policy, human_walls, robot_walls, traversed_bottlenecks, non_traversed_bottlenecks):
    n_rows = len(grid)
    n_cols = len(grid[0])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect('equal')

    cmap = LinearSegmentedColormap.from_list("RedGreen", ["red", "white", "green"])
    norm = plt.Normalize(vmin=min(V), vmax=max(V))

    for i in range(n_rows):
        for j in range(n_cols):
            state = grid[i][j]
            if state == 0:
                rect = Rectangle((j, n_rows-1-i), 1, 1, facecolor='black', edgecolor='black')
                ax.add_patch(rect)
            else:
                state_idx = state - 1
                if state_idx == goal_state:
                    color = 'gold'
                elif state_idx in traversed_bottlenecks:
                    color = 'lightgreen'
                elif state_idx in non_traversed_bottlenecks:
                    color = 'orange'
                elif state_idx in human_walls:
                    color = 'purple'
                elif state_idx in robot_walls:
                    color = 'darkgray'
                else:
                    color = cmap(norm(V[state_idx]))
                rect = Rectangle((j, n_rows-1-i), 1, 1, facecolor=color, edgecolor='black')
                ax.add_patch(rect)
                
                ax.text(j+0.5, n_rows-1-i+0.5, f"S{state}\n{V[state_idx]:.2f}", ha='center', va='center', fontsize=8, color='black')

                if state_idx == initial_state:
                    ax.text(j+0.5, n_rows-1-i+0.8, "I", ha='center', va='center', fontsize=10, color='blue', fontweight='bold')
                
                if state_idx == goal_state:
                    ax.text(j+0.5, n_rows-1-i+0.8, "G", ha='center', va='center', fontsize=10, color='black', fontweight='bold')
                
                if state_idx in bottlenecks:
                    ax.text(j+0.5, n_rows-1-i+0.2, "B", ha='center', va='center', fontsize=10, color='red', fontweight='bold')

                if policy and state_idx in policy:
                    action = policy[state_idx]
                    dx, dy = 0, 0
                    if action == 'up': dx, dy = 0.0, 0.2
                    elif action == 'down': dx, dy = 0, -0.2
                    elif action == 'left': dx, dy = -0.2, 0
                    elif action == 'right': dx, dy = 0.2, 0
                    arrow = Arrow(j+0.5, n_rows-1-i+0.5, dx, dy, width=0.3, color='black')
                    ax.add_patch(arrow)

    ax.set_title("Gridworld MDP: State Values and Policy")
    ax.set_xticks([])
    ax.set_yticks([])
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('State Value')

    legend_elements = [
        plt.Rectangle((0,0),1,1,facecolor='black',edgecolor='black',label='Wall'),
        plt.Rectangle((0,0),1,1,facecolor='gold',edgecolor='black',label='Goal'),
        plt.Rectangle((0,0),1,1,facecolor='lightgreen',edgecolor='black',label='Traversed Bottleneck'),
        plt.Rectangle((0,0),1,1,facecolor='orange',edgecolor='black',label='Non-traversed Human Bottleneck'),
        plt.Rectangle((0,0),1,1,facecolor='purple',edgecolor='black',label='Human Wall'),
        plt.Rectangle((0,0),1,1,facecolor='darkgray',edgecolor='black',label='Robot Wall'),
        plt.Line2D([0], [0], marker='o', color='w', label='Bottleneck', markerfacecolor='r', markersize=10),
        Arrow(0,0,1,0,width=0.3,color='black',label='Policy Action')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

    plt.tight_layout()
    plt.show()
def visualize_grid_with_values(grid, V, bottlenecks, initial_state, goal_state, policy, human_walls, robot_walls, traversed_bottlenecks, non_traversed_bottlenecks):
    n_rows = len(grid)
    n_cols = len(grid[0])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect('equal')

    cmap = LinearSegmentedColormap.from_list("RedGreen", ["red", "white", "green"])
    norm = plt.Normalize(vmin=min(V), vmax=max(V))

    for i in range(n_rows):
        for j in range(n_cols):
            state = grid[i][j]
            if state == 0:
                rect = Rectangle((j, n_rows-1-i), 1, 1, facecolor='black', edgecolor='black')
                ax.add_patch(rect)
            else:
                state_idx = state - 1
                if state_idx == goal_state:
                    color = 'gold'
                elif state_idx in traversed_bottlenecks:
                    color = 'lightgreen'
                elif state_idx in non_traversed_bottlenecks:
                    color = 'orange'
                elif state_idx in human_walls:
                    color = 'purple'
                elif (i, j) in robot_walls:
                    color = 'darkgray'
                else:
                    color = cmap(norm(V[state_idx]))
                rect = Rectangle((j, n_rows-1-i), 1, 1, facecolor=color, edgecolor='black')
                ax.add_patch(rect)
                
                ax.text(j+0.5, n_rows-1-i+0.5, f"S{state}\n{V[state_idx]:.2f}", ha='center', va='center', fontsize=8, color='black')

                if state_idx == initial_state:
                    ax.text(j+0.5, n_rows-1-i+0.8, "I", ha='center', va='center', fontsize=10, color='blue', fontweight='bold')
                
                if state_idx == goal_state:
                    ax.text(j+0.5, n_rows-1-i+0.8, "G", ha='center', va='center', fontsize=10, color='black', fontweight='bold')
                
                if state_idx in bottlenecks:
                    ax.text(j+0.5, n_rows-1-i+0.2, "B", ha='center', va='center', fontsize=10, color='red', fontweight='bold')

                if policy and state_idx in policy:
                    action = policy[state_idx]
                    dx, dy = 0, 0
                    if action == 'up': dx, dy = 0.0, 0.2
                    elif action == 'down': dx, dy = 0, -0.2
                    elif action == 'left': dx, dy = -0.2, 0
                    elif action == 'right': dx, dy = 0.2, 0
                    arrow = Arrow(j+0.5, n_rows-1-i+0.5, dx, dy, width=0.3, color='black')
                    ax.add_patch(arrow)

    ax.set_title("Grid World MDP: State Values and Policy")
    ax.set_xticks([])
    ax.set_yticks([])
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('State Value')

    legend_elements = [
        plt.Rectangle((0,0),1,1,facecolor='black',edgecolor='black',label='Wall'),
        plt.Rectangle((0,0),1,1,facecolor='gold',edgecolor='black',label='Goal'),
        plt.Rectangle((0,0),1,1,facecolor='lightgreen',edgecolor='black',label='Traversed Bottleneck'),
        plt.Rectangle((0,0),1,1,facecolor='orange',edgecolor='black',label='Non-traversed Human Bottleneck'),
        plt.Rectangle((0,0),1,1,facecolor='purple',edgecolor='black',label='Human Wall'),
        plt.Rectangle((0,0),1,1,facecolor='darkgray',edgecolor='black',label='Robot-only Wall'),
        plt.Line2D([0], [0], marker='o', color='w', label='Bottleneck', markerfacecolor='r', markersize=10),
        Arrow(0,0,1,0,width=0.3,color='black',label='Policy Action')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

    plt.tight_layout()
    plt.show()