from determinized_mdp import DeterminizedMDP
from old.mdp_experiments import run_multiple_experiments

def main():
    n_rows, n_cols = 8, 8
    n_human_walls = 5
    n_robot_walls = 10 
    n_experiments = 3

    run_multiple_experiments(n_experiments, n_rows, n_cols, n_human_walls, n_robot_walls)

if __name__ == "__main__":
    main()