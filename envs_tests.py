from Utils import ValueIteration, vectorized_value_iteration
from DeterminizedMDP import DeterminizedMDP
from envs import UnlockEnv, UnlockPickupEnv
import time



def test_unlock_env():
    # Create the UnlockEnv with custom initial state and goal states
    custom_init_state = ((0, 0), 2, 2)  # location, facing South, door locked
    custom_goal_states = [((3, 3), direction, 0) for direction in range(4)]  
    
    env = UnlockEnv(grid_size=5, max_steps=100, slip_prob=0.1, 
                    init_state=custom_init_state, goal_states=custom_goal_states)

    print("Initial state:", env.get_init_state())
    print("Goal states:", env.get_goal_states())

    # Test ValueIteration
    print("\nTesting ValueIteration...")
    V = vectorized_value_iteration(env)
    print("Value function:")
    for state in env.get_state_space()[:]:  
        print(f"State: {state}, Value: {V[env.get_state_hash(state)]}")

    # Test stochastic transitions
    print("\nTesting stochastic transitions...")
    init_state = env.get_init_state()
    for action in env.get_actions():
        print(f"\nAction: {action}")
        transitions = env.return_transition_probabilities(init_state, action)
        total_prob = sum(transitions.values())
        for next_state, prob in transitions.items():
            print(f"  Next state: {next_state}, Probability: {prob:.2f}")
        print(f"  Total probability: {total_prob:.2f}")

    # Test DeterminizedMDP
    print("\nTesting DeterminizedMDP...")
    try:
        det_mdp = DeterminizedMDP(env)
        print("Determinized actions:", det_mdp.get_actions())
        init_state = det_mdp.get_init_state()
        print("Initial state:", init_state)
        
        # Test a few transitions in the determinized MDP
        for action in det_mdp.get_actions()[:5]:  # Let's limit to first 5 actions for brevity
            for next_state in list(det_mdp.get_state_space())[:5]:  # And first 5 states
                prob = det_mdp.get_transition_probability(init_state, action, next_state)
                if prob > 0:
                    print(f"Action: {action}, Next state: {next_state}, Probability: {prob}")
        
        # Test reward function
        print("\nTesting reward function:")
        # Try to find a transition that leads to a goal state
        goal_state = det_mdp.goal_states[0]
        for action in det_mdp.get_actions():
            reward = det_mdp.get_reward(init_state, action, goal_state)
            if reward > 0:
                print(f"Reward for reaching goal state: Action: {action}, Reward: {reward}")
                break
        else:
            print("Couldn't find a direct transition to a goal state. Showing a sample non-goal transition:")
            sample_next_state = list(det_mdp.get_state_space())[1]  # Just take the second state as a sample
            reward = det_mdp.get_reward(init_state, det_mdp.get_actions()[0], sample_next_state)
            print(f"Sample reward for non-goal transition: {reward}")

    except Exception as e:
        print(f"Error occurred while testing DeterminizedMDP: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\nTest completed.")


    # print("\nTest completed.")
    # exit()
    # det_mdp = DeterminizedMDP(env)
    # print(det_mdp.get_actions())
    # # print(det_mdp.get_transition_probability(((0,0),), 'act_0', ((0,0),)))
    # # print(det_mdp.get_reward(((0,0),), 'act_0', ((0,0),)))
    # print(det_mdp.get_init_state())
    # init_state_hash = det_mdp.get_state_hash(det_mdp.get_init_state())
    # det_mdp.reward_func = det_mdp.bottleneck_reward
    # bottleneck_list = []
    # for state in det_mdp.get_state_space():
    #     det_mdp.bottleneck_state = det_mdp.get_state_hash(state)
    #     V = vectorized_value_iteration(det_mdp)
    #     if V[init_state_hash] <= 0:
    #         bottleneck_list.append(state)
    # print(bottleneck_list)




def test_unlock_pickup_env():
    # Create the UnlockPickupEnv
    env = UnlockPickupEnv(grid_size=2, max_steps=100, slip_prob=0.1)

    print("Initial state:", env.get_init_state())
    print("Goal states (first 5):", env.get_goal_states()[:5])

    # Test stochastic transitions
    print("\nTesting stochastic transitions...")
    init_state = env.get_init_state()
    for action in env.get_actions():
        print(f"\nAction: {action}")
        transitions = env.return_transition_probabilities(init_state, action)
        total_prob = sum(transitions.values())
        for next_state, prob in transitions.items():
            if prob > 0:  # Only print non-zero probability transitions
                print(f"  Next state: {next_state}, Probability: {prob:.6f}")
        print(f"  Total probability: {total_prob:.6f}")

    # Test ValueIteration
    print("\nTesting ValueIteration...")
    start_time = time.time()
    V = vectorized_value_iteration(env, epsilon=0.01, max_iterations=1000)
    end_time = time.time()
    print(f"ValueIteration took {end_time - start_time:.4f} seconds")
    print("Value function (first 5 states):")
    for state in list(env.get_state_space())[:5]:
        print(f"State: {state}, Value: {V[env.get_state_hash(state)]}")

    # Test DeterminizedMDP
    print("\nTesting DeterminizedMDP...")
    try:
        det_mdp = DeterminizedMDP(env)
        print("Determinized actions:", det_mdp.get_actions())
        init_state = det_mdp.get_init_state()
        print("Initial state:", init_state)
        
        # Test ValueIteration on DeterminizedMDP
        print("\nTesting ValueIteration on DeterminizedMDP...")
        start_time = time.time()
        V_det = vectorized_value_iteration(det_mdp, epsilon=0.01, max_iterations=1000)
        end_time = time.time()
        print(f"ValueIteration on DeterminizedMDP took {end_time - start_time:.4f} seconds")
        print("Value function (first 5 states):")
        for state in list(det_mdp.get_state_space())[:5]:
            print(f"State: {state}, Value: {V_det[det_mdp.get_state_hash(state)]}")

    except Exception as e:
        print(f"Error occurred while testing DeterminizedMDP: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\nTest completed.")




if __name__ == "__main__":
    # test_unlock_env()
    test_unlock_pickup_env()
