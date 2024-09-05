import numpy as np
def find_bottleneck_states(mdp, S, A, s0, G, epsilon=1e-6):
    bottlenecks = []
    n_states = len(S)
    is_mdp = hasattr(mdp, 'R')

    original_rewards = mdp.R.copy() if is_mdp else np.zeros(n_states)

    for s in S:
        if s != s0 and s != G:
            # set rewards: large negative for the tested state, small positive for the goal
            R = np.zeros(n_states)
            R[s] = -10.0
            R[G] = 0.1

            # run value iteration
            V = np.zeros(n_states)
            gamma = 0.95
            max_iterations = 500
            for iteration in range(max_iterations):
                delta = 0
                for state in S:
                    if state == G:
                        continue
                    v = V[state]
                    if is_mdp:
                        if state in mdp.P:
                            V[state] = R[state] + gamma * max(
                                sum(prob * V[next_state] for next_state, prob in mdp.P[state][a].items())
                                for a in mdp.P[state]
                            )
                    else:
                        next_states = np.argmax(mdp[:, state], axis=0)
                        V[state] = R[state] + gamma * max(V[next_state] for next_state in next_states)
                    delta = max(delta, abs(v - V[state]))
                if delta < epsilon:
                    break

            # check if the initial state's value is negative
            if V[s0] < 0:
                bottlenecks.append(s)

    # restore original rewards if it's an MDP object
    if is_mdp:
        mdp.R = original_rewards

    return bottlenecks



def find_max_bottleneck_policy(mdp, bottlenecks):
    def find_policy_for_bottlenecks(mdp, target_bottlenecks):
        optimal_policy = mdp.get_optimal_policy()
        covered_bottlenecks = check_policy_coverage(mdp, optimal_policy, target_bottlenecks)
        
        return optimal_policy, covered_bottlenecks

    def check_policy_coverage(mdp, policy, target_bottlenecks):
        covered_bottlenecks = set()
        current = mdp.s0
        visited = set()

        while current not in visited and current != mdp.G:
            visited.add(current)
            if current in target_bottlenecks:
                covered_bottlenecks.add(current)
            
            if current in policy:
                action = policy[current]
                if current in mdp.P and action in mdp.P[current]:
                    # for stochastic transitions, choose the most likely next state
                    next_state_probs = mdp.P[current][action]
                    current = max(next_state_probs, key=next_state_probs.get)
                else:
                    break  # no valid transition, end the loop
            else:
                break  # no policy for current state, end the loop

        return covered_bottlenecks

    policy, covered = find_policy_for_bottlenecks(mdp, bottlenecks)
    return policy, list(covered), len(covered)








