from typing import List, Tuple, Any, Set
from itertools import combinations
from Utils import vectorized_value_iteration, get_policy, sparse_value_iteration, get_sparse_policy
import time
class QueryMDP:
    def __init__(self, robot_mdp: Any, bottlenecks: List[Any], achievable_subsets: List[Any]):
        self.robot_mdp = robot_mdp
        self.achievable_subsets = [set(robot_mdp.get_state_hash(state) for state in achievable_subset)
                                   for achievable_subset in achievable_subsets]
        self.union_achievable_subsets = set().union(*self.achievable_subsets)

        print("Union Achievable Subsets:", self.union_achievable_subsets)

        self.bottleneck_hash = set(robot_mdp.get_state_hash(state) for state in bottlenecks)
        self.unachievable_bottlenecks = self.bottleneck_hash - self.union_achievable_subsets
        self.bottleneck_hash_map = {robot_mdp.get_state_hash(state): state for state in bottlenecks}

        self.state_space = []
        self.action_space = []
        
        self.create_state_space()
        print("State Space:", self.state_space)
        self.create_action_space()
        
        self.start_state = (frozenset(), frozenset())
        self.discount = robot_mdp.discount

    def create_state_space(self):
        self.state_space = []
        bottleneck_list = list(self.union_achievable_subsets)
        
        for i in range(len(bottleneck_list) + 1):
            for subgoal_combo in combinations(bottleneck_list, i):
                subgoal_set = frozenset(subgoal_combo)
                remaining_bottlenecks = self.union_achievable_subsets - set(subgoal_combo)
                
                for j in range(len(remaining_bottlenecks) + 1):
                    for non_subgoal_combo in combinations(remaining_bottlenecks, j):
                        non_subgoal_set = frozenset(non_subgoal_combo)
                        self.state_space.append((subgoal_set, non_subgoal_set))

    def create_action_space(self):
        self.action_space = ['Query_' + state for state in self.bottleneck_hash]

    def get_state_space(self):
        return self.state_space

    def get_actions(self):
        return self.action_space

    def check_terminal_state(self, state: Tuple[Set[Any], Set[Any]]) -> bool:
        subgoal, not_subgoal = map(frozenset, state)
        
        if not self.unachievable_bottlenecks.isdisjoint(subgoal):
            return True
            
        possible_subgoals = self.bottleneck_hash - not_subgoal
        
        return any(achievable_subset.issuperset(possible_subgoals) 
                  for achievable_subset in self.achievable_subsets)

    def get_transition_probability(self, state: Tuple[Set[Any], Set[Any]], action: str, 
                                 next_state: Tuple[Set[Any], Set[Any]]) -> float:
        subgoal, not_subgoal = map(frozenset, state)
        next_subgoal, next_non_subgoal = map(frozenset, next_state)
        query_state = action.split('_')[-1]

        if self.check_terminal_state(state):
            return 1.0 if state == next_state else 0.0

        if query_state in subgoal or query_state in not_subgoal:
            return 1.0 if state == next_state else 0.0

        possible_next_states = [
            (frozenset(subgoal | {query_state}), not_subgoal),
            (subgoal, frozenset(not_subgoal | {query_state}))
        ]

        return 0.5 if next_state in possible_next_states else 0.0

    def get_init_state(self):
        return self.start_state

    def get_state_hash(self, state: Tuple[Set[Any], Set[Any]]) -> str:
        subgoal, not_subgoal = map(frozenset, state)
        return f"{sorted(subgoal)}-{sorted(not_subgoal)}"

    def get_reward(self, state: Tuple[Set[Any], Set[Any]], action: str, 
                  next_state: Tuple[Set[Any], Set[Any]]) -> float:
        return 1000 if self.check_terminal_state(next_state) else -1

def simulate_policy(query_mdp: QueryMDP, true_bottlenecks: List[Any], query_threshold: int = 1000) -> int:
    true_bottleneck_hash = frozenset(query_mdp.robot_mdp.get_state_hash(state) 
                                   for state in true_bottlenecks)
    true_bottleneck_hash = frozenset(query_mdp.robot_mdp.get_state_hash(state) 
                                   for state in true_bottlenecks)
    start_value_iteration_time = time.time()
    V = sparse_value_iteration(query_mdp)
    print(f"Value iteration took {time.time() - start_value_iteration_time} seconds")
    start_policy_time = time.time()
    policy = get_sparse_policy(query_mdp, V)
    print(f"Policy generation took {time.time() - start_policy_time} seconds")
    current_state = list(query_mdp.get_init_state())
    
    for count in range(query_threshold):
        action = policy[query_mdp.get_state_hash(tuple(current_state))]
        query_state = action.split('_')[-1]
        
        # update state based on query result
        is_subgoal = query_state in true_bottleneck_hash
        if is_subgoal:
            current_state[0] = frozenset(current_state[0] | {query_state})
        else:
            current_state[1] = frozenset(current_state[1] | {query_state})
            
        if query_mdp.check_terminal_state(tuple(current_state)):
            print(f"reached terminal state after {count + 1} queries")
            return count + 1
            
    print(f"did not reach terminal state after {query_threshold} queries")
    return query_threshold


def simulate_policy_unachievable(query_mdp: QueryMDP, human_bottlenecks: List[Any], query_threshold: int = 1000) -> int:

    human_bottleneck_hash = frozenset(query_mdp.robot_mdp.get_state_hash(state) 
                                    for state in human_bottlenecks)
    query_count = 0
    
    # query each unachievable bottleneck to check if it's necessary for the human
    for unachievable in query_mdp.unachievable_bottlenecks:
        if query_count >= query_threshold:
            print(f"Query threshold {query_threshold} reached during unachievable bottleneck checks")
            return query_threshold
            
        # query human to see if this unachievable bottleneck is their subgoal
        is_human_subgoal = unachievable in human_bottleneck_hash
        query_count += 1
        
        if is_human_subgoal:
            # found an unachievable bottleneck that is necessary for human - implicit goal
            print(f"Found implicit goal: bottleneck that human needs but the robot can't reach")
            return query_count
    
    # if we get here, any unachievable bottlenecks weren't human subgoals
    # run normal policy to identify remaining bottlenecks
    start_value_iteration_time = time.time()
    V = sparse_value_iteration(query_mdp)
    print(f"Value iteration took {time.time() - start_value_iteration_time} seconds")
    
    start_policy_time = time.time()
    policy = get_sparse_policy(query_mdp, V)
    print(f"Policy generation took {time.time() - start_policy_time} seconds")
    
    current_state = list(query_mdp.get_init_state())
    
    while query_count < query_threshold:
        action = policy[query_mdp.get_state_hash(tuple(current_state))]
        query_state = action.split('_')[-1]
        
        # query if this state is a human subgoal
        is_subgoal = query_state in human_bottleneck_hash
        if is_subgoal:
            current_state[0] = frozenset(current_state[0] | {query_state})
        else:
            current_state[1] = frozenset(current_state[1] | {query_state})
        
        query_count += 1
            
        if query_mdp.check_terminal_state(tuple(current_state)):
            print(f"Identified all necessary bottlenecks after {query_count} queries")
            return query_count
            
    print(f"Did not complete bottleneck identification within {query_threshold} queries")
    return query_threshold

def simulate_policy_query_all(query_mdp: QueryMDP, human_bottlenecks: List[Any], query_threshold: int = 1000) -> int:
    """
    Simple baseline strategy that queries all bottleneck states sequentially.
    """
    human_bottleneck_hash = frozenset(query_mdp.robot_mdp.get_state_hash(state) 
                                    for state in human_bottlenecks)
    query_count = 0
    confirmed_subgoals = set()
    confirmed_non_subgoals = set()
    all_bottlenecks = query_mdp.bottleneck_hash
    
    for bottleneck in all_bottlenecks:
        if query_count >= query_threshold:
            return query_threshold
            
        query_count += 1
        is_subgoal = bottleneck in human_bottleneck_hash
        
        if is_subgoal:
            confirmed_subgoals.add(bottleneck)
        else:
            confirmed_non_subgoals.add(bottleneck)
            
        current_state = (frozenset(confirmed_subgoals), frozenset(confirmed_non_subgoals))
        if query_mdp.check_terminal_state(current_state):
            print(f"Found all bottlenecks after {query_count} queries")
            return query_count
    
    return query_threshold

def test_query_mdp(size=5, obstacles_percent=0.1):
    """Test function for QueryMDP."""
    from GridWorldClass import generate_and_visualize_gridworld
    
    print("generating test environment...")
    M_R = generate_and_visualize_gridworld(
        size=size,
        start=(0,0),
        goal=(size-1,size-1),
        obstacles_percent=obstacles_percent,
        divide_rooms=True,
        model_type="Robot Model"
    )

    test_case = {
        'bottlenecks': [((1, 0), (), ()), ((2, 2), (), ())],
        'achievable_subsets': [[((1, 0), (), ())]],
        'true_bottlenecks': [((1, 0), (), ())]
    }

    query_mdp = QueryMDP(
        robot_mdp=M_R,
        bottlenecks=test_case['bottlenecks'],
        achievable_subsets=test_case['achievable_subsets']
    )

    return simulate_policy_unachievable(query_mdp, test_case['true_bottlenecks'])

if __name__ == "__main__":
    test_query_mdp(size=5, obstacles_percent=0.1)
