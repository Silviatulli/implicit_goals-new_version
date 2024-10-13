from typing import List, Tuple, Any, Set
from BottleneckCheckMDP import BottleneckMDP
from Utils import vectorized_value_iteration, get_policy, powerset
import copy

class QueryMDP:
    def __init__(self, robot_mdp: Any, bottlenecks: List[Any], achievable_subsets: List[Any]):
        self.robot_mdp = robot_mdp
        self.achievable_subsets = [set([robot_mdp.get_state_hash(state) for state in achievable_subset])
                                   for achievable_subset in achievable_subsets]
        self.union_achievable_subsets = set().union(*self.achievable_subsets)
        self.bottleneck_hash = set([robot_mdp.get_state_hash(state) for state in bottlenecks])
        self.unachievable_bottlenecks = self.bottleneck_hash - self.union_achievable_subsets

        self.bottleneck_hash_map = {robot_mdp.get_state_hash(state): state for state in bottlenecks}
        self.create_state_space()
        self.create_action_space()
        self.start_state = (set(), set())
        self.discount = robot_mdp.discount

    def create_state_space(self):
        # Contains two parts human subgoals and non human subgoals
        self.state_space = []
        for subgoals in powerset(self.bottleneck_hash):
            subgoal_set = set([i for i in subgoals])
            remaining_bottle_necks = self.bottleneck_hash - subgoal_set
            for non_subgoals in powerset(remaining_bottle_necks):
                non_subgoal_set = set([i for i in non_subgoals])
                self.state_space.append((subgoal_set, non_subgoal_set))

    def create_action_space(self):
        self.action_space = ['Query_'+state for state in self.bottleneck_hash]
    def get_state_space(self):
        return self.state_space

    def get_actions(self):
        return self.action_space

    def check_terminal_state(self, state: Tuple[Set[Any], Set[Any]]) -> bool:
        subgoal, not_subgoal = state
        possible_subgoals = self.bottleneck_hash - not_subgoal
        if len(self.unachievable_bottlenecks & subgoal) > 0:
            #print("Reached an unachievable state", state)
            return True
        # All possible subgoals can be achieved
        for achievable_subset in self.achievable_subsets:
            if achievable_subset.issuperset(possible_subgoals):
                # print("Reached an achievable state", state)
                # print("Possibe subgoals: ", possible_subgoals)
                # print("Achievable subset: ", achievable_subset)
                # exit(0)
                return True
        return False
    def get_transition_probability(self, state: Tuple[Set[Any], Set[Any]], action: str, next_state: Tuple[Set[Any], Set[Any]]) -> float:
        subgoal, not_subgoal = state
        next_subgoal, next_non_subgoal = next_state
        query_state = action.split('_')[-1]

        # Check for absorber state conditions
        if self.check_terminal_state(state):
            if state == next_state:
                return 1
            else:
                return 0

        if query_state in subgoal or query_state in not_subgoal:
            if state == next_state:
                return 1
            else:
                return 0

        is_subgoal_state = copy.deepcopy(state)
        is_subgoal_state[0].add(query_state)
        not_subgoal_state = copy.deepcopy(state)
        not_subgoal_state[1].add(query_state)
        possible_next_states = [is_subgoal_state, not_subgoal_state]
        if next_state not in possible_next_states:
            return 0
        return 0.5

    def get_init_state(self):
        return self.start_state

    def get_state_hash(self, state: Tuple[Set[Any], Set[Any]]) -> str:
        subgoal, not_subgoal = state
        return str(subgoal) + '-' + str(not_subgoal)

    def get_reward(self, state: Tuple[Set[Any], Set[Any]], action: str, next_state: Tuple[Set[Any], Set[Any]]) -> float:
        if self.check_terminal_state(next_state):
            #print("Reached terminal state", next_state)
            #exit(0)
            return 1000
        return -1


def simulate_policy(query_mdp, true_bottlnecks: List[Any], query_threshold=1000) -> int:
    true_bottlneck_hash = set([query_mdp.robot_mdp.get_state_hash(state) for state in true_bottlnecks])
    V = vectorized_value_iteration(query_mdp)
    #for s in M.state_space:
    #    print(s, V[M.get_state_hash(s)])
    #exit(0)
    policy = get_policy(query_mdp, V)
    #print("Policy: ", policy)
    #exit(0)
    start_state = query_mdp.get_init_state()

    count = 0
    current_state = start_state

    while count < query_threshold:
        count += 1
        action = policy[query_mdp.get_state_hash(current_state)]
        print("Querying action: ", action)
        print("Current state: ", current_state)
        query_state = action.split('_')[-1]
        if query_state in true_bottlneck_hash:
            current_state[0].add(query_state)
        else:
            current_state[1].add(query_state)
        if query_mdp.check_terminal_state(current_state):
            print ("Reached terminal state and the number of queries are: ", str(count))
            return count

    print ("Did not reach terminal state and the number of queries are: ", str(count))
    return count

if __name__ == "__main__":
    from GridWorldClass import generate_and_visualize_gridworld
    M_R = generate_and_visualize_gridworld(size=5, start=(0,0), goal=(4,4), obstacles_percent=0.1, divide_rooms=True, model_type="Robot Model")
    test_bottleneck_states = [((1, 0), (), ()), ((4, 3), (), ()), ((1, 1), (), ())]
    achievable_subsets = [[((1, 0), (), ()), ((4, 3), (), ())], [((1, 0), (), ()), ((1, 1), (), ())], [((4, 3), (), ()), ((1, 1), (), ())]]
    true_bottleneck_states = [((1, 0), (), ()), ((4, 3), (), ())]

    M_R = QueryMDP(robot_mdp=M_R, bottlenecks=test_bottleneck_states, achievable_subsets=achievable_subsets)
    simulate_policy(M_R, true_bottleneck_states)