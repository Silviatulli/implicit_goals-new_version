from MDP import MDP
from collections import defaultdict
from GridWorldClass import GridWorld
from Utils import ValueIteration
class DeterminizedMDP(MDP):
    def __init__(self, mdp):
        self.mdp = mdp
        self.state_space = mdp.get_state_space()
        self.original_actions = mdp.get_actions()
        self.actions = []
        self.transition_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.create_determinized_transition_matrix()
        self.init_state = self.get_state_hash(mdp.get_init_state())
        self.reward_func = mdp.reward_func
        self.original_reward_func = mdp.reward_func
        self.goal_states = [self.get_state_hash(state) for state in mdp.get_goal_states()]
        self.discount = mdp.discount
    def create_determinized_transition_matrix(self):
        all_states = self.state_space
        action_prefix = 'act_'
        act_count = 0
        for state in all_states:
            current_act_cnt = 0
            state_hash = self.get_state_hash(state)
            for act in self.original_actions:
                for next_state in all_states:
                    next_state_hash = self.get_state_hash(next_state)
                    if self.mdp.get_transition_probability(state, act, next_state) > 0:
                        self.transition_dict[state_hash][action_prefix + str(current_act_cnt)][next_state_hash] = 1
                        current_act_cnt += 1
            if current_act_cnt > act_count:
                act_count = current_act_cnt
        self.actions = [action_prefix + str(i) for i in range(act_count)]


    def get_state_hash(self, state):
        # TODO: Assumption the function is idempotent
        return str(state)
    def get_actions(self):
        return self.actions

    def get_transition_probability(self, state, action, state_prime):
        state_hash = self.get_state_hash(state)
        state_prime_hash = self.get_state_hash(state_prime)
        return self.transition_dict[state_hash][action][state_prime_hash]

    def get_init_state(self):
        return self.init_state

    def get_reward(self, state, action, next_state):
        return self.reward_func(state, action, next_state)

    def get_state_space(self):
        return self.state_space

    def bottleneck_reward(self, state, action, next_state):
        if self.get_state_hash(next_state) == self.get_state_hash(self.bottleneck_state):
            return -1000
        if self.get_state_hash(next_state) in self.goal_states and self.get_state_hash(state) not in self.goal_states:
            return 1
        return 0


if __name__ == "__main__":
    grid = GridWorld(start=(0,0), goal=(4, 4), divide_rooms=True)
    grid.visualize()
    det_mdp = DeterminizedMDP(grid)
    print(det_mdp.get_actions())
    # print(det_mdp.get_transition_probability(((0,0),), 'act_0', ((0,0),)))
    # print(det_mdp.get_reward(((0,0),), 'act_0', ((0,0),)))
    print(det_mdp.get_init_state())
    init_state_hash = det_mdp.get_state_hash(det_mdp.get_init_state())
    det_mdp.reward_func = det_mdp.bottleneck_reward
    bottleneck_list = []
    for state in det_mdp.get_state_space():
        det_mdp.bottleneck_state = det_mdp.get_state_hash(state)
        V = ValueIteration(det_mdp)
        if V[init_state_hash] <= 0:
            bottleneck_list.append(state)
    print(bottleneck_list)
