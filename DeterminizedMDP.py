from MDP import MDP
from collections import defaultdict
from GridWorldClass import GridWorld
class DeterminizedMDP(MDP):
    def __init__(self, mdp):
        self.mdp = mdp
        self.state_space = mdp.get_state_space()
        self.original_actions = mdp.get_actions()
        self.actions = []
        self.transition_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.create_determinized_transition_matrix()
        self.init_state = mdp.get_init_state()
        self.reward_func = mdp.reward_func
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


if __name__ == "__main__":
    grid = GridWorld(start=(0,0), goal=(4, 4))
    det_mdp = DeterminizedMDP(grid)
    print(det_mdp.get_actions())
    # print(det_mdp.get_transition_probability(((0,0),), 'act_0', ((0,0),)))
    # print(det_mdp.get_reward(((0,0),), 'act_0', ((0,0),)))
    print(det_mdp.get_init_state())
    print(det_mdp.get_state_space())
    init_state_hash = det_mdp.get_state_hash(det_mdp.get_init_state())
    next_state_hash = det_mdp.get_state_hash([(0,0),(),()])
    print(det_mdp.transition_dict[init_state_hash]['act_0'][next_state_hash])
    # print(det_mdp.get_transition_probability(((0,0),), 'act_0', ((0,1),)))
    # print(det_mdp.get_reward(((0,0),), 'act_0', ((0,1),)))
    # print(det_mdp.get_transition_probability(((0,0),), 'act_0', ((1,0),)))
    # print(det_mdp.get_reward(((0,0),), 'act_0', ((1,0),)))
    # print(det_mdp.get_transition_probability(((0,0),), 'act_0', ((1,1),)))
    # print(det_mdp.get_reward(((0,0),), 'act_0', ((1,1),)))
    # print(det_mdp.get_transition_probability(((0,0),), 'act_0', ((1,1),)))
    # print(det_mdp.get_reward(((0,0),), 'act_0', ((1,1),)))
