class MDP(object):
    def __init__(self):
        pass

    def get_state_space(self):
        return self.state_space

    def get_actions(self):
        return self.actions

    def get_transition_probability(self, state, action, next_state):
        return 0

    def get_init_state(self):
        return self.init_state

    def get_state_hash(self, state):
        return str(state)

    def get_goal_states(self):
        return []