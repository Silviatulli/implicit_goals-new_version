from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def ValueIteration(mdp, epsilon=0.001):
    V = {mdp.get_state_hash(s): 0 for s in mdp.get_state_space()}
    print(mdp.discount)
    while True:
        delta = 0
        for s in mdp.get_state_space():
            s_hash = mdp.get_state_hash(s)
            v = V[s_hash]
            #print([([mdp.get_transition_probability(s, a, s_prime) for s_prime in mdp.get_state_space()]) for a in mdp.get_actions()])
            for  a in mdp.get_actions():
                #print("act",a)
                value_for_a = 0
                #prob_list = []
                for s_prime in mdp.get_state_space():
                    value_for_a += mdp.get_transition_probability(s, a, s_prime) * (mdp.get_reward(s, a, s_prime) + V[mdp.get_state_hash(s_prime)])
                    #prob_list.append(mdp.get_transition_probability(s, a, s_prime))
                #print(s[0],a, mdp.discount * value_for_a)
                #print(s,a,prob_list)
            V[s_hash] = max([mdp.discount * sum([mdp.get_transition_probability(s, a, s_prime) * (mdp.get_reward(s, a, s_prime) + V[mdp.get_state_hash(s_prime)]) for s_prime in mdp.get_state_space()]) for a in mdp.get_actions()])
            delta = max(delta, abs(v - V[s_hash]))
        #print(delta)
        if delta < epsilon:
            break
    return V