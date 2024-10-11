from typing import List, Tuple, Any, Set
from BottleneckCheckMDP import BottleneckMDP
from Utils import vectorized_value_iteration, get_policy

class QueryMDP:
    def __init__(self, robot_mdp: Any, achievable_bottlenecks: List[Any], non_achievable_bottlenecks: List[Any]):
        self.robot_mdp = robot_mdp
        self.achievable_bottlenecks = self.flatten_bottlenecks(achievable_bottlenecks)
        self.non_achievable_bottlenecks = self.flatten_bottlenecks(non_achievable_bottlenecks)
        self.necessary_bottlenecks: List[Tuple[int, int]] = []
        self.query_cost = -1

    def flatten_bottlenecks(self, bottlenecks: List[Any]) -> List[Tuple[int, int]]:
        # this function handles nested structures, be careful when modifying
        flattened = []
        for sublist in bottlenecks:
            if isinstance(sublist, list):
                for item in sublist:
                    if isinstance(item, tuple) and len(item) == 3:
                        coords = item[0]
                        if isinstance(coords, tuple) and len(coords) == 2:
                            flattened.append(coords)
            elif isinstance(sublist, tuple) and len(sublist) == 3:
                coords = sublist[0]
                if isinstance(coords, tuple) and len(coords) == 2:
                    flattened.append(coords)
        return list(set(flattened))  # remove duplicates

    def query_human(self):
        print("initiating human query process...")
        print(f"non-achievable bottlenecks: {self.non_achievable_bottlenecks}")
        for bottleneck in self.non_achievable_bottlenecks:
            if bottleneck not in self.achievable_bottlenecks:
                response = input(f"is the bottleneck at {bottleneck} necessary? (y/n): ")
                if response.lower() == 'y':
                    self.necessary_bottlenecks.append(bottleneck)
                    print(f"bottleneck {bottleneck} marked as necessary.")
                else:
                    print(f"bottleneck {bottleneck} not marked as necessary.")
        print("human query process completed.")

    def compute_policy(self):
        print("attempting to compute policy...")
        try:
            bottleneck_mdp = BottleneckMDP(self.robot_mdp, self.achievable_bottlenecks)
            V = vectorized_value_iteration(bottleneck_mdp)
            policy = get_policy(bottleneck_mdp, V)
            return policy, bottleneck_mdp
        except MemoryError as e:
            print(f"memory error during policy computation: {str(e)}")
            return None, None
        except Exception as e:
            print(f"error computing policy: {str(e)}")
            return None, None

    def explain_unreachable_bottlenecks(self) -> List[str]:
        explanations = []
        for bottleneck in self.necessary_bottlenecks:
            display_coords = (bottleneck[0] + 1, 5 - bottleneck[1])
            explanation = f"cannot reach necessary bottleneck at {display_coords}. "
            try:
                if self.robot_mdp.is_obstacle(bottleneck):
                    explanation += "this location is an obstacle in the robot's model."
                elif not self.robot_mdp.is_reachable(bottleneck):
                    explanation += "there is no safe path to this location in the robot's model."
                else:
                    explanation += "the reason for unreachability is unknown."
            except AttributeError:
                explanation += "unable to determine the nature of this location in the robot's model."
            explanations.append(explanation)
        return explanations

    def run(self):
        print("running querymdp...")
        print(f"achievable bottlenecks: {self.achievable_bottlenecks}")
        print(f"non-achievable bottlenecks: {self.non_achievable_bottlenecks}")
        self.query_human()
        
        print("computing policy...")
        policy, bottleneck_mdp = self.compute_policy()
        
        if policy is None or bottleneck_mdp is None:
            print("policy computation failed. unable to provide state-action pairs or explanations.")
            return None, None, []

        print("policy computed. here are some example state-action pairs:")
        try:
            for i, state in enumerate(bottleneck_mdp.get_state_space()):
                if i < 5:  # print first 5 state-action pairs as examples
                    action = policy[bottleneck_mdp.get_state_hash(state)]
                    print(f"state: {state}, action: {action}")
                else:
                    break
        except Exception as e:
            print(f"error processing state-action pairs: {str(e)}")

        explanations = self.explain_unreachable_bottlenecks()
        if explanations:
            print("explanations for unreachable necessary bottlenecks:")
            for explanation in explanations:
                print(explanation)

        return policy, bottleneck_mdp, explanations
