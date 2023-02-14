"""
Quoridor Environment Benchmark
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import time

from fights.base import BaseAgent
from pre import pre_env
from new import new_env

class RandomAgent(BaseAgent):
    env_id = ("quoridor", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _get_all_actions(self, state: pre_env.QuoridorState):
        actions = []
        for action_type in [0, 1, 2]:
            for coordinate_x in range(pre_env.QuoridorEnv.board_size):
                for coordinate_y in range(pre_env.QuoridorEnv.board_size):
                    action = [action_type, coordinate_x, coordinate_y]
                    try:
                        pre_env.QuoridorEnv().step(state, self.agent_id, action)
                    except:
                        ...
                    else:
                        actions.append(action)
        return actions

    def __call__(self, state: pre_env.QuoridorState) -> pre_env.QuoridorAction:
        actions = self._get_all_actions(state)
        return self._rng.choice(actions)

class FasterAgent(BaseAgent):
    env_id = ("quoridor", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def __call__(self, state: new_env.QuoridorState) -> new_env.QuoridorAction:
        legal_actions_np = new_env.QuoridorEnv().legal_actions(state, self.agent_id)
        return self._rng.choice(np.argwhere(legal_actions_np == 1))

def run_original():
    assert pre_env.QuoridorEnv.env_id == RandomAgent.env_id
    start = time.time()

    for game in range(10):

        print(game)

        state = pre_env.QuoridorEnv().initialize_state()
        agents = [RandomAgent(0, game), RandomAgent(1, game)]

        while not state.done:

            for agent in agents:

                action = agent(state)
                state = pre_env.QuoridorEnv().step(state, agent.agent_id, action)

                if state.done:
                    break

    end = time.time()
    print(f"{end - start} sec")

def run_faster():
    assert new_env.QuoridorEnv.env_id == RandomAgent.env_id
    start = time.time()

    for game in range(10):

        state = new_env.QuoridorEnv().initialize_state()
        agents = [FasterAgent(0, game), FasterAgent(1, game)]

        while not state.done:

            for agent in agents:

                action = agent(state)
                state = new_env.QuoridorEnv().step(state, agent.agent_id, action)

                if state.done:
                    break

    end = time.time()
    print(f"{end - start} sec")

if __name__ == "__main__":
    
    run_original()
    run_faster()
    
    