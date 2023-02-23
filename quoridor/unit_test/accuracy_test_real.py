"""
New Quoridor Environment Accuracy Test with Old Quoridor Environment.
"""

import re
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import colorama

import numpy as np
import time
from colorama import Fore, Style

from fights.base import BaseAgent
from fights.envs import quoridor
from pre import pre_env

class TestAgent(BaseAgent):
    env_id = ("quoridor", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _get_all_actions_original(self, state: pre_env.QuoridorState):
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
    
    def _get_all_actions_faster(self, state: quoridor.QuoridorState):
        legal_actions_np = quoridor.QuoridorEnv().legal_actions(state, self.agent_id)
        actions = []
        for action_type in [0, 1, 2]:
            for coordinate_x in range(quoridor.QuoridorEnv.board_size):
                for coordinate_y in range(quoridor.QuoridorEnv.board_size):
                    action = [action_type, coordinate_x, coordinate_y]
                    if legal_actions_np[action_type, coordinate_x, coordinate_y]:
                        actions.append(action)
        return actions

    def __call__(self, original_state: pre_env.QuoridorState, faster_state: quoridor.QuoridorState) -> pre_env.QuoridorAction:
        original_actions = self._get_all_actions_original(original_state)
        faster_actions = self._get_all_actions_faster(faster_state)
        if not original_actions == faster_actions:
            print(original_actions)
            print(faster_actions)
            raise ValueError(f"original actions and faster actions are different!!")
        return self._rng.choice(original_actions)

def run():
    assert pre_env.QuoridorEnv.env_id == TestAgent.env_id
    start = time.time()

    for game in range(100):

        print(game)

        original_state = pre_env.QuoridorEnv().initialize_state()
        faster_state = quoridor.QuoridorEnv().initialize_state()
        agents = [TestAgent(0, game), TestAgent(1, game)]

        while not original_state.done:

            for agent in agents:

                action = agent(original_state, faster_state)
                original_state = pre_env.QuoridorEnv().step(original_state, agent.agent_id, action)
                faster_state = quoridor.QuoridorEnv().step(faster_state, agent.agent_id, action)

                if original_state.done:
                    break

    end = time.time()
    print(f"{end - start} sec")

if __name__ == "__main__":
    run()