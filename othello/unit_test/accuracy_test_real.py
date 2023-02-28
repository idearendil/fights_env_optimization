"""
New Othello Environment Accuracy Test with Old Othello Environment.
"""

import re
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import time

from fights.base import BaseAgent
from fights.envs import othello
from pre import pre_env

class TestAgent(BaseAgent):
    env_id = ("othello", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _get_all_actions_original(self, state: pre_env.OthelloState):
        actions = []
        for coordinate_x in range(pre_env.OthelloEnv.board_size):
            for coordinate_y in range(pre_env.OthelloEnv.board_size):
                action = [coordinate_x, coordinate_y]
                if state.legal_actions[self.agent_id][coordinate_x][coordinate_y]:
                    actions.append(action)
        return actions
    
    def _get_all_actions_faster(self, state: othello.OthelloState):
        actions = []
        for coordinate_x in range(othello.OthelloEnv.board_size):
            for coordinate_y in range(othello.OthelloEnv.board_size):
                action = [coordinate_x, coordinate_y]
                if state.legal_actions[self.agent_id][coordinate_x][coordinate_y]:
                    actions.append(action)
        return actions

    def __call__(self, original_state: pre_env.OthelloState, faster_state: othello.OthelloState) -> pre_env.OthelloAction:
        original_actions = self._get_all_actions_original(original_state)
        faster_actions = self._get_all_actions_faster(faster_state)
        if not original_actions == faster_actions:
            print(original_state.board)
            print(faster_state.board)
            print(original_actions)
            print(faster_actions)
            raise ValueError(f"original actions and faster actions are different!!")
        return self._rng.choice(original_actions)

def run():
    assert pre_env.OthelloEnv.env_id == TestAgent.env_id
    start = time.time()

    for game in range(100):

        print(game)

        original_state = pre_env.OthelloEnv().initialize_state()
        faster_state = othello.OthelloEnv().initialize_state()
        agents = [TestAgent(0, game), TestAgent(1, game)]

        while not original_state.done:

            for agent in agents:

                action = agent(original_state, faster_state)
                original_state = pre_env.OthelloEnv().step(original_state, agent.agent_id, action)
                faster_state = othello.OthelloEnv().step(faster_state, agent.agent_id, action)

                if original_state.done:
                    break

    end = time.time()
    print(f"{end - start} sec")

if __name__ == "__main__":
    run()