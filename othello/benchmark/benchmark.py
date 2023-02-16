"""
Othello Environment Benchmark
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
    env_id = ("othello", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _get_all_actions(self, state: pre_env.OthelloState):
        actions = []
        for coordinate_x in range(pre_env.OthelloEnv.board_size):
            for coordinate_y in range(pre_env.OthelloEnv.board_size):
                action = [coordinate_x, coordinate_y]
                if state.legal_actions[self.agent_id][coordinate_x][coordinate_y]:
                    actions.append(action)
        return actions

    def __call__(self, state: pre_env.OthelloState) -> pre_env.OthelloAction:
        actions = self._get_all_actions(state)
        return self._rng.choice(actions)

class FasterAgent(BaseAgent):
    env_id = ("othello", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _get_all_actions(self, state: new_env.OthelloState):
        actions = []
        for coordinate_x in range(new_env.OthelloEnv.board_size):
            for coordinate_y in range(new_env.OthelloEnv.board_size):
                action = [coordinate_x, coordinate_y]
                if state.legal_actions[self.agent_id][coordinate_x][coordinate_y]:
                    actions.append(action)
        return actions

    def __call__(self, state: new_env.OthelloState) -> new_env.OthelloAction:
        actions = self._get_all_actions(state)
        return self._rng.choice(actions)

def run_original():
    assert pre_env.OthelloEnv.env_id == RandomAgent.env_id
    start = time.time()

    for game in range(2000):

        state = pre_env.OthelloEnv().initialize_state()
        agents = [RandomAgent(0, game), RandomAgent(1, game)]

        while not state.done:

            for agent in agents:

                action = agent(state)
                state = pre_env.OthelloEnv().step(state, agent.agent_id, action)

                if state.done:
                    break

    end = time.time()
    print(f"{end - start} sec")

def run_faster():
    assert new_env.OthelloEnv.env_id == RandomAgent.env_id
    start = time.time()

    for game in range(2000):

        state = new_env.OthelloEnv().initialize_state()
        agents = [FasterAgent(0, game), FasterAgent(1, game)]

        while not state.done:

            for agent in agents:

                action = agent(state)
                state = new_env.OthelloEnv().step(state, agent.agent_id, action)

                if state.done:
                    break

    end = time.time()
    print(f"{end - start} sec")

if __name__ == "__main__":
    
    run_original()
    run_faster()
    
    