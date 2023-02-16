"""
Othello game example.
Prints board state to stdout with random agents by default.
"""

import re
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import colorama
import numpy as np
from colorama import Fore, Style

from fights.base import BaseAgent
from new import new_env

class RandomAgent(BaseAgent):
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

def fallback_to_ascii(s: str) -> str:
    try:
        s.encode(sys.stdout.encoding)
    except UnicodeEncodeError:
        s = re.sub("[┌┬┐├┼┤└┴┘╋]", "+", re.sub("[─━]", "-", re.sub("[│┃]", "|", s)))
    return s

def run():
    assert new_env.OthelloEnv.env_id == RandomAgent.env_id
    colorama.init()

    state = new_env.OthelloEnv().initialize_state()
    agents = [RandomAgent(0), RandomAgent(1)]

    print("\x1b[2J")

    it = 0
    while not state.done:

        print("\x1b[1;1H")
        print(fallback_to_ascii(str(state)))

        for agent in agents:

            action = agent(state)
            state = new_env.OthelloEnv().step(state, agent.agent_id, action)

            print("\x1b[1;1H")
            print(fallback_to_ascii(str(state)))

            a = input()

            if state.done:
                print(f"agent {np.argmax(state.reward)} won in {it} iters")
                break

        it += 1

if __name__ == "__main__":
    run()