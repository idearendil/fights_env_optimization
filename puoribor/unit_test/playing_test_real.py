# coding=latin-1

"""
Puoribor game example.
Prints board state to stdout with random agents by default.
Run `python puoribor.py -h` for more information.
"""



import argparse
import re
import sys
import time
from typing import Optional

import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import colorama
import numpy as np
from colorama import Fore, Style

from fights.base import BaseAgent

from pre import pre_env
from fights.envs import puoribor

import random

class ManualAgent(BaseAgent):
    env_id = ("puoribor", 3)  # type: ignore

    def __init__(self, agent_id: int) -> None:
        self.agent_id = agent_id  # type: ignore

    def __call__(self, state: puoribor.PuoriborState) -> puoribor.PuoriborAction:
        a, b, c = input().split()
        return [int(a), int(b), int(c)]
    
class RandomAgent(BaseAgent):
    env_id = ("puoribor", 3)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def __call__(self, state: puoribor.PuoriborState) -> puoribor.PuoriborAction:
        legal_actions_np = puoribor.PuoriborEnv().legal_actions(state, self.agent_id)
        print(legal_actions_np)
        return self._rng.choice(np.argwhere(legal_actions_np == 1))

def fallback_to_ascii(s: str) -> str:
    try:
        s.encode(sys.stdout.encoding)
    except UnicodeEncodeError:
        s = re.sub("[┌┬┐├┼┤└┴┘╋]", "+", re.sub("[─━]", "-", re.sub("[│┃]", "|", s)))
    return s

def colorize_walls(s: str) -> str:
    return s.replace("━", Fore.BLUE + "━" + Style.RESET_ALL).replace(
        "┃", Fore.RED + "┃" + Style.RESET_ALL
    )

def run():
    assert puoribor.PuoriborEnv.env_id == RandomAgent.env_id
    colorama.init()

    state = puoribor.PuoriborEnv().initialize_state()
    agents = [RandomAgent(0, 1), ManualAgent(1)]

    #print("\x1b[2J")
    #print(fallback_to_ascii(colorize_walls(str(state))))

    it = 0
    while not state.done:
        
        for agent in agents:
            
            action = agent(state)
            
            state = puoribor.PuoriborEnv().step(state, agent.agent_id, action)

            #print("\x1b[1;1H")
            print(fallback_to_ascii(colorize_walls(str(state))))

            #print(action)
            
            #a = input()
            if state.done:
                print(f"agent {agent.agent_id} won in {it} iters")
                break
        it += 1

if __name__ == "__main__":
    run()
