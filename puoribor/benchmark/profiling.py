"""
Puoribor Environment Profiling
"""

import numpy as np
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from fights.base import BaseAgent
from new import new_env
import prof_env

class FasterAgent(BaseAgent):
    env_id = ("puoribor", 3)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _get_all_actions(self, state: new_env.PuoriborState):
        actions = []
        for action_type in [0, 1, 2, 3]:
            for coordinate_x in range(new_env.PuoriborEnv.board_size):
                for coordinate_y in range(new_env.PuoriborEnv.board_size):
                    action = [action_type, coordinate_x, coordinate_y]
                    try:
                        new_env.PuoriborEnv().step(state, self.agent_id, action)
                    except:
                        ...
                    else:
                        actions.append(action)
        return actions

    def __call__(self, state: new_env.PuoriborState) -> new_env.PuoriborAction:
        actions = self._get_all_actions(state)
        return self._rng.choice(actions)

def run():
    assert new_env.PuoriborEnv.env_id == FasterAgent.env_id

    state = new_env.PuoriborEnv().initialize_state()
    agents = [FasterAgent(0, 0), FasterAgent(1, 0)]

    where_step = 5

    for step in range(where_step):
    
        for agent in agents:

            action = agent(state)
            state = new_env.PuoriborEnv().step(state, agent.agent_id, action)
        
    action = agents[0](state)
    state = prof_env.PuoriborEnv().step(state, agent.agent_id, action)

if __name__ == "__main__":
    run()
