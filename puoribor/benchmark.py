"""
Puoribor Environment Speed Test
"""

import numpy as np
import time

from fights.base import BaseAgent
import pre_env
import new_env

class RandomAgent(BaseAgent):
    env_id = ("puoribor", 3)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def __call__(self, state: pre_env.PuoriborState) -> pre_env.PuoriborAction:
        legal_actions_np = pre_env.PuoriborEnv().legal_actions(state, self.agent_id)
        return self._rng.choice(np.argwhere(legal_actions_np == 1))

class FasterAgent(BaseAgent):
    env_id = ("puoribor", 3)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def __call__(self, state: new_env.PuoriborState) -> new_env.PuoriborAction:
        legal_actions_np = new_env.PuoriborEnv().legal_actions(state, self.agent_id)
        return self._rng.choice(np.argwhere(legal_actions_np == 1))

def run_original():
    assert pre_env.PuoriborEnv.env_id == RandomAgent.env_id
    start = time.time()

    for game in range(10):

        state = pre_env.PuoriborEnv().initialize_state()
        agents = [RandomAgent(0, game), RandomAgent(1, game)]

        while not state.done:

            for agent in agents:

                action = agent(state)
                state = pre_env.PuoriborEnv().step(state, agent.agent_id, action)

                if state.done:
                    break

    end = time.time()
    print(f"{end - start} sec")

def run_faster():
    assert new_env.PuoriborEnv.env_id == RandomAgent.env_id
    start = time.time()

    for game in range(10):

        state = new_env.PuoriborEnv().initialize_state()
        agents = [FasterAgent(0, game), FasterAgent(1, game)]

        while not state.done:

            for agent in agents:

                action = agent(state)
                state = new_env.PuoriborEnv().step(state, agent.agent_id, action)

                if state.done:
                    break

    end = time.time()
    print(f"{end - start} sec")

if __name__ == "__main__":
    run_original()
    run_faster()
