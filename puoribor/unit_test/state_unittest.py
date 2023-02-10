import unittest

import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from new.new_env import PuoriborEnv, PuoriborState


class TestPuoriborState(unittest.TestCase):
    def setUp(self):
        self.env = PuoriborEnv()
        self.initial_state = self.env.initialize_state()
        self.state = self.env.step(self.initial_state, 0, [0, 3, 0])
        self.state = self.env.step(self.state, 1, [2, 3, 0])
        self.state = self.env.step(self.state, 0, [0, 3, 1])

    def test_to_dict(self):
        serialized = self.state.to_dict()
        self.assertListEqual(serialized["board"], self.state.board.tolist())
        self.assertListEqual(
            serialized["walls_remaining"], self.state.walls_remaining.tolist()
        )
        self.assertListEqual(
            serialized["memory_cells"], self.state.memory_cells.tolist()
        )
        self.assertEqual(serialized["done"], self.state.done)

    def test_from_dict(self):
        serialized = {
            "board": self.initial_state.board.tolist(),
            "walls_remaining": self.initial_state.walls_remaining.tolist(),
            "memory_cells": self.initial_state.memory_cells.tolist(),
            "done": self.initial_state.done,
        }
        state = PuoriborState.from_dict(serialized)
        np.testing.assert_array_equal(state.board, self.initial_state.board)
        np.testing.assert_array_equal(
            state.walls_remaining, self.initial_state.walls_remaining
        )
        np.testing.assert_array_equal(
            state.memory_cells, self.initial_state.memory_cells
        )
        self.assertEqual(state.done, self.initial_state.done)

    def test_perspective(self):
        before_rotation = self.env.step(self.initial_state, 0, [1, 2, 3])
        before_rotation = self.env.step(before_rotation, 1, [2, 3, 5])
        np.testing.assert_array_equal(
            before_rotation.board, before_rotation.perspective(0)
        )
        rotated_board = before_rotation.perspective(1)
        rotated_state = self.env.step(self.initial_state, 1, [1, 5, 4])
        rotated_state = self.env.step(rotated_state, 0, [2, 4, 2])
        np.testing.assert_array_equal(rotated_board[2:], rotated_state.board[2:])


if __name__ == "__main__":
    unittest.main()