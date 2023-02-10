"""
Puoribor, a variant of the classical `Quoridor <https://en.wikipedia.org/wiki/Quoridor>`_ game.
Coordinates are specified in the form of ``(x, y)``, where ``(0, 0)`` is the top left corner.
All coordinates and directions are absolute and does not change between agents.
Directions
    - Top: `+y`
    - Right: `+x`
    - Bottom: `-y`
    - Left: `-x`
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from queue import PriorityQueue

if sys.version_info < (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias

from fights.base import BaseEnv, BaseState

import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from . import cythonfn

PuoriborAction: TypeAlias = ArrayLike
"""
Alias of :obj:`ArrayLike` to describe the action type.
Encoded as an array of shape ``(3,)``, in the form of
[ `action_type`, `coordinate_x`, `coordinate_y` ].
`action_type`
    - 0 (move piece)
    - 1 (place wall horizontally)
    - 2 (place wall vertically)
    - 3 (rotate section)
`coordinate_x`, `coordinate_y`
    - position to move the piece to
    - top or left position to place the wall
    - top left position of the section to rotate
"""


@dataclass
class PuoriborState(BaseState):
    """
    ``PuoriborState`` represents the game state.
    """

    board: NDArray[np.int_]
    """
    Array of shape ``(C, W, H)``, where C is channel index and W, H is board width,
    height.
    Channels
        - ``C = 0``: one-hot encoded position of agent 0. (starts from top)
        - ``C = 1``: one-hot encoded position of agent 1. (starts from bottom)
        - ``C = 2``: label encoded positions of horizontal walls. (1 for wall placed
          by agent 0, 2 for agent 1)
        - ``C = 3``: label encoded positions of vertical walls. (encoding is same as
          ``C = 2``)
        - ``C = 4``: one-hot encoded positions of horizontal walls' midpoints.
        - ``C = 5``: one-hot encoded positions of vertical walls' midpoints.
    """

    walls_remaining: NDArray[np.int_]
    """
    Array of shape ``(2,)``, in the form of [ `agent0_remaining_walls`,
    `agent1_remaining_walls` ].
    """

    memory_cells: NDArray[np.int_]
    """
    Array of shape ''(2, 9, 9, 2)''.
    First index is agent_id, second and third index is x and y of the cell.
    It should memorize two information per cell.
    One is the shortest distance from the destination, and the other is the pointing direction of the cell.
    
    Pointing Direction
        - 0 : 12 o'clock(up)
        - 1 : 3 o'clock(right)
        - 2 : 6 o'clock(down)
        - 3 : 9 o'clock(left)
    """

    done: bool = False
    """
    Boolean value indicating whether the game is done.
    """

    def __str__(self) -> str:
        """
        Generate a human-readable string representation of the board.
        Uses unicode box drawing characters.
        """

        table_top = "┌───┬───┬───┬───┬───┬───┬───┬───┬───┐"
        vertical_wall = "│"
        vertical_wall_bold = "┃"
        horizontal_wall = "───"
        horizontal_wall_bold = "━━━"
        left_intersection = "├"
        middle_intersection = "┼"
        middle_intersection_bold = "╋"
        right_intersection = "┤"
        left_intersection_bottom = "└"
        middle_intersection_bottom = "┴"
        right_intersection_bottom = "┘"
        result = table_top + "\n"

        for y in range(9):
            board_line = self.board[:, :, y]
            result += vertical_wall
            for x in range(9):
                board_cell = board_line[:, x]
                if board_cell[0]:
                    result += " 0 "
                elif board_cell[1]:
                    result += " 1 "
                else:
                    result += "   "
                if board_cell[3]:
                    result += vertical_wall_bold
                elif x == 8:
                    result += vertical_wall
                else:
                    result += " "
                if x == 8:
                    result += "\n"
            result += left_intersection_bottom if y == 8 else left_intersection
            for x in range(9):
                board_cell = board_line[:, x]
                if board_cell[2]:
                    result += horizontal_wall_bold
                elif y == 8:
                    result += horizontal_wall
                else:
                    result += "   "
                if x == 8:
                    result += (
                        right_intersection_bottom if y == 8 else right_intersection
                    )
                else:
                    if np.any(self.board[4:, x, y]):
                        result += middle_intersection_bold
                    else:
                        result += (
                            middle_intersection_bottom
                            if y == 8
                            else middle_intersection
                        )
            result += "\n"

        return result

    def perspective(self, agent_id: int) -> NDArray[np.int_]:
        """
        Return board where specified agent with ``agent_id`` is on top.
        :arg agent_id:
            The ID of agent to use as base.
        :returns:
            A rotated ``board`` array. The board's channel 0 will contain position of
            agent of id ``agent_id``, and channel 1 will contain the opponent's
            position. In channel 2 and 3, walles labeled with 1 are set by agent of id
            ``agent_id``, and the others are set by the opponent.
        """
        if agent_id == 0:
            return self.board
        inverted_walls = (self.board[2:4] == 2).astype(np.int_) + (
            self.board[2:4] == 1
        ).astype(np.int_) * 2
        rotated = np.stack(
            [
                np.rot90(self.board[1], 2),
                np.rot90(self.board[0], 2),
                np.pad(
                    np.rot90(inverted_walls[0], 2)[:, 1:],
                    ((0, 0), (0, 1)),
                    constant_values=0,
                ),
                np.pad(
                    np.rot90(inverted_walls[1], 2)[1:],
                    ((0, 1), (0, 0)),  # type: ignore
                    constant_values=0,
                ),
                np.pad(
                    np.rot90(self.board[4], 2)[1:, 1:],
                    ((0, 1), (0, 1)),  # type: ignore
                    constant_values=0,
                ),
                np.pad(
                    np.rot90(self.board[5], 2)[1:, 1:],
                    ((0, 1), (0, 1)),  # type: ignore
                    constant_values=0,
                ),
            ]
        )
        return rotated

    def to_dict(self) -> Dict:
        """
        Serialize state object to dict.
        :returns:
            A serialized dict.
        """
        return {
            "board": self.board.tolist(),
            "walls_remaining": self.walls_remaining.tolist(),
            "memory_cells": self.memory_cells.tolist(),
            "done": self.done,
        }

    @staticmethod
    def from_dict(serialized) -> PuoriborState:
        """
        Deserialize from serialized dict.
        :arg serialized:
            A serialized dict.
        :returns:
            Deserialized ``PuoriborState`` object.
        """
        return PuoriborState(
            board=np.array(serialized["board"]),
            walls_remaining=np.array(serialized["walls_remaining"]),
            memory_cells=np.array(serialized["memory_cells"]),
            done=serialized["done"],
        )


class PuoriborEnv(BaseEnv[PuoriborState, PuoriborAction]):
    env_id = ("puoribor", 3)  # type: ignore
    """
    Environment identifier in the form of ``(name, version)``.
    """

    board_size: int = 9
    """
    Size (width and height) of the board.
    """

    max_walls: int = 10
    """
    Maximum allowed walls per agent.
    """

    def step(
        self,
        state: PuoriborState,
        agent_id: int,
        action: PuoriborAction,
        *,
        pre_step_fn: Optional[
            Callable[[PuoriborState, int, PuoriborAction], None]
        ] = None,
        post_step_fn: Optional[
            Callable[[PuoriborState, int, PuoriborAction], None]
        ] = None,
    ) -> PuoriborState:
        """
        Step through the game, calculating the next state given the current state and
        action to take.
        :arg state:
            Current state of the environment.
        :arg agent_id:
            ID of the agent that takes the action. (``0`` or ``1``)
        :arg action:
            Agent action, encoded in the form described by :obj:`PuoriborAction`.
        :arg pre_step_fn:
            Callback to run before executing action. ``state``, ``agent_id`` and
            ``action`` will be provided as arguments.
        :arg post_step_fn:
            Callback to run after executing action. The calculated state, ``agent_id``
            and ``action`` will be provided as arguments.
        :returns:
            A copy of the object with the restored state.
        """

        if pre_step_fn is not None:
            pre_step_fn(state, agent_id, action)

        next_information = cythonfn.fast_step(state.board, state.walls_remaining, state.memory_cells, agent_id, action)

        next_state = PuoriborState(
            board=next_information[0],
            walls_remaining=next_information[1],
            memory_cells=next_information[2],
            done=next_information[3],
        )

        if post_step_fn is not None:
            post_step_fn(next_state, agent_id, action)
        return next_state
    
    def legal_actions(self, state: PuoriborState, agent_id: int) -> NDArray[np.int_]:
        """
        Find possible actions for the agent.

        :arg state:
            Current state of the environment.
        :arg agent_id:
            Agent_id of the agent.
        
        :returns:
            A numpy array of shape (4, 9, 9) which is one-hot encoding of possible actions.
        """
        legal_actions_np = np.zeros((4, 9, 9), dtype=np.int_)
        now_pos = tuple(np.argwhere(state.board[agent_id] == 1)[0])
        directions = ((0, -2), (-1, -1), (0, -1), (1, -1), (-2, 0), (-1, 0), (1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2))
        for dir_x, dir_y in directions:
            next_pos = (now_pos[0] + dir_x, now_pos[1] + dir_y)
            if self._check_in_range(next_pos):
                try:
                    self.step(state, agent_id, (0, next_pos[0], next_pos[1]))
                except:
                    ...
                else:
                    legal_actions_np[0, next_pos[0], next_pos[1]] = 1
        for action_type in (1, 2):
            for coordinate_x in range(self.board_size-1):
                for coordinate_y in range(self.board_size-1):
                    try:
                        self.step(state, agent_id, (action_type, coordinate_x, coordinate_y))
                    except:
                        ...
                    else:
                        legal_actions_np[action_type, coordinate_x, coordinate_y] = 1
        for coordinate_x in range(self.board_size-3):
            for coordinate_y in range(self.board_size-3):
                try:
                    self.step(state, agent_id, (3, coordinate_x, coordinate_y))
                except:
                    ...
                else:
                    legal_actions_np[3, coordinate_x, coordinate_y] = 1
        return legal_actions_np

    def _check_in_range(self, pos: tuple, bottom_right: int = None) -> np.bool_:
        if bottom_right is None:
            bottom_right = self.board_size
        return ((0 <= pos[0] < bottom_right) and (0 <= pos[1] < bottom_right))

    def _check_path_exists(self, board: NDArray[np.int_], memory_cells: NDArray[np.int_], agent_id: int) -> bool:
        agent_pos = np.argwhere(board[agent_id] == 1)[0]
        return memory_cells[agent_id, agent_pos[0], agent_pos[1], 0] < 99999

    def _check_wall_blocked(
        self,
        board: NDArray[np.int_],
        current_pos: tuple,
        new_pos: tuple,
    ) -> bool:
        if new_pos[0] > current_pos[0]:
            return np.any(board[3, current_pos[0] : new_pos[0], current_pos[1]])
        if new_pos[0] < current_pos[0]:
            return np.any(board[3, new_pos[0] : current_pos[0], current_pos[1]])
        if new_pos[1] > current_pos[1]:
            return np.any(board[2, current_pos[0], current_pos[1] : new_pos[1]])
        if new_pos[1] < current_pos[1]:
            return np.any(board[2, current_pos[0], new_pos[1] : current_pos[1]])
        return False

    def _check_wins(self, board: NDArray[np.int_]) -> bool:
        return board[0, :, -1].any() or board[1, :, 0].any()

    def _build_state(self, board: NDArray[np.int_], walls_remaining: NDArray[np.int_], done: bool) -> PuoriborState:
        """
        Build a state(including memory_cells) from the current board information(board, walls_remaining and done).
        :arg state:
            Current state of the environment.
        :returns:
            A state which board is same as the input.
        """
        directions = ((0, -1), (1, 0), (0, 1), (-1, 0))

        memory_cells = np.zeros((2, self.board_size, self.board_size, 2), dtype=np.int_)

        for agent_id in range(2):
            
            q = Deque()
            visited = set()
            if agent_id == 0:
                for coordinate_x in range(self.board_size):
                    q.append((coordinate_x, self.board_size-1))
                    memory_cells[agent_id, coordinate_x, self.board_size-1, 0] = 0
                    memory_cells[agent_id, coordinate_x, self.board_size-1, 1] = 2
                    visited.add((coordinate_x, self.board_size-1))
            else:
                for coordinate_x in range(self.board_size):
                    q.append((coordinate_x, 0))
                    memory_cells[agent_id, coordinate_x, 0, 0] = 0
                    memory_cells[agent_id, coordinate_x, 0, 1] = 0
                    visited.add((coordinate_x, 0))
            while q:
                here = q.popleft()
                for dir_id, (dx, dy) in enumerate(directions):
                    there = (here[0] + dx, here[1] + dy)
                    if (not self._check_in_range(there)) or self._check_wall_blocked(board, here, there):
                        continue
                    if there in visited:
                        continue
                    memory_cells[agent_id, there[0], there[1], 0] = memory_cells[agent_id, here[0], here[1], 0] + 1
                    memory_cells[agent_id, there[0], there[1], 1] = (dir_id + 2) % 4
                    q.append(there)
                    visited.add(there)
        
        new_state = PuoriborState(
            board=board,
            walls_remaining=walls_remaining,
            memory_cells=memory_cells,
            done=done,
        )

        return new_state

    def initialize_state(self) -> PuoriborState:
        """
        Initialize a :obj:`PuoriborState` object with correct environment parameters.
        :returns:
            Created initial state object.
        """
        if self.board_size % 2 == 0:
            raise ValueError(
                f"cannot center pieces with even board_size={self.board_size}, please "
                "initialize state manually"
            )

        starting_pos_0 = np.zeros((self.board_size, self.board_size), dtype=np.int_)
        starting_pos_0[(self.board_size - 1) // 2, 0] = 1

        starting_board = np.stack(
            [
                np.copy(starting_pos_0),
                np.fliplr(starting_pos_0),
                np.zeros((self.board_size, self.board_size), dtype=np.int_),
                np.zeros((self.board_size, self.board_size), dtype=np.int_),
                np.zeros((self.board_size, self.board_size), dtype=np.int_),
                np.zeros((self.board_size, self.board_size), dtype=np.int_),
            ]
        )

        return self._build_state(starting_board, np.array((self.max_walls, self.max_walls)), False)