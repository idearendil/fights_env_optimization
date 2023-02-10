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

def fast_step(
    pre_board,
    pre_walls_remaining,
    pre_memory_cells,
    agent_id,
    action
):

    action = np.asanyarray(action).astype(np.int_)

    cdef int action_type, x, y
    action_type = action[0]
    x = action[1]
    y = action[2]
    
    if not _check_in_range((x, y)):
        raise ValueError(f"out of board: {(x, y)}")
    if not 0 <= agent_id <= 1:
        raise ValueError(f"invalid agent_id: {agent_id}")

    board = np.copy(pre_board)
    walls_remaining = np.copy(pre_walls_remaining)
    memory_cells = np.copy(pre_memory_cells)
    close_ones = [set(), set()]
    open_ones = [set(), set()]

    if action_type == 0:  # Move piece
        current_pos = np.argwhere(pre_board[agent_id] == 1)[0]
        new_pos = np.array([x, y])
        opponent_pos = np.argwhere(pre_board[1 - agent_id] == 1)[0]
        if np.all(new_pos == opponent_pos):
            raise ValueError("cannot move to opponent's position")

        delta = new_pos - current_pos
        taxicab_dist = np.abs(delta).sum()
        if taxicab_dist == 0:
            raise ValueError("cannot move zero blocks")
        elif taxicab_dist > 2:
            raise ValueError("cannot move more than two blocks")
        elif (
            taxicab_dist == 2
            and np.any(delta == 0)
            and not np.all(current_pos + delta // 2 == opponent_pos)
        ):
            raise ValueError("cannot jump over nothing")

        if np.all(delta):  # If moving diagonally
            if np.any(current_pos + delta * [0, 1] != opponent_pos) and np.any(
                current_pos + delta * [1, 0] != opponent_pos
            ):
                # Only diagonal jumps are permitted.
                # Agents cannot simply move in diagonal direction.
                raise ValueError("cannot move diagonally")
            elif _check_wall_blocked(board, tuple(current_pos), tuple(opponent_pos)):
                raise ValueError("cannot jump over walls")

            original_jump_pos = current_pos + 2 * (opponent_pos - current_pos)
            if _check_in_range(
                (original_jump_pos[0], original_jump_pos[1])
            ) and not _check_wall_blocked(
                board, tuple(current_pos), tuple(original_jump_pos)
            ):
                raise ValueError(
                    "cannot diagonally jump if linear jump is possible"
                )
            elif _check_wall_blocked(board, tuple(opponent_pos), tuple(new_pos)):
                raise ValueError("cannot jump over walls")
        elif _check_wall_blocked(board, tuple(current_pos), tuple(new_pos)):
            raise ValueError("cannot jump over walls")

        board[agent_id][tuple(current_pos)] = 0
        board[agent_id][tuple(new_pos)] = 1

    elif action_type == 1:  # Place wall horizontally
        if walls_remaining[agent_id] == 0:
            raise ValueError(f"no walls left for agent {agent_id}")
        if y == 8:
            raise ValueError("cannot place wall on the edge")
        elif x == 8:
            raise ValueError("right section out of board")
        elif np.any(board[2, x : x + 2, y]):
            raise ValueError("wall already placed")
        elif board[5, x, y]:
            raise ValueError("cannot create intersecting walls")
        board[2, x, y] = 1 + agent_id
        board[2, x + 1, y] = 1 + agent_id
        walls_remaining[agent_id] -= 1
        board[4, x, y] = 1

        if memory_cells[0, x, y, 1] == 2:   close_ones[0].add((x, y))
        elif memory_cells[0, x, y+1, 1] == 0:   close_ones[0].add((x, y+1))
        if memory_cells[1, x, y, 1] == 2:   close_ones[1].add((x, y))
        elif memory_cells[1, x, y+1, 1] == 0:   close_ones[1].add((x, y+1))

        if memory_cells[0, x+1, y, 1] == 2:   close_ones[0].add((x+1, y))
        elif memory_cells[0, x+1, y+1, 1] == 0:   close_ones[0].add((x+1, y+1))
        if memory_cells[1, x+1, y, 1] == 2:   close_ones[1].add((x+1, y))
        elif memory_cells[1, x+1, y+1, 1] == 0:   close_ones[1].add((x+1, y+1))

    elif action_type == 2:  # Place wall vertically
        if walls_remaining[agent_id] == 0:
            raise ValueError(f"no walls left for agent {agent_id}")
        if x == 8:
            raise ValueError("cannot place wall on the edge")
        elif y == 8:
            raise ValueError("right section out of board")
        elif np.any(board[3, x, y : y + 2]):
            raise ValueError("wall already placed")
        elif board[4, x, y]:
            raise ValueError("cannot create intersecting walls")
        board[3, x, y] = 1 + agent_id
        board[3, x, y + 1] = 1 + agent_id
        walls_remaining[agent_id] -= 1
        board[5, x, y] = 1

        if memory_cells[0, x, y, 1] == 1:   close_ones[0].add((x, y))
        elif memory_cells[0, x+1, y, 1] == 3:   close_ones[0].add((x+1, y))
        if memory_cells[1, x, y, 1] == 1:   close_ones[1].add((x, y))
        elif memory_cells[1, x+1, y, 1] == 3:   close_ones[1].add((x+1, y))

        if memory_cells[0, x, y+1, 1] == 1:   close_ones[0].add((x, y+1))
        elif memory_cells[0, x+1, y+1, 1] == 3:   close_ones[0].add((x+1, y+1))
        if memory_cells[1, x, y+1, 1] == 1:   close_ones[1].add((x, y+1))
        elif memory_cells[1, x+1, y+1, 1] == 3:   close_ones[1].add((x+1, y+1))

    elif action_type == 3:  # Rotate section
        if not _check_in_range(
            (x, y),
            bottom_right=6
        ):
            raise ValueError("rotation region out of board")
        elif walls_remaining[agent_id] < 2:
            raise ValueError(f"less than two walls left for agent {agent_id}")

        horizontal_walls = set()
        vertical_walls = set()

        for coordinate_x in range(x, x+4, 1):
            for coordinate_y in range(y-1, y+4, 1):
                if coordinate_y >= 0 and coordinate_y <= 7:
                    if board[2, coordinate_x, coordinate_y]:
                        horizontal_walls.add((coordinate_x, coordinate_y))
                    if board[3, coordinate_y, coordinate_x]:
                        vertical_walls.add((coordinate_y, coordinate_x))

        padded_horizontal = np.pad(board[2], 1, constant_values=0)
        padded_vertical = np.pad(board[3], 1, constant_values=0)
        padded_horizontal_midpoints = np.pad(board[4], 1, constant_values=0)
        padded_vertical_midpoints = np.pad(board[5], 1, constant_values=0)
        px, py = x + 1, y + 1
        horizontal_region = np.copy(padded_horizontal[px : px + 4, py - 1 : py + 4])
        vertical_region = np.copy(padded_vertical[px - 1 : px + 4, py : py + 4])
        padded_horizontal_midpoints[px - 1, py - 1 : py + 4] = 0
        padded_horizontal_midpoints[px + 3, py - 1 : py + 4] = 0
        padded_vertical_midpoints[px - 1 : px + 4, py - 1] = 0
        padded_vertical_midpoints[px - 1 : px + 4, py + 3] = 0
        horizontal_region_midpoints = np.copy(
            padded_horizontal_midpoints[px : px + 4, py - 1 : py + 4]
        )
        vertical_region_midpoints = np.copy(
            padded_vertical_midpoints[px - 1 : px + 4, py : py + 4]
        )
        horizontal_region_new = np.rot90(vertical_region)
        vertical_region_new = np.rot90(horizontal_region)
        horizontal_region_midpoints_new = np.rot90(vertical_region_midpoints)
        vertical_region_midpoints_new = np.rot90(horizontal_region_midpoints)
        padded_horizontal[px : px + 4, py - 1 : py + 4] = horizontal_region_new
        padded_vertical[px - 1 : px + 4, py : py + 4] = vertical_region_new
        padded_horizontal_midpoints[
            px - 1 : px + 3, py - 1 : py + 4
        ] = horizontal_region_midpoints_new
        padded_vertical_midpoints[
            px - 1 : px + 4, py : py + 4
        ] = vertical_region_midpoints_new
        board[2] = padded_horizontal[1:-1, 1:-1]
        board[3] = padded_vertical[1:-1, 1:-1]
        board[4] = padded_horizontal_midpoints[1:-1, 1:-1]
        board[5] = padded_vertical_midpoints[1:-1, 1:-1]
        board[2, :, 8] = 0
        board[3, 8, :] = 0
        board[4, :, 8] = 0
        board[5, 8, :] = 0

        walls_remaining[agent_id] -= 2

        for cx in range(x, x+4, 1):
            for cy in range(y-1, y+4, 1):
                if cy >= 0 and cy <= 7:
                    if board[2, cx, cy]:
                        if (cx, cy) not in horizontal_walls:
                            if memory_cells[0, cx, cy, 1] == 2:   close_ones[0].add((cx, cy))
                            elif memory_cells[0, cx, cy+1, 1] == 0:   close_ones[0].add((cx, cy+1))
                            if memory_cells[1, cx, cy, 1] == 2:   close_ones[1].add((cx, cy))
                            elif memory_cells[1, cx, cy+1, 1] == 0:   close_ones[1].add((cx, cy+1))
                    else:
                        if (cx, cy) in horizontal_walls:
                            if memory_cells[0, cx, cy, 0] > memory_cells[0, cx, cy+1, 0] + 1:   open_ones[0].add((cx, cy+1))
                            elif memory_cells[0, cx, cy+1, 0] > memory_cells[0, cx, cy, 0] + 1: open_ones[0].add((cx, cy))
                            if memory_cells[1, cx, cy, 0] > memory_cells[1, cx, cy+1, 0] + 1:   open_ones[1].add((cx, cy+1))
                            elif memory_cells[1, cx, cy+1, 0] > memory_cells[1, cx, cy, 0] + 1: open_ones[1].add((cx, cy))
                    if board[3, cy, cx]:
                        if (cy, cx) not in vertical_walls:
                            if memory_cells[0, cy, cx, 1] == 1:   close_ones[0].add((cy, cx))
                            elif memory_cells[0, cy+1, cx, 1] == 3:   close_ones[0].add((cy+1, cx))
                            if memory_cells[1, cy, cx, 1] == 1:   close_ones[1].add((cy, cx))
                            elif memory_cells[1, cy+1, cx, 1] == 3:   close_ones[1].add((cy+1, cx))
                    else:
                        if (cy, cx) in vertical_walls:
                            if memory_cells[0, cy, cx, 0] > memory_cells[0, cy+1, cx, 0] + 1:   open_ones[0].add((cy+1, cx))
                            elif memory_cells[0, cy+1, cx, 0] > memory_cells[0, cy, cx, 0] + 1: open_ones[0].add((cy, cx))
                            if memory_cells[1, cy, cx, 0] > memory_cells[1, cy+1, cx, 0] + 1:   open_ones[1].add((cy+1, cx))
                            elif memory_cells[1, cy+1, cx, 0] > memory_cells[1, cy, cx, 0] + 1: open_ones[1].add((cy, cx))
    else:
        raise ValueError(f"invalid action_type: {action_type}")

    if action_type > 0:
        
        directions = ((0, -1), (1, 0), (0, 1), (-1, 0))

        for agent_id in range(2):

            visited = close_ones[agent_id]
            q = Deque(close_ones[agent_id])
            in_pri_q = open_ones[agent_id]
            pri_q = PriorityQueue()
            while q:
                here = q.popleft()
                memory_cells[agent_id, here[0], here[1], 0] = 99999
                memory_cells[agent_id, here[0], here[1], 1] = -1
                in_pri_q.discard(here)
                for dir_id, (dx, dy) in enumerate(directions):
                    there = (here[0] + dx, here[1] + dy)
                    if there in visited:
                        continue
                    if (not _check_in_range(there)) or _check_wall_blocked(board, here, there):
                        continue
                    if memory_cells[agent_id, there[0], there[1], 1] == (dir_id + 2) % 4:
                        q.append(there)
                        visited.add(there)
                    else:
                        if memory_cells[agent_id, there[0], there[1], 0] < 99999:
                            in_pri_q.add(there)
            
            for element in in_pri_q:
                pri_q.put((memory_cells[agent_id, element[0], element[1], 0], element))

            while not pri_q.empty():
                dist, here = pri_q.get()
                for dir_id, (dx, dy) in enumerate(directions):
                    there = (here[0] + dx, here[1] + dy)
                    if (not _check_in_range(there)) or _check_wall_blocked(board, here, there):
                        continue
                    if memory_cells[agent_id, there[0], there[1], 0] > dist + 1:
                        memory_cells[agent_id, there[0], there[1], 0] = dist + 1
                        memory_cells[agent_id, there[0], there[1], 1] = (dir_id + 2) % 4
                        pri_q.put((memory_cells[agent_id, there[0], there[1], 0], there))
        
        if not _check_path_exists(board, memory_cells, 0) or not _check_path_exists(board, memory_cells, 1):
            if action_type == 3:
                raise ValueError("cannot rotate to block all paths")
            else:
                raise ValueError("cannot place wall blocking all paths")

    return (board, walls_remaining, memory_cells, _check_wins(board))

def _check_in_range(pos: tuple, bottom_right: int = None) -> np.bool_:
    if bottom_right is None:
        bottom_right = 9
    return ((0 <= pos[0] < bottom_right) and (0 <= pos[1] < bottom_right))

def _check_path_exists(board: NDArray[np.int_], memory_cells: NDArray[np.int_], agent_id: int) -> bool:
    agent_pos = np.argwhere(board[agent_id] == 1)[0]
    return memory_cells[agent_id, agent_pos[0], agent_pos[1], 0] < 99999

def _check_wall_blocked(
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

def _check_wins(board: NDArray[np.int_]) -> bool:
    return board[0, :, -1].any() or board[1, :, 0].any()