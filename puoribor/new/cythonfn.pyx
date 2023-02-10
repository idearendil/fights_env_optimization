import sys

import numpy as np

from queue import PriorityQueue
from collections import deque

def fast_step(
    pre_board,
    pre_walls_remaining,
    pre_memory_cells,
    int agent_id,
    action,
    int board_size
):

    cdef int action_type = action[0]
    cdef int x = action[1]
    cdef int y = action[2]

    board = np.copy(pre_board)
    walls_remaining = np.copy(pre_walls_remaining)
    memory_cells = np.copy(pre_memory_cells)

    cdef int [:,:,:] board_view = board
    cdef int [:] walls_remaining_view = walls_remaining
    cdef int [:,:,:,:] memory_cells_view = memory_cells
    
    if not _check_in_range(x, y, board_size):
        raise ValueError(f"out of board: {(x, y)}")
    if not 0 <= agent_id <= 1:
        raise ValueError(f"invalid agent_id: {agent_id}")

    close_ones = [set(), set()]
    open_ones = [set(), set()]

    if action_type == 0:  # Move piece
        current_pos = np.argwhere(pre_board[agent_id])[0]
        new_pos = np.array([x, y])
        opponent_pos = np.argwhere(pre_board[1 - agent_id])[0]
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
            elif _check_wall_blocked(board, current_pos[0], current_pos[1], opponent_pos[0], opponent_pos[1]):
                raise ValueError("cannot jump over walls")

            original_jump_pos = current_pos + 2 * (opponent_pos - current_pos)
            if _check_in_range(original_jump_pos[0], original_jump_pos[1], board_size) and not _check_wall_blocked(
                board, current_pos[0], current_pos[1], original_jump_pos[0], original_jump_pos[1]
            ):
                raise ValueError(
                    "cannot diagonally jump if linear jump is possible"
                )
            elif _check_wall_blocked(board, opponent_pos[0], opponent_pos[1], new_pos[0], new_pos[1]):
                raise ValueError("cannot jump over walls")
        elif _check_wall_blocked(board, current_pos[0], current_pos[1], new_pos[0], new_pos[1]):
            raise ValueError("cannot jump over walls")

        board_view[agent_id, current_pos[0], current_pos[1]] = 0
        board_view[agent_id, new_pos[0], new_pos[1]] = 1

    elif action_type == 1:  # Place wall horizontally
        if walls_remaining_view[agent_id] == 0:
            raise ValueError(f"no walls left for agent {agent_id}")
        if y == board_size-1:
            raise ValueError("cannot place wall on the edge")
        elif x == board_size-1:
            raise ValueError("right section out of board")
        elif np.any(board[2, x : x + 2, y]):
            raise ValueError("wall already placed")
        elif board_view[5, x, y]:
            raise ValueError("cannot create intersecting walls")
        board_view[2, x, y] = 1 + agent_id
        board_view[2, x + 1, y] = 1 + agent_id
        walls_remaining_view[agent_id] -= 1
        board_view[4, x, y] = 1

        if memory_cells_view[0, x, y, 1] == 2:   close_ones[0].add((x, y))
        elif memory_cells_view[0, x, y+1, 1] == 0:   close_ones[0].add((x, y+1))
        if memory_cells_view[1, x, y, 1] == 2:   close_ones[1].add((x, y))
        elif memory_cells_view[1, x, y+1, 1] == 0:   close_ones[1].add((x, y+1))

        if memory_cells_view[0, x+1, y, 1] == 2:   close_ones[0].add((x+1, y))
        elif memory_cells_view[0, x+1, y+1, 1] == 0:   close_ones[0].add((x+1, y+1))
        if memory_cells_view[1, x+1, y, 1] == 2:   close_ones[1].add((x+1, y))
        elif memory_cells_view[1, x+1, y+1, 1] == 0:   close_ones[1].add((x+1, y+1))

    elif action_type == 2:  # Place wall vertically
        if walls_remaining_view[agent_id] == 0:
            raise ValueError(f"no walls left for agent {agent_id}")
        if x == board_size-1:
            raise ValueError("cannot place wall on the edge")
        elif y == board_size-1:
            raise ValueError("right section out of board")
        elif np.any(board[3, x, y : y + 2]):
            raise ValueError("wall already placed")
        elif board_view[4, x, y]:
            raise ValueError("cannot create intersecting walls")
        board_view[3, x, y] = 1 + agent_id
        board_view[3, x, y + 1] = 1 + agent_id
        walls_remaining_view[agent_id] -= 1
        board_view[5, x, y] = 1

        if memory_cells_view[0, x, y, 1] == 1:   close_ones[0].add((x, y))
        elif memory_cells_view[0, x+1, y, 1] == 3:   close_ones[0].add((x+1, y))
        if memory_cells_view[1, x, y, 1] == 1:   close_ones[1].add((x, y))
        elif memory_cells_view[1, x+1, y, 1] == 3:   close_ones[1].add((x+1, y))

        if memory_cells_view[0, x, y+1, 1] == 1:   close_ones[0].add((x, y+1))
        elif memory_cells_view[0, x+1, y+1, 1] == 3:   close_ones[0].add((x+1, y+1))
        if memory_cells_view[1, x, y+1, 1] == 1:   close_ones[1].add((x, y+1))
        elif memory_cells_view[1, x+1, y+1, 1] == 3:   close_ones[1].add((x+1, y+1))

    elif action_type == 3:  # Rotate section
        if not _check_in_range(
            x, y,
            bottom_right=board_size-3
        ):
            raise ValueError("rotation region out of board")
        elif walls_remaining_view[agent_id] < 2:
            raise ValueError(f"less than two walls left for agent {agent_id}")

        horizontal_walls = set()
        vertical_walls = set()

        for coordinate_x in range(x, x+4, 1):
            for coordinate_y in range(y-1, y+4, 1):
                if coordinate_y >= 0 and coordinate_y <= 7:
                    if board_view[2, coordinate_x, coordinate_y]:
                        horizontal_walls.add((coordinate_x, coordinate_y))
                    if board_view[3, coordinate_y, coordinate_x]:
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
        board_view[2, :, board_size-1] = 0
        board_view[3, board_size-1, :] = 0
        board_view[4, :, board_size-1] = 0
        board_view[5, board_size-1, :] = 0

        walls_remaining_view[agent_id] -= 2

        for cx in range(x, x+4, 1):
            for cy in range(y-1, y+4, 1):
                if cy >= 0 and cy <= 7:
                    if board_view[2, cx, cy]:
                        if (cx, cy) not in horizontal_walls:
                            if memory_cells_view[0, cx, cy, 1] == 2:   close_ones[0].add((cx, cy))
                            elif memory_cells_view[0, cx, cy+1, 1] == 0:   close_ones[0].add((cx, cy+1))
                            if memory_cells_view[1, cx, cy, 1] == 2:   close_ones[1].add((cx, cy))
                            elif memory_cells_view[1, cx, cy+1, 1] == 0:   close_ones[1].add((cx, cy+1))
                    else:
                        if (cx, cy) in horizontal_walls:
                            if memory_cells_view[0, cx, cy, 0] > memory_cells_view[0, cx, cy+1, 0] + 1:   open_ones[0].add((cx, cy+1))
                            elif memory_cells_view[0, cx, cy+1, 0] > memory_cells_view[0, cx, cy, 0] + 1: open_ones[0].add((cx, cy))
                            if memory_cells_view[1, cx, cy, 0] > memory_cells_view[1, cx, cy+1, 0] + 1:   open_ones[1].add((cx, cy+1))
                            elif memory_cells_view[1, cx, cy+1, 0] > memory_cells_view[1, cx, cy, 0] + 1: open_ones[1].add((cx, cy))
                    if board_view[3, cy, cx]:
                        if (cy, cx) not in vertical_walls:
                            if memory_cells_view[0, cy, cx, 1] == 1:   close_ones[0].add((cy, cx))
                            elif memory_cells_view[0, cy+1, cx, 1] == 3:   close_ones[0].add((cy+1, cx))
                            if memory_cells_view[1, cy, cx, 1] == 1:   close_ones[1].add((cy, cx))
                            elif memory_cells_view[1, cy+1, cx, 1] == 3:   close_ones[1].add((cy+1, cx))
                    else:
                        if (cy, cx) in vertical_walls:
                            if memory_cells_view[0, cy, cx, 0] > memory_cells_view[0, cy+1, cx, 0] + 1:   open_ones[0].add((cy+1, cx))
                            elif memory_cells_view[0, cy+1, cx, 0] > memory_cells_view[0, cy, cx, 0] + 1: open_ones[0].add((cy, cx))
                            if memory_cells_view[1, cy, cx, 0] > memory_cells_view[1, cy+1, cx, 0] + 1:   open_ones[1].add((cy+1, cx))
                            elif memory_cells_view[1, cy+1, cx, 0] > memory_cells_view[1, cy, cx, 0] + 1: open_ones[1].add((cy, cx))
    else:
        raise ValueError(f"invalid action_type: {action_type}")

    if action_type > 0:
        
        directions = ((0, -1), (1, 0), (0, 1), (-1, 0))

        for agent_id in range(2):

            visited = close_ones[agent_id]
            q = deque(close_ones[agent_id])
            in_pri_q = open_ones[agent_id]
            pri_q = PriorityQueue()
            while q:
                here = q.popleft()
                memory_cells_view[agent_id, here[0], here[1], 0] = 99999
                memory_cells_view[agent_id, here[0], here[1], 1] = -1
                in_pri_q.discard(here)
                for dir_id, (dx, dy) in enumerate(directions):
                    there = (here[0] + dx, here[1] + dy)
                    if there in visited:
                        continue
                    if (not _check_in_range(there[0], there[1], board_size)) or _check_wall_blocked(board, here[0], here[1], there[0], there[1]):
                        continue
                    if memory_cells_view[agent_id, there[0], there[1], 1] == (dir_id + 2) % 4:
                        q.append(there)
                        visited.add(there)
                    else:
                        if memory_cells_view[agent_id, there[0], there[1], 0] < 99999:
                            in_pri_q.add(there)
            
            for element in in_pri_q:
                pri_q.put((memory_cells[agent_id, element[0], element[1], 0], element))

            while not pri_q.empty():
                dist, here = pri_q.get()
                for dir_id, (dx, dy) in enumerate(directions):
                    there = (here[0] + dx, here[1] + dy)
                    if (not _check_in_range(there[0], there[1], board_size)) or _check_wall_blocked(board, here[0], here[1], there[0], there[1]):
                        continue
                    if memory_cells_view[agent_id, there[0], there[1], 0] > dist + 1:
                        memory_cells_view[agent_id, there[0], there[1], 0] = dist + 1
                        memory_cells_view[agent_id, there[0], there[1], 1] = (dir_id + 2) % 4
                        pri_q.put((memory_cells_view[agent_id, there[0], there[1], 0], there))
        
        if not _check_path_exists(board, memory_cells, 0) or not _check_path_exists(board, memory_cells, 1):
            if action_type == 3:
                raise ValueError("cannot rotate to block all paths")
            else:
                raise ValueError("cannot place wall blocking all paths")

    return (board, walls_remaining, memory_cells, _check_wins(board))

def build_memory_cells(board, walls_remaining, done, board_size):

    directions = ((0, -1), (1, 0), (0, 1), (-1, 0))

    memory_cells = np.zeros((2, board_size, board_size, 2), dtype=np.int_)
    cdef int [:,:,:,:] memory_cells_view = memory_cells

    for agent_id in range(2):
        
        q = deque()
        visited = set()
        if agent_id == 0:
            for coordinate_x in range(board_size):
                q.append((coordinate_x, board_size-1))
                memory_cells_view[agent_id, coordinate_x, board_size-1, 0] = 0
                memory_cells_view[agent_id, coordinate_x, board_size-1, 1] = 2
                visited.add((coordinate_x, board_size-1))
        else:
            for coordinate_x in range(board_size):
                q.append((coordinate_x, 0))
                memory_cells_view[agent_id, coordinate_x, 0, 0] = 0
                memory_cells_view[agent_id, coordinate_x, 0, 1] = 0
                visited.add((coordinate_x, 0))
        while q:
            here = q.popleft()
            for dir_id, (dx, dy) in enumerate(directions):
                there = (here[0] + dx, here[1] + dy)
                if there in visited:
                    continue
                if (not _check_in_range(there[0], there[1], board_size)) or _check_wall_blocked(board, here[0], here[1], there[0], there[1]):
                    continue
                memory_cells_view[agent_id, there[0], there[1], 0] = memory_cells_view[agent_id, here[0], here[1], 0] + 1
                memory_cells_view[agent_id, there[0], there[1], 1] = (dir_id + 2) % 4
                q.append(there)
                visited.add(there)

    return memory_cells

def legal_actions(state, int agent_id, int board_size):
    """
    Find possible actions for the agent.

    :arg state:
        Current state of the environment.
    :arg agent_id:
        Agent_id of the agent.
    
    :returns:
        A numpy array of shape (4, 9, 9) which is one-hot encoding of possible actions.
    """
    cdef int dir_x, dir_y, action_type, next_pos_x, next_pos_y, coordinate_x, coordinate_y

    legal_actions_np = np.zeros((4, 9, 9), dtype=np.int_)
    now_pos = tuple(np.argwhere(state.board[agent_id] == 1)[0])
    directions = ((0, -2), (-1, -1), (0, -1), (1, -1), (-2, 0), (-1, 0), (1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2))

    for dir_x, dir_y in directions:
        next_pos_x = now_pos[0] + dir_x
        next_pos_y = now_pos[1] + dir_y
        if _check_in_range(next_pos_x, next_pos_y, board_size):
            try:
                fast_step(state.board, state.walls_remaining, state.memory_cells, agent_id, (0, next_pos_x, next_pos_y), board_size)
            except:
                ...
            else:
                legal_actions_np[0, next_pos_x, next_pos_y] = 1
    for action_type in (1, 2):
        for coordinate_x in range(board_size-1):
            for coordinate_y in range(board_size-1):
                try:
                    fast_step(state.board, state.walls_remaining, state.memory_cells, agent_id, (action_type, coordinate_x, coordinate_y), board_size)
                except:
                    ...
                else:
                    legal_actions_np[action_type, coordinate_x, coordinate_y] = 1
    for coordinate_x in range(board_size-3):
        for coordinate_y in range(board_size-3):
            try:
                fast_step(state.board, state.walls_remaining, state.memory_cells, agent_id, (3, coordinate_x, coordinate_y), board_size)
            except:
                ...
            else:
                legal_actions_np[3, coordinate_x, coordinate_y] = 1
    return legal_actions_np

cdef _check_in_range(int pos_x, int pos_y, int bottom_right = 9):
    return (0 <= pos_x < bottom_right and 0 <= pos_y < bottom_right)

cdef _check_path_exists(board, memory_cells, agent_id: int):
    agent_pos = np.argwhere(board[agent_id])[0]
    return memory_cells[agent_id, agent_pos[0], agent_pos[1], 0] < 99999

cdef _check_wall_blocked(board, int cx, int cy, int nx, int ny):
    if nx > cx:
        return np.any(board[3, cx : nx, cy])
    if nx < cx:
        return np.any(board[3, nx : cx, cy])
    if ny > cy:
        return np.any(board[2, cx, cy : ny])
    if ny < cy:
        return np.any(board[2, cx, ny : cy])
    return False

cdef _check_wins(board):
    return board[0, :, -1].any() or board[1, :, 0].any()