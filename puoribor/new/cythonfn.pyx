import numpy as np
cimport numpy as np

from queue import PriorityQueue
from collections import deque

def fast_step(
    pre_board,
    pre_walls_remaining,
    int agent_id,
    action,
    int board_size
):

    cdef int action_type = action[0]
    cdef int x = action[1]
    cdef int y = action[2]

    board = np.copy(pre_board)
    walls_remaining = np.copy(pre_walls_remaining)

    cdef int [:,:,:] board_view = board
    cdef int [:] walls_remaining_view = walls_remaining

    cdef int curpos_x, curpos_y, newpos_x, newpos_y, opppos_x, opppos_y, delpos_x, delpos_y, taxicab_dist, original_jump_pos_x, original_jump_pos_y
    
    if not _check_in_range(x, y, board_size):
        raise ValueError(f"out of board: {(x, y)}")
    if not 0 <= agent_id <= 1:
        raise ValueError(f"invalid agent_id: {agent_id}")
    
    close_ones = [set(), set()]
    open_ones = [set(), set()]

    if action_type == 0:  # Move piece
        curpos_x, curpos_y = _agent_pos(board_view, agent_id, board_size)
        opppos_x, opppos_y = _agent_pos(board_view, 1-agent_id, board_size)
        newpos_x = x
        newpos_y = y

        if newpos_x == opppos_x and newpos_y == opppos_y:
            raise ValueError("cannot move to opponent's position")

        delpos_x = newpos_x - curpos_x
        delpos_y = newpos_y - curpos_y
        taxicab_dist = abs(delpos_x) + abs(delpos_y)
        if taxicab_dist == 0:
            raise ValueError("cannot move zero blocks")
        elif taxicab_dist > 2:
            raise ValueError("cannot move more than two blocks")
        elif (
            taxicab_dist == 2
            and (delpos_x == 0 or delpos_y == 0)
            and not (curpos_x + delpos_x / 2 == opppos_x and curpos_y + delpos_y / 2 == opppos_y)
        ):
            raise ValueError("cannot jump over nothing")

        if delpos_x and delpos_y:  # If moving diagonally
            if (curpos_x + delpos_x != opppos_x or curpos_y != opppos_y) and (
                curpos_x != opppos_x or curpos_y + delpos_y != opppos_y
            ):
                # Only diagonal jumps are permitted.
                # Agents cannot simply move in diagonal direction.
                raise ValueError("cannot move diagonally")
            elif _check_wall_blocked(board_view, curpos_x, curpos_y, opppos_x, opppos_y):
                raise ValueError("cannot jump over walls")

            original_jump_pos_x = curpos_x + 2 * (opppos_x - curpos_x)
            original_jump_pos_y = curpos_y + 2 * (opppos_y - curpos_y)
            if _check_in_range(original_jump_pos_x, original_jump_pos_y, board_size) and not _check_wall_blocked(
                board_view, curpos_x, curpos_y, original_jump_pos_x, original_jump_pos_y
            ):
                raise ValueError(
                    "cannot diagonally jump if linear jump is possible"
                )
            elif _check_wall_blocked(board_view, opppos_x, opppos_y, newpos_x, newpos_y):
                raise ValueError("cannot jump over walls")
        elif _check_wall_blocked(board_view, curpos_x, curpos_y, newpos_x, newpos_y):
            raise ValueError("cannot jump over walls")

        board_view[agent_id, curpos_x, curpos_y] = 0
        board_view[agent_id, newpos_x, newpos_y] = 1

    elif action_type == 1:  # Place wall horizontally
        if walls_remaining_view[agent_id] == 0:
            raise ValueError(f"no walls left for agent {agent_id}")
        if y == board_size-1:
            raise ValueError("cannot place wall on the edge")
        elif x == board_size-1:
            raise ValueError("right section out of board")
        elif board_view[2, x, y] or board_view[2, x+1, y]:
            raise ValueError("wall already placed")
        elif board_view[5, x, y]:
            raise ValueError("cannot create intersecting walls")
        board_view[2, x, y] = 1 + agent_id
        board_view[2, x + 1, y] = 1 + agent_id
        walls_remaining_view[agent_id] -= 1
        board_view[4, x, y] = 1

    elif action_type == 2:  # Place wall vertically
        if walls_remaining_view[agent_id] == 0:
            raise ValueError(f"no walls left for agent {agent_id}")
        if x == board_size-1:
            raise ValueError("cannot place wall on the edge")
        elif y == board_size-1:
            raise ValueError("right section out of board")
        elif board_view[3, x, y] or board_view[3, x, y+1]:
            raise ValueError("wall already placed")
        elif board_view[4, x, y]:
            raise ValueError("cannot create intersecting walls")
        board_view[3, x, y] = 1 + agent_id
        board_view[3, x, y + 1] = 1 + agent_id
        walls_remaining_view[agent_id] -= 1
        board_view[5, x, y] = 1

    elif action_type == 3:  # Rotate section
        if not _check_in_range(x, y, bottom_right=board_size-3):
            raise ValueError("rotation region out of board")
        elif walls_remaining_view[agent_id] < 2:
            raise ValueError(f"less than two walls left for agent {agent_id}")

        board_rotation(board, board_view, walls_remaining_view, agent_id, board_size, x, y)

    else:
        raise ValueError(f"invalid action_type: {action_type}")

    if action_type > 0:
        
        if not _check_path_exists(board_view, 0, board_size) or not _check_path_exists(board_view, 1, board_size):
            if action_type == 3:
                raise ValueError("cannot rotate to block all paths")
            else:
                raise ValueError("cannot place wall blocking all paths")

    return (board, walls_remaining, _check_wins(board_view, board_size))

cdef board_rotation(
    board,
    int [:,:,:] board_view,
    int [:] walls_remaining_view,
    int agent_id,
    int board_size,
    int x,
    int y):

    cdef int cx, cy, px, py

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

    return

def legal_actions(state, int agent_id, int board_size):
    
    cdef int dir_x, dir_y, action_type, next_pos_x, next_pos_y, cx, cy

    legal_actions_np = np.zeros((4, 9, 9), dtype=np.int_)
    now_pos = _agent_pos(state.board, agent_id, board_size)
    directions = ((0, -2), (-1, -1), (0, -1), (1, -1), (-2, 0), (-1, 0), (1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2))

    for dir_x, dir_y in directions:
        next_pos_x = now_pos[0] + dir_x
        next_pos_y = now_pos[1] + dir_y
        if _check_in_range(next_pos_x, next_pos_y, board_size):
            try:
                fast_step(state.board, state.walls_remaining, agent_id, (0, next_pos_x, next_pos_y), board_size)
            except:
                ...
            else:
                legal_actions_np[0, next_pos_x, next_pos_y] = 1
    for action_type in (1, 2):
        for cx in range(board_size-1):
            for cy in range(board_size-1):
                try:
                    fast_step(state.board, state.walls_remaining, agent_id, (action_type, cx, cy), board_size)
                except:
                    ...
                else:
                    legal_actions_np[action_type, cx, cy] = 1
    for cx in range(board_size-3):
        for cy in range(board_size-3):
            try:
                fast_step(state.board, state.walls_remaining, agent_id, (3, cx, cy), board_size)
            except:
                ...
            else:
                legal_actions_np[3, cx, cy] = 1
    return legal_actions_np

cdef _check_in_range(int pos_x, int pos_y, int bottom_right = 9):
    return (0 <= pos_x < bottom_right and 0 <= pos_y < bottom_right)

cdef _check_path_exists(int [:,:,:] board_view, int agent_id, int board_size):

    cdef int pos_x = -1, pos_y = -1
    cdef int i, j
    cdef int cnt = 0, tail = 0
    cdef there_x, there_y
    cdef int goal = (1-agent_id) * 8
    cdef int queue[81]
    cdef int [:,:] visited = np.zeros((board_size, board_size), dtype=np.int_)
    cdef int directions[4][2]
    if agent_id:
        directions[0][:] = [0, -1]
        directions[1][:] = [1, 0]
        directions[2][:] = [-1, 0]
        directions[3][:] = [0, 1]
    else:
        directions[0][:] = [0, 1]
        directions[1][:] = [1, 0]
        directions[2][:] = [-1, 0]
        directions[3][:] = [0, -1]

    for i in range(board_size):
        for j in range(board_size):
            if board_view[agent_id, i, j]:
                pos_x = i
                pos_y = j
                if pos_y == goal:   return True
                break
        if pos_x >= 0:   break
    
    queue[tail] = pos_x * board_size + pos_y
    tail += 1
    visited[pos_x, pos_y] = 1

    for i in range(board_size * board_size):
        if cnt == tail: break
        pos_x = queue[cnt] / board_size
        pos_y = queue[cnt] % board_size
        cnt += 1
        for j in range(4):
            there_x = pos_x + directions[j][0]
            there_y = pos_y + directions[j][1]
            if not _check_in_range(there_x, there_y, board_size):
                continue
            if visited[there_x][there_y]:
                continue
            if _check_wall_blocked(board_view, pos_x, pos_y, there_x, there_y):
                continue
            if there_y == goal:
                return True
            visited[there_x][there_y] = 1
            queue[tail] = there_x * board_size + there_y
            tail += 1

    return False

cdef _check_wall_blocked(int [:,:,:] board_view, int cx, int cy, int nx, int ny):
    cdef int i
    if nx > cx:
        for i in range(cx, nx):
            if board_view[3, i, cy]:
                return True
        return False
    if nx < cx:
        for i in range(nx, cx):
            if board_view[3, i, cy]:
                return True
        return False
    if ny > cy:
        for i in range(cy, ny):
            if board_view[2, cx, i]:
                return True
        return False
    if ny < cy:
        for i in range(ny, cy):
            if board_view[2, cx, i]:
                return True
        return False
    return False

cdef _check_wins(int [:,:,:] board_view, int board_size):
    cdef int i
    for i in range(board_size):
        if board_view[0, i, board_size-1]:
            return True
        if board_view[1, i, 0]:
            return True
    return False

cdef _agent_pos(int [:,:,:] board_view, int agent_id, int board_size):
    cdef int i, j
    for i in range(board_size):
        for j in range(board_size):
            if board_view[agent_id, i, j]:
                return (i, j)
    return -1