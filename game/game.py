import numba
import numpy as np

import config
# from four_players import *

utility_func = config.utility_func
decay_probs = config.decay_probs
pm = config.pm

@numba.njit
def allowed_moves_jitted(atom_type_board, player_value):
    out = np.zeros((atom_type_board.shape[0], atom_type_board.shape[1]), dtype=np.bool8)
    player_idx = int(np.log2(player_value))

    for i in range(atom_type_board.shape[0]):
        for j in range(atom_type_board.shape[1]):
            atom_type = atom_type_board[i, j]
            if atom_type == 0:
                out[i, j] = True
            if atom_type & player_value:
                out[i, j] = True

    return out


def allowed_moves(state, player):
    atom_type_board = state.atom_type
    player_value = player.value
    out = allowed_moves_jitted(atom_type_board, player_value)
    return out.nonzero()


@numba.jit
def board_utility(atom_type_board):
    global utility_func
    return utility_func(atom_type_board)


@numba.njit
def is_terminal(atom_type_board):
    # A board is terminal if there is only 1 type of atom left
    seen = set()
    for i in range(atom_type_board.shape[0]):
        for j in range(atom_type_board.shape[1]):
            p = atom_type_board[i, j]
            if p > 0:
                seen.add(p)

    if len(seen) > 1:
        return False
    else:
        return True


@numba.jit
def do_move(state, i, j, player):
    # If the atom at i,j is a union type, make sure we are placing the value of the union type
    atom_type = state.atom_type[i, j]
    if atom_type == 0:
        atom_type = player.value

    state.place_atom(i, j, atom_type)
    queue = [(i, j)]

    k = 0
    while len(queue) > 0:
        i, j = queue.pop(-1)
        if state.check_explosion(i, j):
            affected_neighbours = state.explode(i, j)
            # check if someone has won
            # utilities = board_utility(state.atom_type)
            if is_terminal(state.atom_type):
                return state

            queue.extend(affected_neighbours)
            k += 1

        if k >= 100:
            # print(print_board(state))
            raise Exception('Infinite detected')

    return state

@numba.njit
def game_step(state, player: pm.Players, move):
    global decay_probs
    
    if is_terminal(state.atom_type):
        return state, player, True

    atom_type = state.atom_type[move[0], move[1]]
    player_for_atom_type = pm.map_value_to_enum(atom_type)
    random_player = player
    if player_for_atom_type is not None and pm.is_union_player(player_for_atom_type):
        choice = 0
        with numba.objmode(choice='int64'):
            choice = np.random.choice(3, 1, p=decay_probs)[0]
        
        if choice != 0:
            members = pm.get_union_player_members(player_for_atom_type)
            random_player = members[choice - 1]
            state.clear_cell(move[0], move[1])
            state.place_atom(move[0], move[1], random_player.value)
            # state.atom_type[move[0], move[1]] = random_player.value

    state = do_move(state, move[0], move[1], random_player)
    # Determine who gets to play next. Look for someone having non zero utility
    utilities = board_utility(state.atom_type)
    player = pm.next_player(player)
    while utilities[pm.get_zero_indexed_player_idx(player)] == 0:
        player = pm.next_player(player)

    return state, player, is_terminal(state.atom_type)
