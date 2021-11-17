from enum import Flag, auto

import numba
import numpy as np


class FourPlayers(Flag):
    P1 = auto()
    P2 = auto()
    P3 = auto()
    P4 = auto()

    # combinations
    P12 = P1 | P2
    P13 = P1 | P3
    P14 = P1 | P4
    P23 = P2 | P3
    P24 = P2 | P4
    P34 = P3 | P4

# mapping_dict = numba.typed.Dict.empty(
#     key_type=numba.types.int64, value_type=numba.typeof(FourPlayers.P1)
# )
# for p in FourPlayers:
#     mapping_dict[p.value] = p

@numba.jit
def map_value_to_enum(value):
    if value == FourPlayers.P1.value: return FourPlayers.P1
    if value == FourPlayers.P2.value: return FourPlayers.P2
    if value == FourPlayers.P3.value: return FourPlayers.P3
    if value == FourPlayers.P4.value: return FourPlayers.P4

    if value == FourPlayers.P12.value: return FourPlayers.P12
    if value == FourPlayers.P13.value: return FourPlayers.P13
    if value == FourPlayers.P14.value: return FourPlayers.P14
    if value == FourPlayers.P23.value: return FourPlayers.P23
    if value == FourPlayers.P24.value: return FourPlayers.P24
    if value == FourPlayers.P34.value: return FourPlayers.P34
    return None

@numba.njit
def num_players():
    return 4

@numba.njit
def get_zero_indexed_player_idx(player):
    return int(np.log2(player.value))
    # return player.value.bit_length() - 1

@numba.njit
def next_player(player):
    if player == FourPlayers.P1: return FourPlayers.P2
    if player == FourPlayers.P2: return FourPlayers.P3
    if player == FourPlayers.P3: return FourPlayers.P4
    if player == FourPlayers.P4: return FourPlayers.P1

@numba.njit
def is_union_player(player):
    return (
                player == FourPlayers.P12 or
                player == FourPlayers.P13 or
                player == FourPlayers.P14 or
                player == FourPlayers.P23 or
                player == FourPlayers.P23 or
                player == FourPlayers.P34
        )

@numba.njit
def get_union_player_members(player):
        if player == FourPlayers.P12:
            return FourPlayers.P1, FourPlayers.P2
        if player == FourPlayers.P13:
            return FourPlayers.P1, FourPlayers.P3
        if player == FourPlayers.P14:
            return FourPlayers.P1, FourPlayers.P4
        if player == FourPlayers.P23:
            return FourPlayers.P2, FourPlayers.P3
        if player == FourPlayers.P24:
            return FourPlayers.P2, FourPlayers.P4
        if player == FourPlayers.P34:
            return FourPlayers.P3, FourPlayers.P4
        return None


@numba.njit
def utility_func(atom_type_board):
    utilities = np.zeros(4, dtype=np.float32)
    for i in range(atom_type_board.shape[0]):
        for j in range(atom_type_board.shape[1]):
            p = atom_type_board[i, j]

            count = 0
            if p & FourPlayers.P1.value: count += 1
            if p & FourPlayers.P2.value: count += 1
            if p & FourPlayers.P3.value: count += 1
            if p & FourPlayers.P4.value: count += 1

            if p & FourPlayers.P1.value: utilities[0] += 1.0 / count
            if p & FourPlayers.P2.value: utilities[1] += 1.0 / count
            if p & FourPlayers.P3.value: utilities[2] += 1.0 / count
            if p & FourPlayers.P4.value: utilities[3] += 1.0 / count

    return utilities

Players = FourPlayers