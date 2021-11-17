from enum import Flag, auto

import numba
import numpy as np


class ThreePlayers(Flag):
    P1 = auto()
    P2 = auto()
    P3 = auto()

    # combinations
    P12 = P1 | P2
    P23 = P2 | P3
    P13 = P1 | P3

@numba.njit
def num_players():
    return 3

# @numba.njit
def get_zero_indexed_player_idx(player):
    return player.value.bit_length() - 1

@numba.njit
def next_player(player):
    if player == FourPlayers.P1: return FourPlayers.P2
    if player == FourPlayers.P2: return FourPlayers.P3
    if player == FourPlayers.P3: return FourPlayers.P1

@numba.njit
def is_union_player(player):
    return (
                player == FourPlayers.P12 or
                player == FourPlayers.P23 or
                player == FourPlayers.P13
        )

@numba.njit
def get_union_player_members(player):
        if player == FourPlayers.P12:
            return FourPlayers.P1, FourPlayers.P2
        if player == FourPlayers.P23:
            return FourPlayers.P2, FourPlayers.P3
        if player == FourPlayers.P13:
            return FourPlayers.P1, FourPlayers.P3
        return None


@numba.njit
def utility_func(atom_type_board):
    utilities = np.zeros(3, dtype=np.float32)
    for i in range(atom_type_board.shape[0]):
        for j in range(atom_type_board.shape[1]):
            p = atom_type_board[i, j]

            count = 0
            if p & FourPlayers.P1.value: count += 1
            if p & FourPlayers.P2.value: count += 1
            if p & FourPlayers.P3.value: count += 1

            if p & FourPlayers.P1.value: utilities[0] += 1.0 / count
            if p & FourPlayers.P2.value: utilities[1] += 1.0 / count
            if p & FourPlayers.P3.value: utilities[2] += 1.0 / count

    return utilities

Players = ThreePlayers