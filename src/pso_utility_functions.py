# -*- Mode: Python; tab-width: 8; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
import numpy as np


def compute_neightbors(neighbors, best_scores, best_positions, f=np.argmin):
    n_particle = len(best_scores)
    res_scores = np.empty(n_particle)
    res_positions = np.empty(shape=(n_particle, len(best_positions[0])))
    for i in range(n_particle):
        tmp = best_scores[neighbors[i]]
        idx = f(tmp)
        res_scores[i] = tmp[idx]
        res_positions[i] = best_positions[neighbors[i]][idx]
    return res_scores, res_positions


def init_particle(init_position, init_velocity,
                  n_particle, dimension, fitness_function,
                  min_bound, max_bound):
    positions = np.empty(shape=(n_particle, dimension))
    velocitys = np.empty(shape=(n_particle, dimension))
    scores = np.empty(n_particle)
    best_scores = np.empty(n_particle)
    best_positions = np.empty(shape=(n_particle, dimension))
    for i in range(n_particle):
        positions[i] = init_position(dimension, min_bound, max_bound)
        velocitys[i] = init_velocity(dimension, min_bound, max_bound,
                                     positions[i])
        scores[i] = fitness_function(positions[i])
        best_scores[i] = scores[i]
        best_positions[i] = positions[i]
    return positions, velocitys, scores, best_scores, best_positions


def make_init_particle(init_position, init_velocity):
    return lambda *args: init_particle(init_position, init_velocity, *args)
