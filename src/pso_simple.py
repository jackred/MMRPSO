# -*- Mode: Python; tab-width: 8; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import numpy as np

INERTIA_START = 0.9
INERTIA_END = 0.4
COGNITIVE_TRUST = 2
SOCIAL_TRUST = 2


def compute_neightbors(neighbors, best_scores, best_positions):
    n_particle = len(best_scores)
    res_scores = np.empty(n_particle)
    res_positions = np.empty(shape=(n_particle, len(best_positions[0])))
    for i in range(n_particle):
        tmp = best_scores[neighbors[i]]
        idx = tmp.argmin()
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


def pso(dimension, fitness_function, min_bound, max_bound,
        velocity_function, move,
        form_neighborhood, init_particle,
        max_iter, n_particle,
        cognitive_trust=COGNITIVE_TRUST, social_trust=SOCIAL_TRUST,
        inertia_start=INERTIA_START, inertia_end=INERTIA_END):
    (positions,
     velocitys,
     scores,
     best_scores,
     best_positions) = init_particle(n_particle,
                                     dimension,
                                     fitness_function,
                                     min_bound, max_bound)
    (neighbors,
     neighbors_best_scores,
     neighbors_best_positions) = form_neighborhood(best_scores, best_positions,
                                                   dimension)
    best_score_swarm = min(best_scores)
    for i in range(max_iter):
        print("%d / %d      " % (i, max_iter), end="\r")
        inertia = inertia_start - (inertia_start - inertia_end) / max_iter * i
        for idx in range(n_particle):
            velocitys[idx] = velocity_function(dimension, min_bound, max_bound,
                                               cognitive_trust, social_trust,
                                               inertia,
                                               positions[idx], velocitys[idx],
                                               best_positions[idx],
                                               neighbors_best_positions[idx])
            positions[idx], velocitys[idx] = move(positions[idx],
                                                  velocitys[idx],
                                                  min_bound, max_bound)
            scores[idx] = fitness_function(positions[idx])
            if scores[idx] < best_scores[idx]:
                best_scores[idx] = scores[idx]
                best_positions[idx] = positions[idx]
        tmp_min = min(best_scores)
        if tmp_min < best_score_swarm:
            best_score_swarm = tmp_min
            # (neighbors,
            #  neighbors_best_scores,
            #  neighbors_best_positions) = compute_value_swarm(n_particle,
            #                                                  form_neighborhood,
            #                                                  n_neighbor)
        (neighbors_best_scores,
         neighbors_best_positions) = compute_neightbors(neighbors, best_scores,
                                                        best_positions)
    return best_score_swarm, best_positions[best_scores.argmin()]
