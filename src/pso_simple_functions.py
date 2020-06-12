# -*- Mode: Python; tab-width: 8; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>


import numpy as np
from math import sqrt, ceil, floor


# PSO 2007

def velocity_2007(dimension, min_bound, max_bound,
                  cognitive_trust, social_trust, inertia,
                  position, velocity, best_position, neighbors_best_position):
    res = np.empty(dimension)
    for i in range(dimension):
        res[i] = (inertia * velocity[i]
                  + np.random.uniform(0, 1) * cognitive_trust
                  * (best_position[i] - position[i])
                  + np.random.uniform(0, 1) * social_trust
                  * (neighbors_best_position[i] - position[i]))
    return res


def init_velocity_2007(dimension, min_bound, max_bound, position):
    res = np.empty(dimension)
    for i in range(dimension):
        res[i] = (np.random.uniform(min_bound, max_bound) - i) / 2
    return res


def move_2007(position, velocity, min_bound, max_bound):
    tmp = position + velocity
    for i in range(len(tmp)):
        if tmp[i] < min_bound:
            position[i] = min_bound
            velocity[i] *= 0.0
        elif tmp[i] > max_bound:
            position[i] = max_bound
            velocity[i] *= 0.0
        else:
            position[i] = tmp[i]
    return position, velocity


# PSO 2011

def gravity_center_equation(dimension, position,
                            best_position, neighbors_best_position,
                            cognitive_trust, social_trust):
    res = np.empty(dimension)
    for i in range(dimension):
        pi = position[i] \
            + cognitive_trust * np.random.uniform(0, 1) \
            * (best_position[i] - position[i])
        li = position[i] \
            + social_trust * np.random.uniform(0, 1) \
            * (neighbors_best_position[i] - position[i])
        res[i] = (position[i] + pi + li) / 3
    return res


def dist(a, b):
    return sqrt(sum([(a[i] - b[i]) ** 2 for i in range(len(a))]))


def generate_point_in_sphere(g, position, dimension):
    radius = dist(g, position)
    U = pow(np.random.random(), 1/dimension)
    x = np.random.uniform(-1, 1, size=dimension)
    magnitude = sqrt(sum((x + 0.000000001) ** 2))
    return ((x * U) / magnitude) * radius + g


def velocity_2011(dimension, min_bound, max_bound,
                  cognitive_trust, social_trust, inertia,
                  position, velocity, best_position, neighbors_best_position):
    g = gravity_center_equation(dimension, position, best_position,
                                neighbors_best_position,
                                cognitive_trust, social_trust)
    position_prime = generate_point_in_sphere(g, position, dimension)
    res = np.empty(dimension)
    for i in range(dimension):
        res[i] = inertia * velocity[i] + position_prime[i] - position[i]
    return res


def init_velocity_2011(dimension, min_bound, max_bound, position):
    res = np.empty(dimension)
    for i in range(dimension):
        res = np.random.uniform(min_bound-position[i], max_bound - position[i])
    return res


def move_2011(position, velocity, min_bound, max_bound):
    tmp = position + velocity
    for i in range(len(tmp)):
        if tmp[i] < min_bound:
            position[i] = min_bound
            velocity[i] *= 0.5
        elif tmp[i] > max_bound:
            position[i] = max_bound
            velocity[i] *= 0.5
        else:
            position[i] = tmp[i]
    return position, velocity


# PSO

def init_position(dimension, min_bound, max_bound):
    res = np.empty(dimension)
    for i in range(dimension):
        res[i] = np.random.uniform(min_bound, max_bound)
    return res


def make_list(low, high, n):
    if low < high:
        return list(range(low, high+1))
    else:
        res = list(range(low, n))
        res.extend(list(range(0, high+1)))
        return res


def form_neighborhood_ring(n_neighbor, best_scores, best_positions, dimension):
    n_particle = len(best_scores)
    neighbors = np.empty(shape=(n_particle, n_neighbor+1), dtype=int)
    neighbors_best_scores = np.empty(n_particle)
    neighbors_best_positions = np.empty(shape=(n_particle, dimension))
    n_low = -floor(n_neighbor / 2)
    n_high = ceil(n_neighbor / 2)
    for i in range(n_particle):
        idx = make_list((i+n_low) % n_particle, (i+n_high) % n_particle,
                        n_particle)
        neighbors[i] = np.array(idx)
        idx_min = best_scores[idx].argmin()
        neighbors_best_positions[i] = best_positions[idx][idx_min]
        neighbors_best_scores[i] = best_scores[idx][idx_min]
    return (neighbors, neighbors_best_scores, neighbors_best_positions)


def ring_2(*args):
    return form_neighborhood_ring(2, *args)


def form_neighborhood_dense(best_scores, best_positions):
    n_particle = len(best_scores)
    neighbors = np.full((n_particle, n_particle), range(n_particle), dtype=int)
    idx = best_scores.argmin()
    neighbors_best_positions = np.full(n_particle, best_positions[idx])
    neighbors_best_scores = np.full(n_particle, best_scores[idx])
    return (neighbors, neighbors_best_scores, neighbors_best_positions)
