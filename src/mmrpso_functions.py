# -*- Mode: Python; tab-width: 8; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import numpy as np
from math import sqrt
import pso_simple_functions


def gravity_center_equation_w(dimension, position,
                              worst_position, social_trust):
    res = np.empty(dimension)
    for i in range(dimension):
        pi = position[i] \
            + social_trust \
            * (worst_position[i] - position[i])
        res[i] = (position[i] + pi) / 2
    return res


def dist(a, b):
    return sqrt(sum([(a[i] - b[i]) ** 2 for i in range(len(a))]))


def generate_point_in_sphere(g, position, dimension):
    radius = dist(g, position)
    U = pow(np.random.random(), 1/dimension)
    x = np.random.uniform(-1, 1, size=dimension)
    magnitude = sqrt(sum((x + 0.000000001) ** 2))
    return ((x * U) / magnitude) * radius + g


def velocity_w(dimension,
               cognitive_trust, social_trust,
               # worst_c_val,
               worst_s_val,
               inertia,
               pos, velocity, best_pos, neighbors_best_pos,
               worst_pos):
    g = gravity_center_equation_w(dimension, pos, worst_pos,
                                  social_trust)
    pos_prime = generate_point_in_sphere(g, pos, dimension)
    res = np.empty(dimension)
    for i in range(dimension):
        res[i] = inertia * velocity[i] + pos_prime[i] - pos[i]
    return res


def gravity_center_equation(dimension, pos, best_pos, neighbors_best_pos,
                            cognitive_trust, social_trust,
                            # worst_c_val,
                            worst_s_val,
                            worst_pos):
    res = np.empty(dimension)
    for i in range(dimension):
        pi = pos[i] \
            + cognitive_trust \
            * (best_pos[i] - pos[i])
        li = pos[i] \
            + social_trust \
            * (neighbors_best_pos[i] - pos[i])
        wpi = worst_s_val * (pos[i]
                             + social_trust
                             * (worst_pos[i] - pos[i]))
        res[i] = (pos[i] + pi + li - wpi) / (3 + worst_s_val)
    return res


def velocity(dimension,
             cognitive_trust, social_trust,
             # worst_c_val,
             worst_s_val,
             inertia,
             pos, velocity, best_pos, neighbors_best_pos,
             worst_pos):
    g = gravity_center_equation(dimension, pos, best_pos,
                                neighbors_best_pos,
                                cognitive_trust, social_trust,
                                # worst_c_val,
                                worst_s_val,
                                worst_pos)
    pos_prime = generate_point_in_sphere(g, pos, dimension)
    res = np.empty(dimension)
    for i in range(dimension):
        res[i] = inertia * velocity[i] + pos_prime[i] - pos[i]
    return res


def velocity_both(isWorst, *args):
    if isWorst:
        return velocity_w(*args)
    else:
        return velocity(*args)


def form_worst(cluster_size, n_particle):
    res = np.empty(n_particle, dtype=bool)
    n_good = round(cluster_size * 0.6)
    n_worst = cluster_size - n_good
    i = 0
    while i < n_particle:
        for j in range(n_good):
            res[i+j] = False
        i += n_good
        for j in range(n_worst):
            res[i+j] = True
        i += n_worst
    return res


def form_5_3(n_particle):
    return form_worst(8, n_particle)


def form_3_2(n_particle):
    return form_worst(5, n_particle)


def move_both(isWorst, position, velocity, min_bound, max_bound, best_position,
              dist_neighbors_pos):
    if isWorst:
        dist_neighbors_pos = np.full(len(position), 10)
        # cluster_min_bound = [max(min_bound[i],
        #                          best_position[i] - dist_neighbors_pos[i])
        #                      for i in range(len(min_bound))]
        # cluster_max_bound = [min(max_bound[i],
        #                          best_position[i] + dist_neighbors_pos[i])
        #                      for i in range(len(max_bound))]
        
        return pso_simple_functions.move_2011(position, velocity,
                                              min_bound,
                                              max_bound)
    else:
        return pso_simple_functions.move_2011(position, velocity,
                                              min_bound, max_bound)
