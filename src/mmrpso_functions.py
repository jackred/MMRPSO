# -*- Mode: Python; tab-width: 8; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import numpy as np
from math import sqrt


def gravity_center_equation_w(dimension, position,
                              worst_position, social_trust):
    res = np.empty(dimension)
    for i in range(dimension):
        pi = position[i] \
            + social_trust * np.random.uniform(0, 1) \
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


def velocity_w(dimension, min_bound, max_bound,
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
            + cognitive_trust * np.random.uniform(0, 1) \
            * (best_pos[i] - pos[i])
        li = pos[i] \
            + social_trust * np.random.uniform(0, 1) \
            * (neighbors_best_pos[i] - pos[i])
        wpi = worst_s_val * (pos[i]
                             + social_trust * np.random.uniform(0, 1)
                             * (worst_pos[i] - pos[i]))
        res[i] = (pos[i] + pi + li - wpi) / (3 + worst_s_val)
    return res


def velocity(dimension, min_bound, max_bound,
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
    res = np.empty(n_particle)
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
