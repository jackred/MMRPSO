# -*- Mode: Python; tab-width: 8; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import numpy as np
from math import sqrt


def gravity_center_equation_w(dimension, position,
                              worst_position, neighbors_worst_position,
                              cognitive_trust, social_trust):
    res = np.empty(dimension)
    for i in range(dimension):
        pi = position[i] \
            + cognitive_trust * np.random.uniform(0, 1) \
            * (worst_position[i] - position[i])
        li = position[i] \
            + social_trust * np.random.uniform(0, 1) \
            * (neighbors_worst_position[i] - position[i])
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


def velocity_w(dimension, min_bound, max_bound,
               cognitive_trust, social_trust, inertia,
               pos, velocity, worst_pos, neighbors_worst_pos):
    g = gravity_center_equation_w(dimension, pos, worst_pos,
                                  neighbors_worst_pos,
                                  cognitive_trust, social_trust)
    pos_prime = generate_point_in_sphere(g, pos, dimension)
    res = np.empty(dimension)
    for i in range(dimension):
        res[i] = inertia * velocity[i] + pos_prime[i] - pos[i]
    return res


def gravity_center_equation(dimension, pos, best_pos, neighbors_best_pos,
                            cognitive_trust, social_trust,
                            worst_c_val, worst_s_val,
                            worst_pos, neighbors_worst_pos):
    res = np.empty(dimension)
    cog = 0.015
    soc = 0.015
    divide = 3 + cog + soc
    for i in range(dimension):
        pi = pos[i] \
            + cognitive_trust * np.random.uniform(0, 1) \
            * (best_pos[i] - pos[i])
        li = pos[i] \
            + social_trust * np.random.uniform(0, 1) \
            * (neighbors_best_pos[i] - pos[i])
        wpi = pos[i] \
            + worst_c_val * np.random.uniform(0, 1) \
            * (worst_pos[i] - pos[i])
        wli = pos[i] \
            + worst_s_val * np.random.uniform(0, 1) \
            * (neighbors_worst_pos[i] - pos[i])
        res[i] = (pos[i] + pi + li - wpi*cog - wli*soc) / (divide)
    return res


def velocity(dimension, min_bound, max_bound,
             cognitive_trust, social_trust,
             worst_c_val, worst_s_val,
             inertia,
             pos, velocity, best_pos, neighbors_best_pos,
             worst_pos, neighbors_worst_pos):
    g = gravity_center_equation(dimension, pos, best_pos,
                                neighbors_best_pos,
                                cognitive_trust, social_trust,
                                worst_c_val, worst_s_val,
                                worst_pos, neighbors_worst_pos)
    pos_prime = generate_point_in_sphere(g, pos, dimension)
    res = np.empty(dimension)
    for i in range(dimension):
        res[i] = inertia * velocity[i] + pos_prime[i] - pos[i]
    return res
