# -*- Mode: Python; tab-width: 8; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https:/w/opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import numpy as np
from pso_utility_functions import compute_neightbors
import visualize

INERTIA_START = 0.9
INERTIA_END = 0.4
COGNITIVE_TRUST = 2
SOCIAL_TRUST = 2


# non optimized number of operation
# example: computation of the worst_positions, should be in form neighbor and
# compute_neightbors
# optimized iff result are nice
def mmrpso(dim, fitness_function, min_bound, max_bound,
           velocity_fn, move,
           form_neighbor, init_particle, form_worst,
           max_iter, n_particle,
           cog_val=COGNITIVE_TRUST, soc_val=SOCIAL_TRUST,
           # worst_c_val=COGNITIVE_TRUST,
           worst_s_val=SOCIAL_TRUST,
           inertia_start=INERTIA_START, inertia_end=INERTIA_END):
    (positions,
     velocitys,
     scores,
     best_scores,
     best_positions) = init_particle(n_particle,
                                     dim,
                                     fitness_function,
                                     min_bound, max_bound)
    (neighbors,
     neighbors_best_scores,
     neighbors_best_positions) = form_neighbor(best_scores, best_positions,
                                               dim)
    worst_positions = np.array([
        positions[neighbors[i]][scores[neighbors[i]].argmax()]
        for i in range(n_particle)
    ])
    worst_array = form_worst(n_particle)
    best_score_swarm = min(best_scores)
    i = 0
    while i < max_iter and best_score_swarm > 1e-08:
        worst_v = max(0, worst_s_val - (worst_s_val - 0) / (max_iter//2) * i)
        # if i == max_iter // 2:
        #     worst_array = np.array([False] * n_particle)
        print("%d / %d      " % (i, max_iter), end="\r")
        if i % 10 == 0:
            print(worst_s_val, "<>", i, "->", "b", best_score_swarm, best_positions[best_scores.argmin()])
            visualize.plot_data(positions, worst_array, min_bound, max_bound)
        inertia = inertia_start - (inertia_start - inertia_end) / max_iter * i
        for idx in range(n_particle):
            velocitys[idx] = velocity_fn(worst_array[idx],
                                         dim, cog_val, soc_val,
                                         # worst_c_val,
                                         worst_v,
                                         inertia,
                                         positions[idx], velocitys[idx],
                                         best_positions[idx],
                                         neighbors_best_positions[idx],
                                         worst_positions[idx])
            (positions[idx],
             velocitys[idx]) = move(worst_array[idx],
                                    positions[idx],
                                    velocitys[idx],
                                    min_bound, max_bound,
                                    positions[scores[neighbors[idx]].argmin()],
                                    np.max(positions[neighbors[idx]][worst_array[neighbors[idx]]], axis=0))
            scores[idx] = fitness_function(positions[idx])
        for idx in range(n_particle):
            if scores[idx] <= best_scores[idx]:
                best_scores[idx] = scores[idx]
                best_positions[idx] = positions[idx]
        worst_positions = np.array([
            positions[neighbors[i]][scores[neighbors[i]].argmax()]
            for i in range(n_particle)
        ])
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
        i += 1
    return best_score_swarm, best_positions[best_scores.argmin()]
