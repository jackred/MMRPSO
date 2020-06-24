# -*- Mode: Python; tab-width: 8; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https:/w/opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import numpy as np
from pso_utility_functions import compute_neightbors

INERTIA_START = 0.9
INERTIA_END = 0.4
COGNITIVE_TRUST = 2
SOCIAL_TRUST = 2


def mmrpso(dim, fitness_function, min_bound, max_bound,
           velocity_function, velocity_function_w, move,
           form_neighbor, init_particle,
           max_iter, n_particle,
           cog_val=COGNITIVE_TRUST, soc_val=SOCIAL_TRUST,
           worst_c_val=COGNITIVE_TRUST, worst_s_val=SOCIAL_TRUST,
           inertia_start=INERTIA_START, inertia_end=INERTIA_END):
    (pos_opti,
     velo_opti,
     scores_opti,
     best_scores,
     best_pos) = init_particle(n_particle, dim, fitness_function,
                               min_bound, max_bound)
    (pos_worst,
     velo_worst,
     scores_worst,
     worst_scores,
     worst_pos) = init_particle(n_particle, dim, fitness_function,
                                min_bound, max_bound)
    (neighbors_opti,
     neighbors_best_scores,
     neighbors_best_pos) = form_neighbor(best_scores, best_pos, dim)
    (neighbors_worst,
     neighbors_worst_scores,
     neighbors_worst_pos) = form_neighbor(worst_scores, worst_pos, dim)
    best_score_swarm = min(best_scores)
    worst_score_swarm = max(worst_scores)
    i = 0
    while i < max_iter and best_score_swarm > 1e-08:
        print("%d / %d        " % (i, max_iter),  end="\r")
        if i % 200 == 0:
            print(i, "->", "b", best_score_swarm, best_pos[best_scores.argmin()],
                  "w", worst_score_swarm, worst_pos[worst_scores.argmax()])
        inertia = inertia_start - (inertia_start - inertia_end) / max_iter * i
        for idx in range(n_particle):
            velo_worst[idx] = velocity_function_w(dim, min_bound, max_bound,
                                                  cog_val, soc_val,
                                                  inertia,
                                                  pos_worst[idx],
                                                  velo_worst[idx],
                                                  worst_pos[idx],
                                                  neighbors_worst_pos[idx])
            velo_opti[idx] = velocity_function(dim, min_bound, max_bound,
                                               cog_val, soc_val,
                                               worst_c_val, worst_s_val,
                                               inertia,
                                               pos_opti[idx], velo_opti[idx],
                                               best_pos[idx],
                                               neighbors_best_pos[idx],
                                               worst_pos[idx],
                                               neighbors_worst_pos[idx])
            pos_opti[idx], velo_opti[idx] = move(pos_opti[idx],
                                                 velo_opti[idx],
                                                 min_bound, max_bound)
            pos_worst[idx], velo_worst[idx] = move(pos_worst[idx],
                                                   velo_opti[idx],
                                                   min_bound, max_bound)
            scores_opti[idx] = fitness_function(pos_opti[idx])
            scores_worst[idx] = fitness_function(pos_worst[idx])
        for idx in range(n_particle):
            if scores_opti[idx] <= best_scores[idx]:
                best_scores[idx] = scores_opti[idx]
                best_pos[idx] = pos_opti[idx]
            if scores_worst[idx] >= worst_scores[idx]:
                worst_scores[idx] = scores_worst[idx]
                worst_pos[idx] = pos_worst[idx]
        tmp_min = min(best_scores)
        if tmp_min < best_score_swarm:
            best_score_swarm = tmp_min
        tmp_max = max(worst_scores)
        if tmp_max > worst_score_swarm:
            worst_score_swarm = tmp_max
        (neighbors_best_scores,
         neighbors_best_positions) = compute_neightbors(neighbors_opti,
                                                        best_scores,
                                                        best_pos)
        (neighbors_worst_scores,
         neighbors_worst_positions) = compute_neightbors(neighbors_worst,
                                                         worst_scores,
                                                         worst_pos,
                                                         np.argmax)
        i += 1
    return (best_score_swarm, best_pos[best_scores.argmin()],
            worst_score_swarm, worst_pos[worst_scores.argmax()])
