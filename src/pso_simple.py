# -*- Mode: Python; tab-width: 8; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
from pso_utility_functions import compute_neightbors


INERTIA_START = 0.9
INERTIA_END = 0.4
COGNITIVE_TRUST = 2
SOCIAL_TRUST = 2


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
    i = 0
    while i < max_iter and best_score_swarm > 1e-08:
        print("%d / %d      " % (i, max_iter), end="\r")
        if i % 200 == 0:
            print(i, "->", "b", best_score_swarm, best_positions[best_scores.argmin()])
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
        for idx in range(n_particle):
            if scores[idx] <= best_scores[idx]:
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
        i += 1
    return best_score_swarm, best_positions[best_scores.argmin()]
