# -*- Mode: Python; tab-width: 8; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import pso_simple as pso
import pso_simple_functions as pso_functions
from benchmark import TestBenchmark

bench = TestBenchmark()

dimension = 50
fn_number = 21
fitness_function = bench.composite_f1
info = bench.get_info(fn_number)
lower = info["lower"]
upper = info["upper"]

velocity_function = pso_functions.velocity_2011
form_neighborhood = pso_functions.ring_2
init_particle = pso.make_init_particle(pso_functions.init_position,
                                       pso_functions.init_velocity_2011)
move = pso_functions.move_2011

max_iter = 10000*dimension
n_particle = 40

score, position = pso.pso(dimension, fitness_function, lower, upper,
                          velocity_function, move,
                          form_neighborhood, init_particle,
                          max_iter, n_particle)
print("best", score, "at", position)
