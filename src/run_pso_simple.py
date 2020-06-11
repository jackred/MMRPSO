# -*- Mode: Python; tab-width: 8; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import sys
import pso_simple as pso
import pso_simple_functions as pso_functions
from benchmark import TestBenchmark

bench = TestBenchmark()

if len(sys.argv) != 3:
    exit("need 2 arguments: run_pso_simple.py dim fn_num")

dimension = int(sys.argv[1])
fn_number = int(sys.argv[2])
fitness_function = bench.lambda_function(fn_number)
info = bench.get_info(fn_number)
lower = info["lower"]
upper = info["upper"]

velocity_function = pso_functions.velocity_2011
form_neighborhood = pso_functions.ring_2
init_particle = pso.make_init_particle(pso_functions.init_position,
                                       pso_functions.init_velocity_2011)
move = pso_functions.move_2011

max_iter = 13500*dimension
n_particle = 40

res = []
print("function %d in dimension %d" % (fn_number, dimension))
for i in range(10):
    score, position = pso.pso(dimension, fitness_function, lower, upper,
                              velocity_function, move,
                              form_neighborhood, init_particle,
                              max_iter, n_particle)
    print("i->", i, ": best", score, "at", position)
    res.append(score)
print("average: ", sum(res) / len(res))
