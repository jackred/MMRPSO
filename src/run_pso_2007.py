# -*- Mode: Python; tab-width: 8; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import sys
from math import sqrt
import pso_simple as pso
import pso_simple_functions as pso_functions
from pso_utility_functions import make_init_particle
from benchmark import TestBenchmark


def round8(a):
    return 0.0 if a < 1e-08 else a


bench = TestBenchmark()

if len(sys.argv) != 5:
    exit("need 4 arguments: run_pso_2007.py dim fn_num ignore_same"
         + "nb_neighbor")

dimension = int(sys.argv[1])
fn_number = int(sys.argv[2])
ignore_same = bool(int(sys.argv[3]))
nb_neighbor = int(sys.argv[4])

fitness_function = bench.lambda_function(fn_number)
info = bench.get_info(fn_number)
lower = info["lower"]
upper = info["upper"]

velocity_function = pso_functions.velocity_2007_ignore if ignore_same \
    else pso_functions.velocity_2007
form_neighborhood = pso_functions.make_ring(nb_neighbor)
init_particle = make_init_particle(pso_functions.init_position,
                                   pso_functions.init_velocity_2007)
move = pso_functions.move_2007

max_iter = 10000*dimension
n_particle = 40

res = []
print("function %d in dimension %d" % (fn_number, dimension))

if dimension in [2, 10]:
    nb = 51
elif dimension in [30, 50]:
    nb = 30
elif dimension == 100:
    nb = 10

print("%s runs" % nb)
for i in range(nb):
    score, position = pso.pso(dimension, fitness_function, lower, upper,
                              velocity_function, move,
                              form_neighborhood, init_particle,
                              max_iter, n_particle)
    print("i->", i, ": best", round8(score), "at", position)
    res.append(round8(score))
res.sort()
average = sum(res) / len(res)
print("min", min(res))
print("max", max(res))
print("median", res[len(res)//2])
print("average: ", average)
print("std", sqrt((1/(len(res) - 1)) * sum([(i - average) ** 2for i in res])))
print("function %d in dimension %d" % (fn_number, dimension))
print("pso 2007, ignore same: %r" % ignore_same)
