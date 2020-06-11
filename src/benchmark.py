# -*- Mode: Python; tab-width: 8; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

from cec2013single.cec2013 import Benchmark
import numpy as np


# 1 3 8 9 15 21 25
class TestBenchmark(Benchmark):
    def exec_function(self, fn_number, argument_vector):
        arg_np = np.array(argument_vector)
        return self.get_function(fn_number)(arg_np)

    # unimodal
    def sphere(self, argument_vector):
        return self.exec_function(1, argument_vector)

    def bent_cigar_rotated(self, argument_vector):
        return self.exec_function(3, argument_vector)

    # multi modal
    def ackley(self, argument_vector):
        return self.exec_function(8, argument_vector)

    def weierstrass_rotated(self, argument_vector):
        return self.exec_function(9, argument_vector)

    def schwefel(self, argument_vector):
        return self.exec_function(15, argument_vector)

    def expanded_schaffer_f6(self, argument_vector):
        return self.exec_function(20, argument_vector)

    # composite
    def composite_f1(self, argument_vector):
        return self.exec_function(21, argument_vector)

    def composite_f2(self, argument_vector):
        return self.exec_function(22, argument_vector)

    def composite_f8(self, argument_vector):
        return self.exec_function(28, argument_vector)
