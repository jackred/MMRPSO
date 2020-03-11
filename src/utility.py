# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the GPL3 License.
# If a copy of the GPL3 was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/GPL-3.0

# author: Dorian Gouzou (github.com/jackred) <jackred@tuta.io>

import random


def minimise(a, b):
    return a <= b


def maximise(a, b):
    return a >= b


class Rosenbrock:
    def __init__(self, dimension=2):
        self.max_bound = 10
        self.min_bound = -5
        self.dimension = dimension

    def generate_random(self):
        return [random.uniform(self.min_bound, self.max_bound)
                for _ in range(self.dimension)]

    def evaluate(self, xx):
        d = len(xx)
        int_sum = 0
        for i in range(d-1):
            xi = xx[i]
            xnext = xx[i+1]
            new = 100*(xnext-xi**2)**2 + (xi-1)**2
            int_sum = int_sum + new
        y = int_sum
        return y


class Sphere:
    def __init__(self, dimension=3):
        self.max_bound = 5.12
        self.min_bound = -5.12
        self.dimension = dimension

    def generate_random(self):
        return [random.uniform(self.min_bound, self.max_bound)
                for _ in range(self.dimension)]

    def evaluate(self, xx):
        d = len(xx)
        int_sum = 0
        for i in range(d):
            xi = xx[i]
            int_sum += xi ** 2
        return int_sum
