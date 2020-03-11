# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the GPL3 License.
# If a copy of the GPL3 was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/GPL-3.0

# author: Dorian Gouzou (github.com/jackred) <jackred@tuta.io>

from pso import PSO
import utility


def main():
    dim = 5
    ros = utility.Rosenbrock(dim)
    pso = PSO(dim, ros.evaluate, 2500, comparator=utility.minimise,
              min_bound=-5, max_bound=10)
    pso.run()
    print(pso.best_global_score, pso.best_position)


if __name__ == '__main__':
    main()
