# -*- Mode: Python; tab-width: 8; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import numpy as np
import matplotlib.pyplot as plt
import pso_simple_functions as pso
import mmrpso_functions as mmrpso


position = np.array([0, 0])
best_position = np.array([6.5, 0])
neighbors_best_position = np.array([0, 1])
worst_position = np.array([1.5, 3])
neighbors_worst_position = np.array([-4, 0])

C_COG = 2
C_SOC = 2
WC_COG = 1
WC_SOC = 1
DIMENSION = 2

N = 1400

gs = mmrpso.gravity_center_equation(DIMENSION, position,
                                    best_position, neighbors_best_position,
                                    C_COG, C_SOC,
                                    WC_COG, WC_SOC,
                                    worst_position,
                                    neighbors_worst_position)

nps = [pso.generate_point_in_sphere2(gs, position, DIMENSION)
       for i in range(N)]

x2, y2 = zip(*nps)

plt.scatter(x2, y2, c="#FF0000")
plt.scatter(gs[0], gs[1], c="#0000FF")
plt.text(gs[0], gs[1] + 0.22, r"$G$", fontsize=20)
plt.scatter(position[0], position[1], c="#020100")
plt.text(position[0], position[1] + 0.22, r"$x$", fontsize=20)
plt.scatter(best_position[0], best_position[1], c="#00FF00")
plt.text(best_position[0], best_position[1] + 0.22, r"$l$", fontsize=20)
plt.scatter(neighbors_best_position[0], neighbors_best_position[1],
            c="#daf202")
plt.text(neighbors_best_position[0], neighbors_best_position[1] + 0.22,
         r"$p$", fontsize=20)
plt.scatter(worst_position[0], worst_position[1], c="#da02f2")
plt.text(worst_position[0], worst_position[1] + 0.22, r"$l_{w}$", fontsize=20)
plt.scatter(neighbors_worst_position[0], neighbors_worst_position[1],
            c="#7a02f2")
plt.text(neighbors_worst_position[0], neighbors_worst_position[1] + 0.22,
         r"$p_{w}$", fontsize=20)

X = 3
plt.axis([-7, 7, -7, 7])
plt.show()
