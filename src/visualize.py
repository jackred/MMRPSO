# -*- Mode: Python; tab-width: 8; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
import matplotlib.pyplot as plt


def plot_data(data, is_worst, min_bound, max_bound):
    for i in range(len(data)):
        if is_worst[i]:
            plt.scatter(data[i][0], data[i][1], c="#FF0000")
        else:
            plt.scatter(data[i][0], data[i][1], c="#00FF00")
    plt.axis([min_bound, max_bound, min_bound, max_bound])
    plt.draw()
    plt.pause(0.1)
    plt.clf()
