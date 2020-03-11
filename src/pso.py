# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the GPL3 License.
# If a copy of the GPL3 was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/GPL-3.0

# author: Dorian Gouzou (github.com/jackred) <jackred@tuta.io>
# Co-Authored: Thimothée Couble for BIC Coursework at HW University


# local
import utility
# module
import random
import matplotlib.pyplot as plt
from copy import deepcopy

# Hard coded constant, default value, easy to change
INERTIA_START = 0.9
INERTIA_END = 0.4
COGNITIVE_TRUST = 2
SOCIAL_TRUST = 2
VELOCITY_MAX = 20
CONVERGENCE_DELTA = 0.0001


# TODO: bound -> vector
class Particle:
    """
    constructor for the Particles class
    input:
      - dimension: int -> dimensionality of the problem
      - min_bound: float -> min boundary of the search space
        (multidimensional plane)
      - max_bound: float -> max boundary of the search space
        (multidimensional plane)
      - fitness_function: fn(list of float) return float ->
        fitness_function to evaluate the particle score.
    """
    def __init__(self, dimension, min_bound, max_bound, fitness_function,
                 comparator=utility.maximise, version=2007,
                 cognitive_trust=COGNITIVE_TRUST, social_trust=SOCIAL_TRUST,
                 velocity_max=VELOCITY_MAX):
        # ---
        # description of the search space
        # ---
        # dimensionality of the problem
        self.dimension = dimension
        # boundary of the multidimensional plane
        self.min_bound = min_bound
        self.max_bound = max_bound
        # fitness function
        self.fitness_function = fitness_function
        # ---
        # PSO hyperparameters
        # ---
        # maximise or minimise problem
        self.comparator = comparator
        # version of PSO (2007 method or 2011)
        if version in [2011, 2007]:
            self.version = version
        else:
            raise ValueError('Version of PSO are 2007 or 2011, not %d'
                             % version)
        # how much trusting its best position
        self.cognitive_trust = cognitive_trust
        # how much trusting its best neighbor
        self.social_trust = social_trust
        # limit to the velocity, to avoid going to fast
        self.velocity_max = velocity_max
        # ---
        # particle parameter
        # ---
        # initialize position of the particle
        self.init_position()
        # initialize velocity of the particle
        self.init_velocity()
        # score of the particle at its position
        self.score = 0
        # param for graph
        self.res = None
        # evaluate first time to set self.score and self.res
        self.evaluate()
        # best score/position encountered by the particle
        self.best_scores_iterations = self.score
        self.best_position = deepcopy(self.position)
        # ---
        # neighbor and best neighbor
        # will be initialized by the swarm
        # set to [] or infinity
        # ---
        self.best_neighbor_position = []
        self.best_neighbor_score = float("inf") if comparator(0, 1) \
            else -float("inf")
        self.neighbors = []

    """
        initialize randomly the position in the search space
    """
    def init_position(self):
        self.position = [random.uniform(self.min_bound, self.max_bound)
                         for _ in range(self.dimension)]

    """
    assign the SPSO 2007 or 2011 velocity equation, move method and
    initialize the velocity differently
    """
    def init_velocity(self):
        if self.version == 2007:
            self.init_velocity_2007()
            self.update_velocity = self.update_velocity_2007
            self.move = self.move_2007
        elif self.version == 2011:
            raise NotImplementedError('2011 not yet supported')
            # self.init_velocity_2011()
            # self.update_velocity = self.update_velocity_2011
            # self.move = self.move_2011

    """
    vi(0) = (U(min_bound, max_bound) - xi(0)) / 2
    """
    def init_velocity_2007(self):
        self.velocity = [(random.uniform(self.min_bound, self.max_bound)-i)/2
                         for i in self.position]

    """
    vi(0) = U(min_bound − xi(0),max_bound − xi(0))
    """
    def init_velocity_2011(self):
        self.velocity = [random.uniform(self.min_bound-self.position[i],
                                        self.max_bound - self.position[i])
                         for i in range(self.dimension)]

    """
    score: fitness value (float)
    res: result of the function (for a graph)
    """
    def evaluate(self):
        res = self.fitness_function(deepcopy(self.position))
        if type(res) != tuple:
            self.score = res
        else:
            self.score, self.res = res

    """
    iterate through all neighbors and return the best past position
    """
    def update_best_neighbor(self):
        for particle in self.neighbors:
            if self.comparator(particle.best_scores_iterations,
                               self.best_neighbor_score):
                self.best_neighbor_position = deepcopy(particle.best_position)
                self.best_neighbor_score = particle.best_scores_iterations

    def update_velocity_2011(self, inertia):
        pass

    """
    new velocity is by dimension
    w: inertia
    vi(t+1) = wv(t) + rand(0,c_trust)*(p_besti(t) - xi(t))
              + rand(0, s_trust)*(i_besti - xi(t)
    """
    def update_velocity_2007(self, inertia):
        self.update_best_neighbor()
        for i in range(self.dimension):
            self.velocity[i] = min(
                self.velocity_max,
                (inertia * self.velocity[i]
                 + random.uniform(0, 1) * self.cognitive_trust
                 * (self.best_position[i] - self.position[i])
                 + random.uniform(0, 1) * self.social_trust
                 * (self.best_neighbor_position[i] - self.position[i]))
            )

    """
    change the best previous position of the particle if the new one
    is better
    """
    def update_best_position(self):
        if self.comparator(self.score, self.best_scores_iterations):
            self.best_scores_iterations = self.score
            self.best_position = deepcopy(self.position)

    """
    move the particle following the method of PSO 2007
    confine the particle at the boundary of the search space
    and set the velocity to 0 if it hit the wall
    """
    def move_2007(self):
        for i in range(self.dimension):
            new_pos = self.position[i] + self.velocity[i]
            if new_pos > self.max_bound:
                self.position[i] = self.max_bound
                self.velocity[i] = 0
            elif new_pos < self.min_bound:
                self.position[i] = self.min_bound
                self.velocity[i] = 0
            else:
                self.position[i] = new_pos

    """
    move the particle following the method of PSO 2007
    confine the particle at the boundary of the search space
    and divide by 2 the velocity if it hit the wall
    """
    def move_2011(self):
        for i in range(self.dimension):
            new_pos = self.position[i] + self.velocity[i]
            if new_pos > self.max_bound:
                self.position[i] = self.max_bound
                self.velocity[i] *= 0.5
            elif new_pos < self.min_bound:
                self.position[i] = self.min_bound
                self.velocity[i] *= 0.5
            else:
                self.position[i] = new_pos


class PSO:
    def __init__(self, dimension, fitness_function, max_iter, n_particle=40,
                 n_neighbor=4,
                 cognitive_trust=COGNITIVE_TRUST, social_trust=SOCIAL_TRUST,
                 inertia_start=INERTIA_START, inertia_end=INERTIA_END,
                 velocity_max=VELOCITY_MAX,
                 comparator=utility.maximise, min_bound=-10, max_bound=10,
                 endl='',
                 version=2007):
        self.endl = endl
        # ---
        # description of the search space
        # ---
        # dimensionality of the problem
        if dimension <= 0:
            raise ValueError('The vector dimension should be greater than 0')
        self.dimension = dimension
        # boundary of the multidimensional plane
        self.max_bound = max_bound
        self.min_bound = min_bound
        # ---
        # PSO hyperparametres
        # ---
        # maximum number of iterations
        if max_iter <= 0:
            raise ValueError('The max iteration should be greater than 0')
        self.max_iter = max_iter
        # maximise or minimise problem
        self.comparator = comparator
        # inertia to start and end. In between, linear change
        self.inertia_start = inertia_start
        self.inertia_end = inertia_end
        if n_neighbor < 0:
            raise ValueError('The nb of neighbor should be greater than 0')
        elif n_neighbor > n_particle - 1:
            raise ValueError('The nb of neighbor should be smaller than the '
                             'number of particle - 1')
        # ---
        # Swarm initialisation
        # ---
        self.generate_swarm(n_particle, n_neighbor, fitness_function, version,
                            cognitive_trust, social_trust, velocity_max)
        # best position of the swarm
        self.best_position = []
        self.best_global_score = float("inf") if comparator(0, 1) \
            else -float("inf")
        # ---
        # creation of to be used variable for display
        # ----
        self.average_score = []
        self.best_scores_iterations = []
        self.graph_config = {}
        self.best_res = []

    """
    generate a swarm of N particle
    the initialisation of the particle position is made by the particle
    itself, as well as the score evaluation
    The information initialisation is done by the swarm
    """
    def generate_swarm(self, n_particle, n_neighbor, fn, version, c_trust,
                       s_trust, v_max):
        # create the particle
        self.particles = [
            Particle(self.dimension, self.min_bound, self.max_bound, fn,
                     self.comparator, version,
                     c_trust, s_trust, v_max)
            for _ in range(n_particle)
        ]
        # initialise the neighbors for each particles
        # determine the best neighbor
        for particle in self.particles:
            idx = self.particles.index(particle)
            for i in range(1, int((n_neighbor + 1) / 2) + 1):
                particle.neighbors.append(self.particles[idx - i])
                particle.neighbors.append(
                   self.particles[(idx + i) % n_particle]
                )
                particle.update_best_neighbor()
            # particle.neighbors = [i for i in self.particles if i != particle]

    """
    main function of PSO
    run for Nmax iteration and for global_best_score < 0.001
    the inertia is calculated linearly with the number of iteration
    """
    def run(self):
        i = 0
        while i < self.max_iter and self.best_global_score > CONVERGENCE_DELTA:
            print(self.endl + '%d / %d  ' % (i+1, self.max_iter), end="\r")
            inertia = self.inertia_start \
                - ((self.inertia_start - self.inertia_end) / self.max_iter) * i
            best_actual_score = self.particles[0].score
            for particle in self.particles:
                if self.comparator(particle.score, self.best_global_score):
                    self.best_global_score = particle.score
                    self.best_position = deepcopy(particle.position)
                    self.best_res = deepcopy(particle.res)
                if self.comparator(particle.score, best_actual_score):
                    best_actual_score = particle.score
            self.best_scores_iterations.append(best_actual_score)
            self.average_score.append(
                sum(x.score for x in self.particles) / len(self.particles))
            for particle in self.particles:
                particle.update_velocity(inertia)
                particle.move()
                particle.evaluate()
                particle.update_best_position()
            if self.graph_config:
                self.draw_graphs()
            i += 1

    """
    configure graph parameters
    """
    def set_graph_config(self, res_ex, inputs, opso=False, res=True):
        inputs_str = [f"{i}: {inputs[i]}" for i in range(len(inputs))]
        self.graph_config = {
            'res_ex': res_ex,
            'inputs': inputs_str,
            'opso': opso,
            'res': res
        }
        plt.figure(1)
        self.graph_config['ann_ax'] = plt.subplot(212)
        self.graph_config['pso_ax'] = plt.subplot(221 if opso else 211)
        if opso:
            self.graph_config['opso_ax'] = plt.subplot(222)

    """
    draw mean square error graph for pso and opso
    """
    @staticmethod
    def draw_graph_pso(pso, ax, name="PSO"):
        ax.clear()
        plt.subplot(ax)
        plt.title(name + " Mean square error evolution")
        plt.plot(pso.best_scores_iterations, color='g', label='Best')
        plt.plot(pso.average_score, color='c', label='Average')
        plt.legend()

    """
    draw graph for comparing target function and ann result
    """
    def draw_graph_ann(self, res):
        self.graph_config['ann_ax'].clear()
        plt.subplot(self.graph_config['ann_ax'])
        plt.title("Desired output and the ANN output comparison")
        plt.plot(self.graph_config['inputs'], res, label='Result')
        plt.plot(self.graph_config['inputs'], self.graph_config['res_ex'],
                 linestyle=':', label='Target')
        plt.legend()
        plt.tick_params(axis='x', labelrotation=70, width=0.5)
        plt.xticks(range(0, len(self.graph_config['inputs']), 5))

    """
    draw graph for opso and for pso
    """
    def draw_graphs(self):
        if not self.graph_config['opso']:
            if self.graph_config['res']:
                self.draw_graph_ann(self.best_res)
            self.draw_graph_pso(self, self.graph_config['pso_ax'])
        else:
            if self.graph_config['res']:
                self.draw_graph_ann(self.best_res.best_res)
            self.draw_graph_pso(self.best_res, self.graph_config['pso_ax'])
            self.draw_graph_pso(self, self.graph_config['opso_ax'], "OPSO")
        plt.pause(0.0005)
