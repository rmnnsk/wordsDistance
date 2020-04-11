import csv
import math
import random
import itertools
import copy

import numpy

# Словарь с данными ребрами (u, v) : dist
from past.builtins import raw_input

edges = dict()

# Замены слов на номера
word_ind = dict()
ind_word = dict()
POINT_CNT = 0

RESULT_DIM = 4  # Размерность решающего пространства
RANGE_MIN = 0  # Минимальная случайная координата точки
RANGE_MAX = 10  # Максимальная случайная координата точки
POPULATION_SIZE = 50  # Размер популяции
MUTATION1_PROB = 0.3  # Вероятность мутации1
MUTATION2_PROB = 0.7  # Вероятность мутации2
MUTATION3_PROB = 0.7  # Вероятность мутации3
ALIVE_PROC = 0.6  # Доля выживающих особей


# random.seed(228)


class Point:

    def __init__(self, coords=[], dim=RESULT_DIM):
        if not coords:
            self.dim = dim
            self.coords = numpy.array([0 for _ in range(self.dim)], dtype=float)
        else:
            if len(coords) != dim:
                print("WRONG POINT DIMENSION")
            else:
                self.coords = numpy.array(coords, dtype=float)

    def rand_init(self):
        # Инициализация точки случайными координатами
        for i in range(self.dim):
            self.coords[i] = random.uniform(RANGE_MIN, RANGE_MAX)

    def dist(self, pt):
        res = 0
        for i in range(self.dim):
            res += (self.coords[i] - pt.coords[i]) ** 2
        return math.sqrt(res)


class Individual:
    """
        МАССИВ ТОЧЕК ЗАДАЮЩИХ ПОЛОЖЕНИЕ СЛОВ В ПРОСТРАНСТВЕ
        i-ая точка лежит по i-ому индексу в массиве
    """

    def __init__(self):
        self.chromo = [Point() for i in range(POINT_CNT)]
        for i in self.chromo:
            i.rand_init()

    def set_chromo(self, new_chromo):
        self.chromo = new_chromo

    def get_fitness(self):
        res = 0
        for _edg in edges:
            u = _edg[0]
            v = _edg[1]
            need_dist = edges[_edg]
            cur_dist = self.chromo[u].dist(self.chromo[v])
            res = max(res, abs(need_dist - cur_dist))
        return res

    def print_ind(self):
        print("Individual: ")
        for i in self.chromo:
            print(i.coords)
        print("---------------------------")

    @staticmethod
    def crossover(ind1, ind2):
        cross_point = random.randint(0, POINT_CNT - 1)
        new_chomo = ind1.chromo[:cross_point] + ind2.chromo[cross_point:]
        new_ind = Individual()
        new_ind.set_chromo(new_chomo)
        return new_ind

    def mutate(self):
        random.shuffle(self.chromo)

    @staticmethod
    def mutate2(indiv):
        global edges
        mutate_pair = random.choice(list(edges.keys()))
        need_dist = edges[mutate_pair]
        ind_copy = copy.deepcopy(indiv)
        swp = random.randint(0, 1)
        if swp == 1:
            mutate_pair = mutate_pair[::-1]
        move_vector = ind_copy.chromo[mutate_pair[1]].coords - ind_copy.chromo[mutate_pair[0]].coords

        length = 0
        for i in move_vector:
            length += i ** 2
        length = math.sqrt(length)
        if length == 0:
            return ind_copy
        move_vector = move_vector / length
        move_vector = move_vector * need_dist
        ind_copy.chromo[mutate_pair[1]].coords = ind_copy.chromo[mutate_pair[0]].coords + move_vector
        return ind_copy

    @staticmethod
    def mutate3(indiv):
        ind_copy = copy.deepcopy(indiv)
        move_vector = numpy.zeros(RESULT_DIM)
        # print("move: ", len(move_vector))
        for _i in range(len(move_vector)):
            move_vector[_i] = random.uniform(-1, 1)
        point_ind = random.randint(0, len(ind_copy.chromo) - 1)
        ind_copy.chromo[point_ind].coords += move_vector
        return ind_copy


class Population:
    _cnt = itertools.count(0)
    fits = []

    def __init__(self, pop_size):
        self.individs = [Individual() for i in range(pop_size)]
        self.size = pop_size
        self.alive = math.ceil(self.size * ALIVE_PROC)

    def set_population(self, new_pop):
        self.individs = new_pop
        self.calc_fitness()

    def print(self):
        for i in range(len(self.individs)):
            print(i, end=':\n')
            self.individs[i].print_ind()

    def calc_fitness(self):
        tmp_fits = []
        for i in self.individs:
            tmp_fits.append(i.get_fitness())
        self.fits = tmp_fits

    def sigma_selection(self):
        all_fits = self.fits
        F = []
        f_avg = numpy.mean(all_fits)
        sigma = numpy.std(all_fits)
        if sigma == 0:
            for _i in range(self.size):
                F.append((all_fits, _i))
            return sorted(F, key=lambda pair: pair[0])
        for _i in range(self.size):
            F.append((1 + (all_fits[_i] - f_avg) / (2.0 * sigma), _i))
        return sorted(F, key=lambda pair: pair[0])

    def generate_child(self):
        parent1 = random.randint(0, self.size - 1)
        parent2 = random.randint(0, self.size - 1)
        while parent2 == parent1:
            parent2 = random.randint(0, self.size - 1)
        # print(parent1, parent2)
        new_child = copy.deepcopy(self.individs[parent1])
        new_child = Individual.crossover(new_child, self.individs[parent2])
        mut_prob = random.uniform(0, 1)
        if 0 <= mut_prob <= MUTATION1_PROB:
            new_child.mutate()
        if 0 <= mut_prob <= MUTATION2_PROB:
            new_child = Individual.mutate2(new_child)
        if 0 <= mut_prob <= MUTATION3_PROB:
            new_child = Individual.mutate3(new_child)
        return new_child


def read_data(file):
    global POINT_CNT, edges
    with open(file, newline='') as file:
        reader = csv.reader(file, delimiter=' ', quotechar='|')
        for row in reader:
            spl_row = row[0].split(',')
            print(spl_row)
            words = spl_row[0].split('_')
            i1 = word_ind.get(words[0], -1)
            if i1 == -1:
                i1 = POINT_CNT
                word_ind[words[0]] = POINT_CNT
                POINT_CNT += 1
            i2 = word_ind.get(words[1], -1)
            if i2 == -1:
                i2 = POINT_CNT
                word_ind[words[1]] = POINT_CNT
                POINT_CNT += 1
            edg = tuple(sorted((i1, i2)))
            edges[edg] = float(spl_row[1])


if __name__ == "__main__":
    read_data("small.csv")
    cur_population = 1
    pop = Population(POPULATION_SIZE)
    ans_min = 10 ** 9
    for i in word_ind:
        ind = word_ind[i]
        ind_word[ind] = i
    while cur_population < 200:
        pop.calc_fitness()
        probs = pop.sigma_selection()
        alive_inds = [it[1] for it in probs[:pop.alive]]
        alive = [pop.individs[it] for it in alive_inds]
        while len(alive) < POPULATION_SIZE:
            child = pop.generate_child()
            alive.append(child)
        pop.set_population(alive)
        pop.calc_fitness()
        ind = numpy.argmin(pop.fits)
        print("POPULATION: ", cur_population)
        print("lowest: ", pop.fits[ind])
        # print("--------------------------------------")
        cur_population += 1
    pop.calc_fitness()
    ind = numpy.argmin(pop.fits)
    print("lowest: ", pop.fits[ind])
    pop.individs[ind].print_ind()
    ans_ind = pop.individs[ind]
    ans_list = []
     for i in range(len(ind_word)):
         for j in range(i + 1, len(ind_word)):
             ans_str = ind_word[i] + '_' + ind_word[j]
             ans_list.append({'word_i_word_j' : ans_str, 'distance_i_j' : ans_ind.chromo[i].dist(ans_ind.chromo[j])})
     with open('ans.csv', "w", newline='') as out_file:
         fieldnames = ['word_i_word_j', 'distance_i_j']
         writer = csv.DictWriter(out_file, delimiter=',', fieldnames=fieldnames)
         writer.writeheader()
         for row in ans_list:
             writer.writerow(row)
