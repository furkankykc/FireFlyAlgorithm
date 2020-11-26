import numpy as np
import math
from numpy.random import random_sample
from fitness import Ackley, Michalewicz

population_size = 50  # 10
problem_dim = 2
min_bound = -5
max_bound = 5
generations = 100  # 10
gamma = 0.97
alpha = 0.05  # 0.25
beta_init = 1
beta_min = 0.65  # 0.2

sphere = lambda x: sum([np.power(i, 2) for i in x])
griewank = lambda x: (1 / 4000) * sum([np.power(i, 2) for i in x]) - \
                     np.prod(
                         [np.cos(i / np.power(i, 0.5)) for i in x]
                         + 1
                     )


def check_position(position):
    position[position > max_bound] = max_bound
    position[position < min_bound] = min_bound
    return position


def generate_population(population_size, problem_dim, min_bound, max_bound):
    error = 1e-10
    data = (max_bound + error - min_bound) * random_sample((population_size, problem_dim)) + min_bound
    data[data > max_bound] = max_bound
    return data


population = []
for i in range(population_size):
    pos = generate_population(1, problem_dim, min_bound, max_bound)[0]
    population.append([pos, griewank([pos])])

for t in range(generations):
    print('Generation %s, best fitness %s' % (t, population[0][1]))
    population.sort(key=lambda tup: tup[1], reverse=True)

    delta = 1 - (10 ** (-4) / 0.9) ** (1 / generations)
    alpha = (1 - delta) * alpha

    tmp_population = population

    for i in range(population_size):
        for j in range(population_size):
            if population[i][1] > tmp_population[j][1]:
                r = math.sqrt(np.sum((population[i][0] - tmp_population[j][0]) ** 2))
                beta = (beta_init - beta_min) * math.exp(-gamma * r ** 2) + beta_min
                tmp = alpha * (np.random.random_sample((1, problem_dim))[0] - 0.5) * (
                        max_bound - min_bound)
                population[j][0] = check_position(
                    population[i][0] * (1 - beta) + tmp_population[
                        j][0] * beta + tmp)
                # population[j][1] = Michalewicz(problem_dim).get_y(population[j][0])
                population[j][1] = griewank([population[j][0]])
    population[0][0] = generate_population(1, problem_dim, min_bound, max_bound)[0]
    # population[0][1] = Michalewicz(problem_dim).get_y(population[0][0])
    population[0][1] = griewank([population[0][0]])
population.sort(key=lambda tup: tup[1], reverse=True)
print(f'result = {population[0]}')
