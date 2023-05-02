import random
from math import *

def function(x, y, z):
    return x ** 2 + 3.5 * y - log(10, z) - 10

def fitness(gene):
    result = function(gene[0], gene[1], gene[2])


    return abs(1/result)


generations = 20000

population = []
for i in range(5000):
    population.append([random.uniform(0, 1000), random.uniform(0, 1000), random.uniform(0, 1000)])


for generation in range(generations):
    new_population = []

    fitted_population = []
    for gene in population:
        fitted_population.append([fitness(gene), gene])

    fitted_population.sort(reverse=True)

    if fitted_population[0][0] > 1000:
        print(fitted_population[0])
        print(function(fitted_population[0][1][0], fitted_population[0][1][1], fitted_population[0][1][2]) + 10)
        break


    best_genes = [best[1] for best in fitted_population[:100]]
    for i in range(1000):
        parent1 = random.choice(best_genes)
        parent2 = random.choice(best_genes)

        child = [parent1[0] * random.uniform(0.99, 1.01), parent2[1] * random.uniform(0.99, 1.01), parent1[2] * random.uniform(0.99, 1.01)]

        new_population.append(child)

    population = new_population
