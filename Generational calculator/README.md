# Generational calculator

Is a calculator that find minimum of a multiparameter functions. It use genetic algorithm for this.

## How algorithm works

We start by importing random and math, we will use it later. Then we declare our function we want to find minimum to it, then we declare our fitness function. Fitness function calculate how well does our generated result fits to our final result.

```Python
import random
from math import *

def function(x, y, z):
    return x ** 2 + 3.5 * y - log(10, z) - 10

def fitness(gene):
    result = function(gene[0], gene[1], gene[2])


    return abs(1/result)
```
    

Here we set max generation to 2000 and our population, it consist of 5000 chromosomes, chromosome is an array, it contains 3 random numbers because we have 3 parameters (z, y, z) in our original function


```Python
generations = 2000
population = []
for i in range(5000):
    population.append([random.uniform(0, 1000), random.uniform(0, 1000), random.uniform(0, 1000)])
```

In this generation for loop we create new new_population array, it will contain our new population. In the second for loop we fitness all chromosomes in population and store their fitness, then we sort our fitted_population so we know which chromosomes have highest fit score. Then we take 2 from best 100 chromosemes and crossover then to combine best genes and mutate then to maintain genetic diversity and we add our child to new_population and new generation can start. This proccess repeat for 2000 generations and then we get our best fitting numbers for our x, y, z paramaters for function.

```Python
for generation in range(generations):
    new_population = []

    fitted_population = []
    for gene in population:
        fitted_population.append([fitness(gene), gene])

    fitted_population.sort(reverse=True)

    best_genes = [best[1] for best in fitted_population[:100]]
    for i in range(1000):
        parent1 = random.choice(best_genes)
        parent2 = random.choice(best_genes)

        child = [parent1[0] * random.uniform(0.99, 1.01), parent2[1] * random.uniform(0.99, 1.01), parent1[2] * random.uniform(0.99, 1.01)]

        new_population.append(child)

    population = new_population
```
