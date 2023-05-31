import random
import numpy as np


# Define the fitness function
def fitness_function(x):
    return sum([-(i**2) for i in x])


# Normalize values to positive range [0, +inf) (fitnesses)
# Do nothing if already in range [0, +inf)
def pos_normalize(values):
    min_v = min(values)
    if min_v < 0:
        return [i + abs(min_v) for i in values]
    else:
        return values


# Define the genetic algorithm
GENE_DIM = 10
MUTATION_RATE = 0.1
def genetic_algorithm(population_size, num_generations):
    # Generate initial population
    population = []
    for _ in range(population_size):
        individual = np.random.uniform(-10, +10, GENE_DIM)  # Randomly initialize individuals
        population.append(individual)

    # Iterate through generations
    for generation in range(num_generations):

        # Evaluate the fitness of each individual
        fitness_scores = [fitness_function(x) for x in population]

        # Select parents for reproduction
        parents = []
        normed_fitness = pos_normalize(fitness_scores)          # Normalize fitness's to [0, +inf)
        probabilities = normed_fitness / np.sum(normed_fitness)
        for _ in range(population_size // 2):
            p1_i, p2_i = np.random.choice(np.arange(len(probabilities)), replace=False, p=probabilities, size=2)
            p1, p2 = population[p1_i], population[p2_i]
            parents.append((p1, p2))

        # Generate offspring through crossover and mutation
        offspring = []
        for parent1, parent2 in parents:
            # Crossover
            crossover_point = random.randint(0, GENE_DIM)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

            # Mutation
            if random.random() < 0.1:
                mutation_point = np.random.randint(0, GENE_DIM)
                child[mutation_point] += random.uniform(-MUTATION_RATE, +MUTATION_RATE)

            offspring.append(child)

        # Replace the old population with the new offspring
        population = offspring

        print(max(fitness_scores))

    # Find the best individual after all generations
    best_individual = max(population, key=fitness_function)
    best_fitness = fitness_function(best_individual)

    return best_individual, best_fitness

# Run the genetic algorithm
population_size = 100
num_generations = 1000
best_individual, best_fitness = genetic_algorithm(population_size, num_generations)

print(f"\nBest individual: {best_individual}")
print(f"Best fitness: {best_fitness}")