import logging
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
from numpy.random import default_rng

from algorithms import memetic, moea_hd, evolutionary
from models import Individual, Population

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    """
    GeneticAlgorithm class for implementing a genetic algorithm.

    ...

    Attributes
    ----------
    population_size : int
        The size of the population used in the algorithm.
    generations : int
        The number of generations to evolve the population.
    selection_probability : float
        The probability of an individual being selected for reproduction.
    crossover_probability : float
        The probability of crossover occurring between two selected individuals.
    mutation_probability : float
        The probability of mutation occurring for each gene.
    elite_count : int
        The number of elite individuals to preserve between generations.
    seed : int, optional
        The random seed used for reproducibility, by default None.

    Methods
    -------
    initialize_population(self) -> Population:
        Initialize a population of individuals.
    select_parents(self, population: Population) -> List[Individual]:
        Select parent individuals for reproduction based on selection probability.
    crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        Perform crossover between two parent individuals to create two offspring.
    mutation(self, individual: Individual) -> Individual:
        Apply mutation to an individual's genes based on the mutation probability.
    evaluate_fitness(self, population: Population) -> Population:
        Evaluate the fitness of each individual in the population.
    generate_elite(self, population: Population) -> List[Individual]:
        Select elite individuals to preserve between generations.
    evolve_population(self, population: Population) -> Population:
        Evolve the population through selection, reproduction, and mutation.
    optimize(self, generations: int) -> Population:
        Optimize the population over multiple generations.
    """

    def __init__(
        self,
        population_size: int,
        generations: int,
        selection_probability: float,
        crossover_probability: float,
        mutation_probability: float,
        elite_count: int,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the GeneticAlgorithm with the given parameters.

        Parameters
        ----------
        population_size : int
            The size of the population used in the algorithm.
        generations : int
            The number of generations to evolve the population.
        selection_probability : float
            The probability of an individual being selected for reproduction.
        crossover_probability : float
            The probability of crossover occurring between two selected individuals.
        mutation_probability : float
            The probability of mutation occurring for each gene.
        elite_count : int
            The number of elite individuals to preserve between generations.
        seed : int, optional
            The random seed used for reproducibility, by default None.

        Returns
        -------
        None
        """
        self.population_size = population_size
        self.generations = generations
        self.selection_probability = selection_probability
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elite_count = elite_count
        self.seed = seed
        self.rng = default_rng(self.seed)

    def initialize_population(self) -> Population:
        """
        Initialize a population of individuals with random genes.

        Returns
        -------
        Population
            A population of individuals.
        """
        population = Population(self.population_size)
        for i in range(self.population_size):
            individual = Individual(np.random.randint(0, 2, self.problem_size))
            population.add_individual(individual)
        return population

    def select_parents(self, population: Population) -> List[Individual]:
        """
        Select parent individuals for reproduction based on selection probability.

        Parameters
        ----------
        population : Population
            The current population from which to select parents.

        Returns
        -------
        List[Individual]
            A list of selected parent individuals.
        """
        selected_parents = []
        for _ in range(self.population_size // 2):
            parent1 = random.choices(population.individuals, k=1, weights=population.fitness)[0]
            parent2 = random.choices(
                population.individuals, k=1, weights=population.fitness
            )[0]
            selected_parents.append((parent1, parent2))
        return selected_parents

    def crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        Perform crossover between two parent individuals to create two offspring.

        Parameters
        ----------
        parent1 : Individual
            The first parent individual.
        parent2 : Individual
            The second parent individual.

        Returns
        -------
        Tuple[Individual, Individual]
            Two offspring individuals resulting from the crossover.
        """
        child1_genes = np.where(
            self.rng.random(self.problem_size) < self.crossover_probability,
            parent1.genes,
            parent2.genes,
        )
        child2_genes = np.where(
            self.rng.random(self.problem_size) < self.crossover_probability,
            parent2.genes,
            parent1.genes,
        )
        child1 = Individual(child1_genes)
        child2 = Individual(child2_genes)
        return child1, child2

    def mutation(self, individual: Individual) -> Individual:
        """
        Apply mutation to an individual's genes based on the mutation probability.

        Parameters
        ----------
        individual : Individual
            The individual to mutate.

        Returns
        -------
        Individual
            The mutated individual.
        """
        for gene_index, gene in enumerate(individual.genes):
            if self.rng.random() < self.mutation_probability:
                individual.genes[gene_index] = 1 - gene
        return individual

    def evaluate_fitness(self, population: Population) -> Population:
        """
        Evaluate the fitness of each individual in the population.

        Parameters
        ----------
        population : Population
            The population to evaluate.

        Returns
        -------
        Population
            The population with updated fitness values.
        """
        for individual in population.individuals:
            fitness_value = self.fitness_function(individual.genes)
            individual.fitness = fitness_value
        return population

    def generate_elite(self, population: Population) -> List[Individual]:
        """
        Select elite individuals to preserve between generations.

        Parameters
        ----------
        population : Population
            The current population.

        Returns
        -------
        List[Individual]
            A list of elite individuals.
        """
        sorted_population = sorted(population.individuals, key=lambda x: x.fitness, reverse=True)
        return sorted_population[: self.elite_count]

    def evolve_population(self, population: Population) -> Population:
        """
        Evolve the population through selection, reproduction, and mutation.

        Parameters
        ----------
        population : Population
            The current population.

        Returns
        -------
        Population
            The evolved population.
        """
        parents = self.select_parents(population)
        offspring = []
        for parent1, parent2 in parents:
            child1, child2 = self.crossover(parent1, parent2)
            mutated_child1 = self.mutation(child1)
            mutated_child2 = self.mutation(child2)
            offspring.append(mutated_child1)
            offspring.append(mutated_child2)
        new_population = Population(len(offspring))
        for i, individual in enumerate(offspring):
            new_population.add_individual(individual, index=i)
        return new_population

    def optimize(self, generations: int) -> Population:
        """
        Optimize the population over multiple generations.

        Parameters
        ----------
        generations : int
            The number of generations to evolve the population.

        Returns
        -------
        Population
            The optimized population.
        """
        current_generation = self.initialize_population()
        for generation in range(generations):
            logger.info(f"Starting generation {generation+1}/{generations}")
            current_generation = self.evaluate_fitness(current_generation)
            elite = self.generate_elite(current_generation)
            current_generation = self.evolve_population(current_generation)
            for elite_individual in elite:
                current_generation.add_individual(elite_individual)
            logger.info(
                f"Best fitness in generation {generation+1}: {current_generation.max_fitness()}"
            )
        return current_generation


class MemeticAlgorithm:
    # Similar implementation as above, integrating memetic algorithm specifics


class MoeaHdAlgorithm:
    # Similar implementation as above, integrating MOEA/HD algorithm specifics


# Example usage
if __name__ == "__main__":
    problem_size = 100  # Example problem size
    fitness_function = lambda genes: np.sum(genes)  # Example fitness function

    ga_params = {
        "population_size": 100,
        "generations": 100,
        "selection_probability": 0.8,
        "crossover_probability": 0.7,
        "mutation_probability": 0.1,
        "elite_count": 5,
        "seed": 42,
    }

    ga = GeneticAlgorithm(**ga_params, problem_size=problem_size)
    optimized_population = ga.optimize(ga_params["generations"])
    best_individual = optimized_population.max_fitness_individual()
    print(f"Best individual: {best_individual.genes}")
    print(f"Best fitness: {best_individual.fitness}")