import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from optimization_project.config import Config
from optimization_project.exceptions import VisualizationError
from optimization_project.models import Solution, Objective

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Visualization:
    def __init__(self, config: Config):
        self.config = config
        self.solutions = []
        self.objectives = []

    def load_solutions(self, solutions: List[Solution]):
        self.solutions = solutions

    def load_objectives(self, objectives: List[Objective]):
        self.objectives = objectives

    def plot_solutions(self):
        try:
            # Create a figure and axis
            fig, ax = plt.subplots()

            # Plot the solutions
            for i, solution in enumerate(self.solutions):
                ax.plot(solution.x, label=f'Solution {i+1}')

            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Solutions')

            # Show the legend and plot
            ax.legend()
            plt.show()
        except Exception as e:
            logger.error(f'Error plotting solutions: {e}')
            raise VisualizationError('Failed to plot solutions')

    def plot_objectives(self):
        try:
            # Create a figure and axis
            fig, ax = plt.subplots()

            # Plot the objectives
            for i, objective in enumerate(self.objectives):
                ax.plot(objective.values, label=f'Objective {i+1}')

            # Set labels and title
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
            ax.set_title('Objectives')

            # Show the legend and plot
            ax.legend()
            plt.show()
        except Exception as e:
            logger.error(f'Error plotting objectives: {e}')
            raise VisualizationError('Failed to plot objectives')

    def plot_pareto_front(self):
        try:
            # Create a figure and axis
            fig, ax = plt.subplots()

            # Plot the Pareto front
            for i, solution in enumerate(self.solutions):
                ax.plot(solution.objectives[0].values, solution.objectives[1].values, label=f'Solution {i+1}')

            # Set labels and title
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_title('Pareto Front')

            # Show the legend and plot
            ax.legend()
            plt.show()
        except Exception as e:
            logger.error(f'Error plotting Pareto front: {e}')
            raise VisualizationError('Failed to plot Pareto front')

    def save_plot(self, filename: str):
        try:
            # Save the plot to a file
            plt.savefig(filename)
            logger.info(f'Plot saved to {filename}')
        except Exception as e:
            logger.error(f'Error saving plot: {e}')
            raise VisualizationError('Failed to save plot')

class VisualizationError(Exception):
    pass

if __name__ == '__main__':
    # Load the configuration
    config = Config()

    # Create a visualization object
    visualization = Visualization(config)

    # Load the solutions and objectives
    solutions = [Solution(x=np.array([1, 2, 3]), objectives=[Objective(values=[4, 5, 6])]) for _ in range(10)]
    objectives = [Objective(values=[7, 8, 9]) for _ in range(10)]
    visualization.load_solutions(solutions)
    visualization.load_objectives(objectives)

    # Plot the solutions
    visualization.plot_solutions()

    # Plot the objectives
    visualization.plot_objectives()

    # Plot the Pareto front
    visualization.plot_pareto_front()

    # Save the plot to a file
    visualization.save_plot('plot.png')