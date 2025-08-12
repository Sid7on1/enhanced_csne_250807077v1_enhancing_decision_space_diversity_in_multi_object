import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from objective_functions.config import Config
from objective_functions.exceptions import (
    InvalidObjectiveFunctionError,
    MissingRequiredParameterError,
)
from objective_functions.metrics import (
    calculate_hamming_distance,
    calculate_velocity_threshold,
    calculate_flow_theory,
)
from objective_functions.utils import (
    validate_objective_function_parameters,
    validate_objective_function_type,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectiveFunction:
    """Base class for objective functions."""

    def __init__(self, config: Config):
        """Initialize the objective function.

        Args:
            config: Configuration object.
        """
        self.config = config

    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """Evaluate the objective function for a given solution.

        Args:
            solution: Solution to evaluate.

        Returns:
            Evaluation result.
        """
        raise NotImplementedError

class DietProblemObjectiveFunction(ObjectiveFunction):
    """Objective function for the diet problem."""

    def __init__(self, config: Config):
        """Initialize the objective function.

        Args:
            config: Configuration object.
        """
        super().__init__(config)
        self.velocity_threshold = config.velocity_threshold
        self.flow_theory = config.flow_theory

    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """Evaluate the objective function for a given solution.

        Args:
            solution: Solution to evaluate.

        Returns:
            Evaluation result.
        """
        # Calculate Hamming distance
        hamming_distance = calculate_hamming_distance(solution)

        # Calculate velocity threshold
        velocity_threshold = calculate_velocity_threshold(
            solution, self.velocity_threshold
        )

        # Calculate flow theory
        flow_theory = calculate_flow_theory(solution, self.flow_theory)

        # Return evaluation result
        return np.array([hamming_distance, velocity_threshold, flow_theory])

class MemeticObjectiveFunction(ObjectiveFunction):
    """Objective function for memetic algorithms."""

    def __init__(self, config: Config):
        """Initialize the objective function.

        Args:
            config: Configuration object.
        """
        super().__init__(config)
        self.mutation_rate = config.mutation_rate
        self.crossover_rate = config.crossover_rate

    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """Evaluate the objective function for a given solution.

        Args:
            solution: Solution to evaluate.

        Returns:
            Evaluation result.
        """
        # Validate solution
        validate_objective_function_parameters(solution)

        # Calculate evaluation result
        evaluation_result = np.array([
            self.calculate_mutation_rate(solution, self.mutation_rate),
            self.calculate_crossover_rate(solution, self.crossover_rate),
        ])

        # Return evaluation result
        return evaluation_result

    def calculate_mutation_rate(self, solution: np.ndarray, mutation_rate: float) -> float:
        """Calculate mutation rate.

        Args:
            solution: Solution to evaluate.
            mutation_rate: Mutation rate.

        Returns:
            Mutation rate.
        """
        # Calculate mutation rate
        mutation_rate_result = np.sum(solution) / len(solution)

        # Return mutation rate result
        return mutation_rate_result

    def calculate_crossover_rate(self, solution: np.ndarray, crossover_rate: float) -> float:
        """Calculate crossover rate.

        Args:
            solution: Solution to evaluate.
            crossover_rate: Crossover rate.

        Returns:
            Crossover rate.
        """
        # Calculate crossover rate
        crossover_rate_result = np.sum(solution) / len(solution)

        # Return crossover rate result
        return crossover_rate_result

def get_objective_function(config: Config) -> ObjectiveFunction:
    """Get the objective function based on the configuration.

    Args:
        config: Configuration object.

    Returns:
        Objective function.
    """
    # Validate configuration
    validate_objective_function_type(config)

    # Get objective function
    if config.objective_function_type == "diet_problem":
        return DietProblemObjectiveFunction(config)
    elif config.objective_function_type == "memetic":
        return MemeticObjectiveFunction(config)
    else:
        raise InvalidObjectiveFunctionError(
            f"Invalid objective function type: {config.objective_function_type}"
        )

def evaluate_objective_function(
    objective_function: ObjectiveFunction, solution: np.ndarray
) -> np.ndarray:
    """Evaluate the objective function for a given solution.

    Args:
        objective_function: Objective function to evaluate.
        solution: Solution to evaluate.

    Returns:
        Evaluation result.
    """
    # Evaluate objective function
    evaluation_result = objective_function.evaluate(solution)

    # Return evaluation result
    return evaluation_result