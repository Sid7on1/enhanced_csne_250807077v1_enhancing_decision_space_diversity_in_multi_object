import logging
import os
import sys
import yaml
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("config.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

# Define constants
CONFIG_FILE = "config.yaml"
DEFAULT_CONFIG = {
    "optimization": {
        "algorithm": "MOEA",
        "population_size": 100,
        "max_generations": 100,
        "mutation_rate": 0.1,
        "crossover_rate": 0.5,
    },
    "flow_theory": {
        "velocity_threshold": 0.5,
        "flow_rate": 0.2,
    },
    "diet_problem": {
        "nutritional_content": {
            "protein": 0.3,
            "fat": 0.2,
            "carbohydrates": 0.5,
        },
        "cost": {
            "price_per_unit": 10,
            "discount_rate": 0.1,
        },
    },
}

# Define exception classes
class ConfigError(Exception):
    """Base exception class for configuration errors."""

class ConfigLoadError(ConfigError):
    """Exception raised when the configuration file cannot be loaded."""

class ConfigValidationError(ConfigError):
    """Exception raised when the configuration is invalid."""

# Define configuration class
class Configuration:
    """Represents the optimization configuration."""

    def __init__(self, config_file: Optional[str] = None):
        """Initializes the configuration object.

        Args:
            config_file (Optional[str]): The path to the configuration file. Defaults to None.
        """
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """Loads the configuration from the file.

        Returns:
            Dict: The loaded configuration.
        """
        try:
            with open(self.config_file, "r") as file:
                config = yaml.safe_load(file)
                return config
        except FileNotFoundError:
            raise ConfigLoadError(f"Configuration file '{self.config_file}' not found.")
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Error parsing configuration file: {e}")

    def validate_config(self) -> None:
        """Validates the configuration.

        Raises:
            ConfigValidationError: If the configuration is invalid.
        """
        # Validate optimization configuration
        if "optimization" not in self.config:
            raise ConfigValidationError("Optimization configuration is missing.")
        optimization_config = self.config["optimization"]
        if not isinstance(optimization_config, dict):
            raise ConfigValidationError("Optimization configuration is not a dictionary.")
        required_keys = ["algorithm", "population_size", "max_generations", "mutation_rate", "crossover_rate"]
        for key in required_keys:
            if key not in optimization_config:
                raise ConfigValidationError(f"Missing required key '{key}' in optimization configuration.")

        # Validate flow theory configuration
        if "flow_theory" not in self.config:
            raise ConfigValidationError("Flow theory configuration is missing.")
        flow_theory_config = self.config["flow_theory"]
        if not isinstance(flow_theory_config, dict):
            raise ConfigValidationError("Flow theory configuration is not a dictionary.")
        required_keys = ["velocity_threshold", "flow_rate"]
        for key in required_keys:
            if key not in flow_theory_config:
                raise ConfigValidationError(f"Missing required key '{key}' in flow theory configuration.")

        # Validate diet problem configuration
        if "diet_problem" not in self.config:
            raise ConfigValidationError("Diet problem configuration is missing.")
        diet_problem_config = self.config["diet_problem"]
        if not isinstance(diet_problem_config, dict):
            raise ConfigValidationError("Diet problem configuration is not a dictionary.")
        required_keys = ["nutritional_content", "cost"]
        for key in required_keys:
            if key not in diet_problem_config:
                raise ConfigValidationError(f"Missing required key '{key}' in diet problem configuration.")

    def get_config(self) -> Dict:
        """Returns the configuration.

        Returns:
            Dict: The configuration.
        """
        self.validate_config()
        return self.config

# Define configuration manager class
class ConfigurationManager:
    """Manages the configuration."""

    def __init__(self, config_file: Optional[str] = None):
        """Initializes the configuration manager object.

        Args:
            config_file (Optional[str]): The path to the configuration file. Defaults to None.
        """
        self.config = Configuration(config_file)

    def get_config(self) -> Dict:
        """Returns the configuration.

        Returns:
            Dict: The configuration.
        """
        return self.config.get_config()

# Define main class
class Config:
    """Represents the main configuration class."""

    def __init__(self):
        """Initializes the main configuration object."""
        self.config_manager = ConfigurationManager(CONFIG_FILE)

    def get_config(self) -> Dict:
        """Returns the configuration.

        Returns:
            Dict: The configuration.
        """
        return self.config_manager.get_config()

# Define constants for the research paper
VELOCITY_THRESHOLD = 0.5
FLOW_RATE = 0.2
NUTRITIONAL_CONTENT = {
    "protein": 0.3,
    "fat": 0.2,
    "carbohydrates": 0.5,
}
COST = {
    "price_per_unit": 10,
    "discount_rate": 0.1,
}

# Define main function
def main():
    config = Config()
    config_dict = config.get_config()
    logging.info(f"Loaded configuration: {config_dict}")

# Run the main function
if __name__ == "__main__":
    main()