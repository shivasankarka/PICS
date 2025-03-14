import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Callable, Union, Optional, Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PICSHEP")


class Model:
    """
    Model class for PICSHEP (Particle Interaction Cascade Simulation for High Energy Physics)

    This class encapsulates the physical model parameters and functions needed for
    cascade equation calculations.

    Attributes:
        e_min: Minimum energy value for the model
        e_max: Maximum energy value for the model
        flux: Function to calculate the initial flux
        cross_section: Function to calculate the interaction cross-section
        diff_cross_section: Function to calculate the differential cross-section
    """

    def __init__(self, e_min: float, e_max: float):
        """
        Initialize the model with energy bounds.

        Args:
            e_min: Minimum energy value
            e_max: Maximum energy value

        Raises:
            ValueError: If e_min is greater than or equal to e_max
        """
        if e_min >= e_max:
            raise ValueError(f"e_min ({e_min}) must be less than e_max ({e_max})")

        self.e_min = e_min
        self.e_max = e_max
        self.flux = None
        self.cross_section = None
        self.diff_cross_section = None

        logger.debug(f"Model initialized with e_min={e_min}, e_max={e_max}")

    def set_flux(self, flux: Callable) -> None:
        """
        Set the flux function.

        Args:
            flux: Function that takes energy as input and returns flux value
        """
        self.flux = flux
        logger.debug("Flux function set")

    def set_cross_section(self, cross_section: Callable) -> None:
        """
        Set the cross-section function.

        Args:
            cross_section: Function that calculates interaction cross-section
        """
        self.cross_section = cross_section
        logger.debug("Cross-section function set")

    def set_diff_cross_section(self, diff_cross_section: Callable) -> None:
        """
        Set the differential cross-section function.

        Args:
            diff_cross_section: Function that calculates differential cross-section
        """
        self.diff_cross_section = diff_cross_section
        logger.debug("Differential cross-section function set")

    def validate(self) -> bool:
        """
        Validate that all required functions are set.

        Returns:
            True if model is valid, False otherwise
        """
        if self.flux is None:
            logger.error("Flux function not set")
            return False
        if self.cross_section is None:
            logger.error("Cross-section function not set")
            return False
        if self.diff_cross_section is None:
            logger.error("Differential cross-section function not set")
            return False
        return True


class CascadeEquationSolver:
    """
    Cascade equation solver for PICSHEP

    This class implements numerical methods to solve the cascade equation
    for particle propagation through a medium.

    Attributes:
        model: The physical model to use for calculations
    """

    def __init__(self, model: Model):
        """
        Initialize the solver with a physical model.

        Args:
            model: The physical model to use

        Raises:
            ValueError: If the model is not properly configured
        """
        if not model.validate():
            raise ValueError("Invalid model: missing required functions")

        self.model = model
        logger.debug("CascadeEquationSolver initialized with model")

    def solve(
        self,
        energy: np.ndarray,
        a: float,
        b: float,
        num_intervals: int = 100,
        integration_steps: int = 50,
    ) -> np.ndarray:
        """
        Solve the cascade equation for a given energy array.

        Args:
            energy: Array of energy values to solve for
            a: a parameter for cross-section calculations
            b: b parameter for cross-section calculations
            num_intervals: Number of intervals for energy binning
            integration_steps: Number of steps for optical depth integration

        Returns:
            Array of solution values corresponding to input energy array

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If numerical calculation fails
        """
        logger.info(
            f"Solving cascade equation with a={a}, b={b}, intervals={num_intervals}"
        )

        # Validate inputs
        if a <= 0 or b <= 0:
            raise ValueError(f"Parameters a ({a}) and b ({b}) must be positive")
        if num_intervals < 10:
            raise ValueError(
                f"Number of intervals ({num_intervals}) is too small, minimum is 10"
            )

        try:
            # Create energy bins on logarithmic scale
            energy_bins = np.logspace(
                np.log10(self.model.e_min),
                np.log10(self.model.e_max),
                num_intervals,
                dtype=np.float64,
            )

            # Calculate bin widths in log space
            delta_energy_bins = np.diff(np.log(energy_bins))

            # Initialize solution with initial flux
            phi_solution = self.model.flux(energy_bins)

            # Calculate cross-section for each energy bin
            dm_mass = 9 / b  # Dark matter mass parameter
            sigma_array = self.model.cross_section(energy_bins, a, b, dm_mass)

            # Pre-calculate differential cross-section matrix
            logger.debug("Calculating differential cross-section matrix")
            dxs_array = np.zeros((len(energy_bins), len(energy_bins)))
            for i in range(len(energy_bins)):
                for j in range(i + 1, len(energy_bins)):  # Only calculate for j > i
                    dxs_array[i, j] = (
                        self.model.diff_cross_section(
                            energy_bins[i], energy_bins[j], a, b, dm_mass
                        )
                        * self.model.flux(energy_bins[j])
                        * delta_energy_bins[j - 1]
                    )

            # Optical depth integration with improved Euler method
            logger.debug(f"Performing integration with {integration_steps} steps")
            y = np.linspace(0, 1, integration_steps)
            delta_y = np.diff(y)

            for i in range(len(delta_y)):
                # Loss term: attenuation due to interactions
                loss_term = -phi_solution * sigma_array

                # Gain term: cascading from higher energies
                gain_term = np.zeros_like(phi_solution)
                for k in range(len(energy_bins)):
                    gain_term[k] = np.sum(dxs_array[k, :])

                # Update solution using Euler method
                phi_solution = phi_solution + delta_y[i] * (loss_term + gain_term)

                # Check for numerical instabilities
                if np.any(np.isnan(phi_solution)) or np.any(np.isinf(phi_solution)):
                    raise RuntimeError("Numerical instability detected in integration")

            # Interpolate solution to requested energy points
            logger.info("Cascade equation solved successfully")
            return np.interp(energy, energy_bins, phi_solution)

        except Exception as e:
            logger.error(f"Error solving cascade equation: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to solve cascade equation: {str(e)}")
