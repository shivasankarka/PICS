import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Callable, Optional

# Setup proper path handling
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/"))
sys.path.append(src_path)

try:
    from pics import CascadeEquationSolver, Model
except ImportError:
    logging.critical("Failed to import required modules from src directory")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("agn_model.log")],
)
logger = logging.getLogger("AGN_Model")

# Constants
F0 = 13.22
F1 = 1.498
F2 = -0.00167
F3 = 4.119

# Model parameters
xchi = 2.0
xH = -1
xv = -1 / 2 * (xH) - 1.0
QX = xchi**2 * xv**2


def diff_cross_section(
    enu: float, enu_x: float, a: float, b: float, dm_mass: float
) -> float:
    """
    Calculate the differential cross-section.

    Args:
        E: Energy value
        x: x parameter
        a: a parameter
        b: b parameter
        dm_mass: Dark matter mass

    Returns:
        Differential cross-section value
    """
    return QX * a * (1 + enu**2 / enu_x**2) * 1 / ((1 + 2 * b * (enu_x - enu)) ** 2)


def cross_section(enu: float, a: float, b: float, dm_mass: float) -> float:
    """
    Calculate the cross-section.

    Args:
        E: Energy value
        a: a parameter
        b: b parameter
        m_chi: Chi mass parameter

    Returns:
        Cross-section value
    """
    return (
        QX
        * (1 / (4 * b**3 * enu**2))
        * a
        * (
            2
            * b
            * enu**2
            * (
                (4 * b * enu * (2 * b * enu + 1) + 1)
                / (4 * b * enu**2 + 2 * enu + dm_mass)
            )
            + 1 / (2 * enu + dm_mass)
            - (2 * b * enu + 1) * np.log(4 * b * enu**2 + 2 * enu + dm_mass)
            + (2 * b * enu + 1) * np.log(2 * enu + dm_mass)
        )
    )


def flux(E: float) -> float:
    """
    Calculate the flux at a given energy.

    Args:
        E: Energy value

    Returns:
        Flux value
    """
    return 4.9032578920279493e-11 * (E) ** -3.196


def run_agn_model(
    e_min: float = 1.5,
    e_max: float = 15,
    num_points: int = 100,
    a_param: float = 1.0,
    b_param: float = 1.0,
    intervals: int = 100,
    save_plot: bool = False,
    output_path: str = "agn_results.png",
    show_plot: bool = True,
) -> tuple:
    """
    Run the AGN model with the specified parameters.

    Args:
        e_min: Minimum energy value
        e_max: Maximum energy value
        num_points: Number of energy points to evaluate
        a_param: a parameter for cross-section
        b_param: b parameter for cross-section
        intervals: Number of intervals for solver
        flux_func: Custom flux function (if None, default flux is used)
        save_plot: Whether to save the plot
        output_path: Path to save the plot
        show_plot: Whether to show the plot

    Returns:
        Tuple of (energy array, result array)
    """
    logger.info(
        f"Running AGN model with parameters: e_min={e_min}, e_max={e_max}, a={a_param}, b={b_param}"
    )

    try:
        # Initialize model
        model = Model(e_min=e_min, e_max=e_max)
        model.set_flux(flux)
        model.set_cross_section(cross_section)
        model.set_diff_cross_section(diff_cross_section)
        model.validate()

        # Solve cascade equation
        ces = CascadeEquationSolver(model)
        energy = np.linspace(e_min, e_max, num_points)
        result = ces.solve(energy, a_param, b_param, intervals)

        # Plot results
        if show_plot:
            plt.figure(figsize=(10, 6))
            plt.plot(energy, model.flux(energy), label="Initial Flux")
            plt.plot(energy, result, label="Cascade Result")
            # plt.plot(
            #     energy,
            #     ces.model.cross_section(energy, a_param, b_param, 0.01),
            #     label="Cross-section",
            # )
            # plt.plot(
            #     energy,
            #     ces.model.cross_section(energy, a_param, b_param, 0.1),
            #     label="Cross-section",
            # )
            # plt.plot(
            #     energy, ces.model.diff_cross_section(energy, 100.0, a_param, b_param, 1000),
            #     label="Differential cross-section",
            # )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Energy (TeV)")
            plt.ylabel("Flux (cm$^{-2}$ s$^{-1}$ TeV$^{-1}$)")
            plt.title("AGN Model: Initial Flux vs Cascade Result")
            plt.legend()

        if save_plot:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Plot saved to {output_path}")

        if show_plot:
            plt.show()

        logger.info("AGN model calculation completed successfully")
        return energy, result

    except Exception as e:
        logger.error(f"Error in AGN model calculation: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        run_agn_model()
    except Exception as e:
        logger.critical(f"AGN model execution failed: {str(e)}")
        sys.exit(1)
