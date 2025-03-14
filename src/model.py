import sys
import os
import warnings
import numpy as np
from scipy.interpolate import interp1d

class Model():
    """
    This class is used to define the required parameters for the model.
    """
    def __init__(self, path="./"):
        self.modelpath = path
        self.dm_mass = 0.0

    def set_model_name(self, model_name: str, model_type: str) -> None:
        """
        Set the model name and type.

        Args:
            model_name (str): The name of the model.
            model_type (str): The type of the model.
        """
        self.model_type = model_type
        self.model_name = model_name

        if "Blazar" in self.model_name:
            sys.path.append("../Models/Blazar_U1X")
            from DM_density import blazar_DM_density
            self.sigmacalc = blazar_DM_density()
        elif "AGN" in self.model_name:
            sys.path.append("../Models/AGN_U1X")
            from DM_density import AGN_DM_density
            self.sigmacalc = AGN_DM_density()
        else:
            warnings.warn("Invalid model name: {}".format(self.model_name))

    def set_energy_range(self, e_min: float, e_max: float) -> None:
        """
        Set the energy range.

        Args:
            e_min (float): The minimum energy.
            e_max (float): The maximum energy.
        """
        self.e_min = e_min
        self.e_max = e_max

    def set_diff_cross_section(self, diff_cross_section: str) -> None:
        """
        Set the differential cross section function with user defined function.

        Args:
            diff_cross_section (str): The user defined function for the differential cross section.
        """
        self.diff_cross_section_eqn = diff_cross_section
        self.dxs = eval(self.diff_cross_section_eqn)

    def set_diff_cross_section_function(self, diff_cross_section: str) -> None:
        """
        Set the differential cross section function with user defined function.

        Args:
            diff_cross_section (str): The user defined function for the differential cross section.
        """
        self.diff_cross_section_eqn = diff_cross_section
        self.dxs = lambda args: self.diff_cross_section_eqn(*args)

    def set_cross_section(self, cross_section: str) -> None:
        """
        Set the cross section function with user provided data.

        Args:
            cross_section (str): The user defined function for the cross section.
        """
        self.cross_section_eqn = cross_section
        self.xs = eval(self.cross_section_eqn)

    def set_cross_section_function(self, cross_section: str) -> None:
        """
        Set the cross section function with user defined function.

        Args:
            cross_section (str): The user defined function for the cross section.
        """ 
        self.cross_section_eqn = cross_section
        self.xs = lambda args: self.cross_section_eqn(*args)

    def set_eff_area_data(self, effective_area: str) -> None:
        """
        Set the effective interaction area function with user provided data.

        Args:
            effective_area (str): The file name of the effective area data.
        """
        if os.path.exists(self.modelpath + effective_area):
            data = np.genfromtxt(self.modelpath + effective_area, delimiter=',')
            self.eff_area_data = interp1d(data[:, 0], data[:, 1], kind='linear', fill_value='extrapolate')
            self.eff_area = lambda E: self.eff_area_data(E)[()]
        else:
            warnings.warn("Effective area file not found: {}".format(self.modelpath + effective_area))

    def set_eff_area_func(self, effective_area: str) -> None:
        """
        Set the effective interaction area function with user defined function.

        Args:
            effective_area (str): The user defined function for the effective area.
        """
        self.eff_area_eqn = effective_area
        self.eff_area = lambda E: eval(self.eff_area_eqn)

    def set_flux_data(self, flux: str) -> None:
        """
        Set the flux function with user provided data.

        Args:
            flux (str): The file name of the flux data.
        """
        if os.path.exists(self.modelpath + flux):
            data = np.genfromtxt(self.modelpath + flux, delimiter=',')
            self.flux_data = interp1d(data[:, 0], data[:, 1], kind='linear', fill_value="extrapolate")
            self.flux = lambda E: self.flux_data(E)[()]
        else:
            warnings.warn("Flux file not found: {}".format(self.modelpath + "input/flux.csv"))

    def set_flux_func(self, flux) -> None:
        """
        Set the flux function with user defined function.

        Args:
            flux (str): The user defined function for the flux.
        """
        self.flux_eqn = flux
        self.flux = lambda args: self.flux_eqn(args)

    def set_np_parameterization(self, m: str, g: str) -> None:
        """
        Set the new physics mass and coupling relation with user provided parameterizations.

        Args:
            m (str): The user defined function for the mass.
            g (str): The user defined function for the coupling relation.
        """
        self.m_eqn = lambda dm_mass, B: eval(m)
        self.g_eqn = lambda mzp, A, Sigma: eval(g)
        print("Parameterization of new physics parameters initialized:\n m=" + m + "\n g=" + g + "\n")

    def set_dm_model_info(self, model_type: str, model_mass: float) -> None:
        """
        Set the dark matter model information.

        Args:
            model_type (str): The type of the model.
            model_mass (float): The mass of the model.
        """
        self.dm_model = model_type
        self.dm_mass = model_mass

        # Blazar
        # self.models = [["CIA",7.19725*10**25,1.0],["CIIA",5.78693*10**21,1.0],["CIB",7.48421*10**26,0.48],["CIIB",1.64899*10**24,0.73]]
        # AGN
        self.models = [["CIA",2.42932*10**28, 1.0],["CIIA",3.62965*10**23, 1.0],["CIB",9.51132*10**27, 0.48],["CIIB",4.94941*10**23, 0.73]]
        [(A := self.models[i][1], B := self.models[i][2]) for i in range(len(self.models)) if self.dm_model == self.models[i][0]]
        A = float(A)
        B = float(B)
        # self.SigmaChi = 10**A * (self.dm_mass)**(1-B) * (1.98* 10**-14)**2 * 10**-3
        self.SigmaChi = A * (1.98* 10**-14)**2

    def get_sigma(self, m_chi: float) -> float:
        """
        Calculate the sigma value for a given dark matter mass.

        Args:
            m_chi (float): The dark matter mass.

        Returns:
            float: The calculated sigma value.
        """
        alpha = {
            "CIA": 7/3,
            "CIIA": 7/3,
            "CIB": 3/2,
            "CIIB": 3/2
        }[self.model_type]
        sigma_val = {
            "CIA": 10**-8,
            "CIIA": 3,
            "CIB": 10**-8,
            "CIIB": 3
        }[self.model_type]

        sigma_calc = self.sigmacalc.Sigma(alpha=alpha, r=10**3, m_chi=m_chi, sigma=sigma_val)
        return sigma_calc * 3.086*(10**18) * (1.98* 10**-14)**2  # conversion factors for pc to cm and GeV to cm
