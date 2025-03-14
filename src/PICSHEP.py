###############################################################################
# * Import libraries
###############################################################################
import time
import os
import sys
import warnings

import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad

import numpy as np
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import multiprocessing as mp

from model import Model

"""
Update: 
2023/03/27: Major update along with paper publication.
2024/12/18: Modularizing the code.
"""

markdown_text = """
**************************************************************************
Particle Interactions Cascade equation Solver for High Energy Physics

Author: ShivaSankar K.A
        シヴァサンカール 
        旭輝

Affiliation: PhD, 北海道大学宇宙理学専攻、大学院理学院, 北海道大学
Department of CosmoSciences, Graduate School of Science, Hokkaido University

Email: shivasankar.ka@gmail.com
*************************************************************************
"""
print(markdown_text)

class CascadeEquationSolver():
    """
    Particle Interactions Cascade equation Solver for High Energy Physics.

    This class provides methods for solving the cascade equation and calculating the number of events and attenuated flux for a given energy range and observation time.

    Attributes:
        e_min (float): The minimum energy.
        e_max (float): The maximum energy.
        N (int): The number of points in the range.
        N_eig (int): The number of eigenvectors.
        model (Model): The model used for calculations.

    Methods:
        set_model(model): Set the model for calculations.
        eigcalc(Energy, num, a, b): Calculate the attenuated flux at the required energy.
        events(e_min, e_max, t_obs, A_range, B_range, N=20, N_eig=20): Calculate the number of events for given energy range and observation time.
        total_events(e_min, e_max, Aval, Bval, t_obs, N_eig): Calculate the total number of events.
        attenuated_flux(e_min, e_max, N_eig, Aval, Bval): Calculate the attenuated flux.
    """
    def __init__(self):
        self.e_min = None
        self.e_max = None
        self.num = None
        self.num_eig = None

    def set_model(self, model: Model) -> None:
        """
        Set the model for calculations.

        Args:
            model (Model): The model used for calculations.
        """
        self.model = model

    def eigcalc(self, energy: float, num: int, a_mat: float, b_mat: float) -> np.ndarray:
        """
        Calculate the attenuated flux at the required energy E.

        Args:
            Energy (float): The required energy.
            num (int): The number of energy points.
            a (float): Parameter a.
            b (float): Parameter b.

        Returns:
            np.ndarray: The attenuated flux at the required energy E.
        """
        model = self.model
        energy_arr = np.logspace(np.log10(self.e_min), np.log10(self.e_max), num, dtype=np.float64)
        delta_e = np.diff(np.log(energy_arr))

        phi_0 = model.flux(energy_arr)
        sigma_array = model.xs(energy_arr, a_mat, b_mat, 9/b_mat)
        dxs_array = np.triu(model.dxs(energy_arr[:, None], energy_arr, a_mat, b_mat, 9/b_mat))
        
        rhn = np.zeros((len(energy_arr), len(energy_arr)))
        i_upper, j_upper = np.triu_indices(len(energy_arr), 1)
        rhn[i_upper, j_upper] = delta_e[j_upper - 1] * dxs_array[i_upper, j_upper] * energy_arr[j_upper]**1

        # calculating eigenvalues, eigenvectors and solving for the coefficients
        w, v = LA.eig(-np.diag(sigma_array) + rhn)
        ci = LA.solve(v, phi_0)
        phisol = np.dot(v, (ci * np.exp(w)))
        return np.interp(energy, energy_arr, phisol)

    def attenuated_flux(self, energy: float, num: int, a_mat: float, b_mat: float) -> np.ndarray:
        """
        Calculate the attenuated flux at the required energy E.

        Args:
            Energy (float): The required energy.
            num (int): The number of energy points.
            a (float): Parameter a.
            b (float): Parameter b.

        Returns:
            np.ndarray: The attenuated flux at the required energy E.
        """
        model = self.model
        energy_arr = np.logspace(np.log10(model.e_min), np.log10(model.e_max), num, dtype=np.float64)
        delta_e = np.diff(np.log(energy_arr))
        
        # Initial flux
        phisol = model.flux(energy_arr)
        
        # Calculate cross-section for each energy
        sigma_array = np.array([model.xs([e, a_mat, b_mat, 9/b_mat]) for e in energy_arr])
        
        # Prepare differential cross-section matrix (upper triangular)
        dxs_array = np.zeros((len(energy_arr), len(energy_arr)))
        for i in range(len(energy_arr)):
            for j in range(i+1, len(energy_arr)):  # Only calculate for j > i
                # Properly weight the differential cross-section with flux and energy bin width
                dxs_array[i, j] = model.dxs([energy_arr[i], energy_arr[j], a_mat, b_mat, 9/b_mat]) * model.flux(energy_arr[j])
        
        # Apply energy bin widths to the differential cross-section matrix
        for j in range(1, len(energy_arr)):
            dxs_array[:, j] *= delta_e[j-1]
        
        # Optical depth integration with improved Euler method
        y = np.linspace(0, 1, 50)  # Increase resolution of optical depth grid
        deltay = np.diff(y)
        
        for i in range(len(deltay)):
            # Loss term: attenuation due to interactions
            loss_term = -phisol * sigma_array
            
            # Gain term: cascading from higher energies
            gain_term = np.zeros_like(phisol)
            for k in range(len(energy_arr)):
                gain_term += np.sum(dxs_array[:, k:], axis=1)
            
            # Update solution using Euler method
            phisol = phisol + deltay[i] * (loss_term + gain_term)
        
        return np.interp(energy, energy_arr, phisol)

    def events(self, e_min: float, e_max: float, t_obs: float, a_range: list[float], b_range: list[float], n_val: int = 20, n_eig: int = 20) -> None:
        """
        Calculate the number of events for given energy range and observation time.

        Args:
            e_min (float): The minimum energy.
            e_max (float): The maximum energy.
            t_obs (float): The observation time.
            A_range (List[float]): The range of parameter A.
            B_range (List[float]): The range of parameter B.
            N (int, optional): The number of points in the range. Defaults to 20.
            N_eig (int, optional): The number of eigenvectors. Defaults to 20.
        """
        self.num = n_val
        self.num_eig = n_eig
        model = self.model

        self.e_min = e_min
        self.e_max = e_max
        aval = np.logspace(a_range[0], a_range[1], num=self.num, endpoint=True)
        bval = np.logspace(b_range[0], b_range[1], num=self.num, endpoint=True)

        steps = 20000
        delta_e = (10**np.log10(self.e_max) - 10**np.log10(self.e_min)) / steps
        enn = np.linspace(10**np.log10(self.e_min), 10**np.log10(self.e_max), steps)

        start_time = time.time()
        if os.path.exists("events_data/events.txt"):
            os.remove("events_data/events.txt")
        print("\n")

        # s = self.eigcalc(enn, self.num_eig, aval[1], bval[1]) * model.eff_area(enn)

        with open('events_data/events.txt', mode='a', newline='', encoding='utf-8') as file:
            for i in range(self.num):
                for j in range(self.num):
                    tmp = 0.0
                    tmp = t_obs * np.sum(self.eigcalc(enn, self.num_eig, aval[i], bval[j]) * model.eff_area(enn)) * delta_e
                    file.write(f"{aval[i]} {bval[j]} {tmp}\n")
        end_time = time.time()
        print("\nTime taken: ", end_time - start_time, " seconds\n")

    def total_events(self, e_min: float, e_max: float, a_val: float, b_val: float, t_obs: float, n_eig: int) -> None:
        """
        Calculate the total number of events for a given energy range, observation time, and parameter values.

        Args:
            e_min (float): The minimum energy.
            e_max (float): The maximum energy.
            Aval (float): The value of parameter A.
            Bval (float): The value of parameter B.
            t_obs (float): The observation time.
            N_eig (int): The number of eigenvectors.
        """
        self.e_min = e_min
        self.e_max = e_max
        model  = self.model
        steps = 100000
        delta_e = (10**np.log10(self.e_max)-10**np.log10(self.e_min))/steps
        enn = np.linspace(10**np.log10(self.e_min),10**np.log10(self.e_max),steps)
        print("No of events: " + str(np.sum(t_obs* self.eigcalc(enn, n_eig, a_val, b_val)*model.eff_area(enn))*delta_e))

    def plot_attenuated_flux(self, e_min: float, e_max: float, n_eig: int, a_val: float, b_val: float) -> None:
        """
        Calculate the attenuated flux for a given energy range, number of eigenvectors, and parameter values.

        Args:
            e_min (float): The minimum energy.
            e_max (float): The maximum energy.
            n_eig (int): The number of eigenvectors.
            a_val (float): The value of parameter A.
            b_val (float): The value of parameter B.
        """
        self.e_min = e_min
        self.e_max = e_max
        model = self.model
        enn = np.linspace(10**np.log10(self.e_min),10**np.log10(self.e_max),100)
        phi = self.eigcalc(enn,n_eig, a_val, b_val)
        phi_0 = model.flux(enn)
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
        plt.rcParams['axes.linewidth'] = 2
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['axes.linewidth'] = 2

        fig = plt.figure(figsize=(10,8))
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.35)
        ax = fig.add_subplot(111)
        ax.tick_params(which='major',direction='in',width=2,length=10,top=True,right=True, pad=7)
        ax.tick_params(which='minor',direction='in',width=1,length=7,top=True,right=True)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.xscale('log')
        plt.yscale('log')
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)

        plt.plot(enn, phi,color ='r',label=r"$\Phi_{att}$")
        plt.plot(enn, phi_0,color ='g',label=r"$\Phi$")
        plt.legend()
        plt.ylabel(r"$\Phi$",fontsize=22)
        plt.xlabel(r"$E\mathrm{~(TeV)}$",fontsize=22)
        plt.show()

    def new_physics(self, x_coords, y_coords):
        """
        Calculate the new physics parameters based on the given x and y coordinates.

        Args:
            x_coords (np.ndarray): The x coordinates.
            y_coords (np.ndarray): The y coordinates.
        """
        model = self.model
        mvsg = np.empty([len(x_coords),2])
        for i, x in enumerate(x_coords):
            # mvsg[i,0] = np.sqrt(dmmass/(y_coords[i]))
            # mvsg[i,1] = mvsg[i,0]**2 * np.sqrt(x_coords[i]/(SigmaChi * 10**3)) * np.sqrt(4*np.pi)
            mvsg[i, 0] = model.m_eqn(model.dm_mass, y_coords[i])
            sigma_chi  = model.get_sigma(3 * mvsg[i, 0])
            mvsg[i, 1] = model.g_eqn(mvsg[i, 0], x, sigma_chi)
        return mvsg, model.dm_model, model.dm_mass

    def plot(self, xlim: float, ylim: float, title: str, do_plot: bool, plot_save: bool, event_threshold: float) -> None:
        """
        Plot the data with specified x and y limits, title, and plot options.

        Args:
            xlim (float): The x-axis limits.
            ylim (float): The y-axis limits.
            title (str): The title of the plot.
            do_plot (bool): Whether to display the plot.
            plotsave (bool): Whether to save the plot.
            event_threshold (float): The event threshold for contour plot.
        """
        dat_fin = np.loadtxt("events_data/events.txt", delimiter=" ")
        df = pd.DataFrame(dat_fin, columns = ['Column_A','Column_B','Column_C'])
        xcol, ycol, zcol = 'Column_A', 'Column_B', 'Column_C'
        df = df.sort_values(by=[xcol, ycol])

        xvals = df[xcol].unique()
        yvals = df[ycol].unique()
        zvals = df[zcol].values.reshape(len(xvals), len(yvals)).T

        mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
        plt.rcParams['axes.linewidth'] = 2
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['axes.linewidth'] = 2

        fig = plt.figure(figsize=(10,8))
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.35)
        ax = fig.add_subplot(111)
        ax.tick_params(which='major',direction='in',width=2,length=10,top=True,right=True, pad=7)
        ax.tick_params(which='minor',direction='in',width=1,length=7,top=True,right=True)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.xscale('log')
        plt.yscale('log')

        CP = plt.contour(xvals, yvals, zvals, levels=[event_threshold], colors='r',linestyles='solid')

        x_coords = CP.allsegs[0][0][:,0]
        y_coords = CP.allsegs[0][0][:,1]

        plt.ylabel(r'$B$',fontsize=22)
        plt.xlabel(r'$A$',fontsize=22)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if plot_save is True:
            if os.path.exists("plots/"+title+".pdf"):
                os.remove("plots/"+title+".pdf")
            plt.savefig("plots/"+title+".pdf")
        with open("events_data/AvsB.txt", mode="w", newline="", encoding='utf-8') as csvfile:
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                csvfile.write(f"{x} {y}\n")
        if do_plot is False:
            plt.show(block=False)
        else:
            plt.show()

    def plot_mvsg(self, title: str, plot_save: bool) -> None:
        """
        Plot the data with specified x and y limits, title, and plot options.

        Args:
            title (str): The title of the plot.
            plot_save (bool): Whether to save the plot.
        """
        model = self.model
        data = np.loadtxt("events_data/AvsB.txt", delimiter=" ")
        mvsg_dat, model, dmmass = self.new_physics(data[:,0],data[:,1])

        mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
        plt.rcParams['axes.linewidth'] = 2
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['axes.linewidth'] = 2

        fig = plt.figure(figsize=(10,8))
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.35)
        ax = fig.add_subplot(111)
        ax.tick_params(which='major',direction='in',width=2,length=10,top=True,right=True, pad=7)
        ax.tick_params(which='minor',direction='in',width=1,length=7,top=True,right=True)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.xscale('log')
        plt.yscale('log')
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)

        plt.plot(mvsg_dat[:,0], np.sqrt(mvsg_dat[:,1]))

        plt.ylabel(r"$g_{\nu}$",fontsize=22)
        plt.xlabel(r"$m_{Z'}\mathrm{~(GeV)}$",fontsize=22)
        if plot_save is True:
            if os.path.exists("events_data/mvsg_"+title+"_m_dm="+str(dmmass)+"-"+str(model)+".txt"):
                os.remove("events_data/mvsg_"+title+"_m_dm="+str(dmmass)+"-"+str(model)+".txt")
            with open("events_data/mvsg_"+title+"_m_dm="+str(dmmass)+"-"+str(model)+".txt", mode="w", newline="", encoding='utf-8') as csvfile:
                for i in range(len(mvsg_dat)):
                    csvfile.write(f"{mvsg_dat[i,0]} {np.sqrt(mvsg_dat[i,1])}\n")
                    # writer.writerow([mvsg_dat[i,0], mvsg_dat[i,1]])
            if os.path.exists("plots/mvsg_"+title+"_m_dm="+str(dmmass)+"-"+str(model)+".pdf"):
                os.remove("plots/mvsg_"+title+"_m_dm="+str(dmmass)+"-"+str(model)+".pdf")
            plt.savefig("plots/mvsg_"+title+"_m_dm="+str(dmmass)+"-"+str(model)+".pdf")

        print("Thy Bidding is done, My Master \n")

