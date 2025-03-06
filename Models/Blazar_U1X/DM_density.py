import numpy as np
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as tck
import multiprocessing as mp
import matplotlib.gridspec as gridspec
from scipy.integrate import quad
from scipy.interpolate import interp1d

class blazar_DM_density():
    def __init__(self) -> None:
        self.tBH = 10**9*365*24*3600
        self.RS = 2.97*10**-6  #pc
        self.ri = 4*self.RS

    def rho_c(self, m_chi, sigma):
        return m_chi / (sigma * 10**-26 * self.tBH)  # GeV/cm^3

    def rho_alpha(self, alpha, r):
        return self.num(alpha) * (1 - 4 * self.RS/r)**3 * r**(-alpha) * 1/(3.086 * 10**18)**3

    def num(self, alpha):
        r_min = self.ri
        r_max = 10**5 * self.RS
        num_points = 5000
        r = np.logspace(np.log10(r_min),np.log10(r_max), num=num_points)
        dr = np.diff(r)
        integrand = (1 - 4 * self.RS / r)**3 * r**(-alpha)
        integral = np.sum(integrand[1:] * dr)
        # integral = quad(lambda rr: (1 - 4 * self.RS / rr)**3 * rr**(-alpha), self.ri, 10**5 * self.RS)[0]
        return (3.09 * 10**8 * 1.988 * 10**30 * 5.62 * 10**26) / (4 * np.pi * integral)

    def rho_chi(self, alpha, r, m_chi, sigma):
        return self.rho_alpha(alpha, r) * self.rho_c(m_chi, sigma) / (self.rho_c(m_chi, sigma) + self.rho_alpha(alpha, r))

    def Sigma(self, alpha, r, m_chi, sigma):
        rmin = 0.00002958
        r_max = 10**4
        # changes r_max from 10**10 *RS to 10**4 since it converges

        # x = np.logspace(np.log10(rmin), np.log10(10**4), 5000)
        # dx = np.diff(x)
        # dat = np.zeros((len(x)))
        # dat = self.rho_chi(alpha, x[1:], m_chi, sigma) * dx
        # result = np.sum(dat)
        x = np.logspace(np.log10(rmin),np.log10(r), num=5000)
        dx = np.diff(x)
        dat = self.rho_chi(alpha, x[1:], m_chi, sigma) * dx
        result = np.sum(dat)
        # result, _ = quad(lambda x: self.rho_chi(alpha, x, m_chi, sigma), rmin, r)
        return result

"""
Plotting the DM density profile
"""
sigmacalc = blazar_DM_density()
print(sigmacalc.Sigma(7/3, 10**3, 10**-6, 10**-8))

# x = np.logspace(np.log10(0.00002958), 4, 5000)
# rho_chi_values = np.zeros(len(x))
# rho_chi_values1 = np.zeros(len(x))
# for i in range(len(x)):
#     rho_chi_values[i]  = sigmacalc.rho_chi(7/3, x[i], 10**-6, 10**-8)
#     rho_chi_values1[i] = sigmacalc.rho_chi(3/2, x[i], 10**-6, 10**-8)
# plt.plot(x, rho_chi_values, color='r')
# plt.plot(x, rho_chi_values1, color='b')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('r')
# plt.ylabel(r'$\rho_\chi$')
# plt.show()

####
# sigmacalc = blazar_DM_density()
# m = np.logspace(-5, 3, 50)
# rho_chi_values = np.zeros(len(m))
# rho_chi_values1 = np.zeros(len(m))
# rho_chi_values2 = np.zeros(len(m))
# rho_chi_values3 = np.zeros(len(m))
# for i in range(len(m)):
#     rho_chi_values[i]  = sigmacalc.Sigma(7/3, m[i], 10**-6, 10**-8) *  3.086*(10**18)
#     rho_chi_values1[i] = sigmacalc.Sigma(7/3, m[i], 10**-6, 3) *  3.086*(10**18)
#     rho_chi_values2[i]  = sigmacalc.Sigma(3/2, m[i], 10**-6, 10**-8) *  3.086*(10**18)
#     rho_chi_values3[i]  = sigmacalc.Sigma(3/2, m[i], 10**-6, 3) *  3.086*(10**18)

# plt.rcParams['axes.linewidth'] = 2
# plt.rcParams.update({'font.size': 16})
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rcParams['axes.linewidth'] = 2
# fig = plt.figure(figsize=(8, 6))
# ax1 = plt.subplot()
# ax1.set_facecolor('white')
# plt.plot(m, rho_chi_values, color='r', label="CIA")
# plt.plot(m, rho_chi_values1, color='r', label="CIIA", linestyle='dashed')
# plt.plot(m, rho_chi_values2, color='b', label="CIB")
# plt.plot(m, rho_chi_values3, color='b', label="CIIB", linestyle='dashed')

# ax1.tick_params(which='major',direction='in',width=2,length=7,top=True,right=True, pad=7)
# ax1.tick_params(which='minor',direction='in',width=1,length=5,top=True,right=True)

# ax1.set_xlabel("r [pc]")
# ax1.set_ylabel(r"$\Sigma(r) [\mathrm{GeV/cm^2}]$")

# ax1.set_yscale('log')
# ax1.set_xscale('log')

# ax1.set_xlim([10**-6,10**3])
# # ax1.set_ylim([10**0.0,10**20.0])

# ax1.set_xticks(10**np.arange(-6.0,3.1, 2))
# # ax1.set_yticks(10**np.arange(0.0,20.1, 4))

# legend =ax1.legend(frameon=True, fancybox=True, shadow=True, borderpad=1, bbox_to_anchor=(0.9,0.2), ncol=2, borderaxespad=0, prop={'size': 13})
# legend.get_frame().set_facecolor('white')

# ax1.text(10**0.1, 10**18.5, r'$m_{DM} = 1\;\mathrm{keV}$', ha='center', va='center',
#                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),fontsize = 15)

# #############################
# # Show the plot
# plt.tight_layout()
# plt.savefig("DM_blazar_r.pdf",dpi=500)
# plt.show()

#####
# sigmacalc = blazar_DM_density()
# m = np.logspace(-6, 3, 50)
# rho_chi_values = np.zeros(len(m))
# rho_chi_values1 = np.zeros(len(m))
# rho_chi_values2 = np.zeros(len(m))
# rho_chi_values3 = np.zeros(len(m))
# for i in range(len(m)):
#     rho_chi_values[i]  = sigmacalc.Sigma(7/3, 10**3, m[i], 10**-8) *  3.086*(10**18)
#     rho_chi_values1[i] = sigmacalc.Sigma(7/3, 10**3, m[i], 3) *  3.086*(10**18)
#     rho_chi_values2[i]  = sigmacalc.Sigma(3/2,10**3, m[i],10**-8) *  3.086*(10**18)
#     rho_chi_values3[i]  = sigmacalc.Sigma(3/2,10**3, m[i], 3) *  3.086*(10**18)

# plt.rcParams['axes.linewidth'] = 2
# plt.rcParams.update({'font.size': 16})
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rcParams['axes.linewidth'] = 2
# fig = plt.figure(figsize=(8, 6))
# ax1 = plt.subplot()
# ax1.set_facecolor('white')
# plt.plot(m, rho_chi_values/m, color='r', label="CIA")
# plt.plot(m, rho_chi_values1/m, color='r', label="CIIA", linestyle='dashed')
# plt.plot(m, rho_chi_values2/m, color='b', label="CIB")
# plt.plot(m, rho_chi_values3/m, color='b', label="CIIB", linestyle='dashed')

# ax1.tick_params(which='major',direction='in',width=2,length=7,top=True,right=True, pad=7)
# ax1.tick_params(which='minor',direction='in',width=1,length=5,top=True,right=True)

# ax1.set_xlabel("r [pc]")
# ax1.set_ylabel(r"$\frac{\Sigma(r)}{m_{DM}} [\mathrm{cm^{-2}}]$")

# ax1.set_yscale('log')
# ax1.set_xscale('log')

# ax1.set_xlim([10**-6,10**3])
# # ax1.set_ylim([10**0.0,10**20.0])

# ax1.set_xticks(10**np.arange(-6.0,3.1, 2))
# # ax1.set_yticks(10**np.arange(0.0,20.1, 4))

# legend =ax1.legend(frameon=True, fancybox=True, shadow=True, borderpad=1, bbox_to_anchor=(0.5,0.75), ncol=2, borderaxespad=0, prop={'size': 13})
# legend.get_frame().set_facecolor('white')

# # ax1.text(10**0.1, 10**18.5, r'$m_{DM} = 1\;\mathrm{keV}$', ha='center', va='center',
# #                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),fontsize = 15)

# #############################
# # Show the plot
# plt.tight_layout()
# plt.savefig("DM_blazar_mdm.pdf",dpi=500)
# plt.show()
