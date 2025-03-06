import sys
src_path = "../../src/"
sys.path.append(src_path)
from PICSHEP import *

CES = CES()
model = Model(path="./")
model.set_model_name(model_name="Blazar",model_type="CIB")

F0 = 13.22
F1 = 1.498
F2 = -0.00167
F3 = 4.119

# charge = [4.0,2.0,1.0,0.0,-1.0]
# modelz = ['Dirac','Majorana-II','Complex']
modelz = ["Dirac"]
# charge = [4.0,2.0,0.0]
charge = [2.0]

for mod in modelz:
    for xH in charge:
        if mod=='Dirac':
                #xchi = 2.0
                xchi = 1000.0
                xv = -1/2*(xH) - 1.0
                QX = xchi**2 * xv**2
                #Fermionic cross section
                model.set_diff_cross_section(diff_cross_section="lambda E,x,a,b,dm_mass: "+str(QX)+"* a*(1 + E**2/x**2)* 1/((1+2*b*(x-E))**2)")
                model.set_cross_section(cross_section="lambda Ev,A,B,mchi: "+str(QX)+"*(2 * B * Ev**2 * ((4 * B * Ev * (2 * B * Ev + 1) + 1) / (4 * B * Ev**2 + 2 * Ev + mchi) + 1 / (2 * Ev + mchi)) - (2 * B * Ev + 1) * np.log(4 * B * Ev**2 + 2 * Ev + mchi) + (2 * B * Ev + 1) * np.log(2 * Ev + mchi))*A/(4 * B**3 * Ev**2)")

        elif mod=='Majorana-I':
                xchi = 1.0
                xv = -1/2*(xH) - 1.0
                QX = xchi**2 * xv**2
                #Fermionic cross section
                model.set_diff_cross_section(diff_cross_section="lambda E,x,a,b,dm_mass: "+str(QX)+"* a*(1 + E**2/x**2)* 1/((1+2*b*(x-E))**2)")
                model.set_cross_section(cross_section="lambda Ev,A,B,mchi: "+str(QX)+"*(2 * B * Ev**2 * ((4 * B * Ev * (2 * B * Ev + 1) + 1) / (4 * B * Ev**2 + 2 * Ev + mchi) + 1 / (2 * Ev + mchi)) - (2 * B * Ev + 1) * np.log(4 * B * Ev**2 + 2 * Ev + mchi) + (2 * B * Ev + 1) * np.log(2 * Ev + mchi))*A/(4 * B**3 * Ev**2)")

        elif mod=='Majorana-II':
                xchi = 5.0
                xv = -1/2*(xH) - 1.0
                QX = xchi**2 * xv**2
                #Fermionic cross section
                model.set_diff_cross_section(diff_cross_section="lambda E,x,a,b,dm_mass: "+str(QX)+"* a*(1 + E**2/x**2)* 1/((1+2*b*(x-E))**2)")
                model.set_cross_section(cross_section="lambda Ev,A,B,mchi: "+str(QX)+"*(2 * B * Ev**2 * ((4 * B * Ev * (2 * B * Ev + 1) + 1) / (4 * B * Ev**2 + 2 * Ev + mchi) + 1 / (2 * Ev + mchi)) - (2 * B * Ev + 1) * np.log(4 * B * Ev**2 + 2 * Ev + mchi) + (2 * B * Ev + 1) * np.log(2 * Ev + mchi))*A/(4 * B**3 * Ev**2)")

        elif mod=='Complex':
                xchi = 1000.0
                xv = -1/2*(xH) - 1.0
                QX = xchi**2 * xv**2
                #Complex scalar section
                model.set_diff_cross_section(diff_cross_section="lambda E,x,a,b,dm_mass: "+str(QX)+"* a*(2*E/x)* 1/((1+2*b*(x-E))**2)")
                model.set_cross_section(cross_section="lambda Evp,A,B,m_chi: "+str(QX)+"* -2*A*Evp**2*m_chi**2*(-((-((2*B*Evp*m_chi)/(4*B*Evp**2+2*Evp+m_chi)) - np.log(4*B*Evp**2+2*Evp+m_chi) + np.log(2*Evp+m_chi))/(4*B**2*Evp**3*m_chi**2))-1/(2*B*Evp**2*m_chi**2))")

        model.set_flux_func(flux="10**(-"+str(F0)+" - ("+str(F1)+"*np.log10(E))/(1 + "+str(F2)+" * np.abs(np.log10(E))**"+str(F3)+"))")

        model.set_eff_area_func(effective_area="10**(3.57 + 2.007*np.log10(E) -0.5263* np.log10(E)**2 +0.0922 * np.log10(E)**3 -0.0072* np.log10(E)**4)")

        # model.set_NP_parameterization(m="np.sqrt(dm_mass/(B))",
        #                               g="(mzp**2) * np.sqrt(A/(Sigma*10**3)) * np.sqrt(8*np.pi)"
        #                               )
        model.set_np_parameterization(m="3/(B)",
                                      g="(mzp**2) * np.sqrt(A/(Sigma)) * np.sqrt(8*np.pi)"
                                      )

        model.set_energy_range(e_min=290,e_max=10**4)

        model.set_dm_model_info(model_type="CIB", model_mass=10**-3)

        CES.set_model(model=model)

        # A and B range for normal charge = [-7,3]
        # for x_chi=100, Arange=[-12,-1], Brange=[-10,3]
        # for Majorana case, Arange = [-4,4] for rest Brange = [-3,-4]

        CES.events(e_min=290,
                   e_max=10**4,
                   t_obs= 898*24*3600,
                   a_range=[-9,-2],
                   b_range=[-4,4],
                   n_val=50,
                   n_eig=30)

        CES.plot(xlim=[10**-9,10**-2],
                 ylim=[10**-4,10**4],
                 title="AvsB_U(1)X_xH="+str(xH)+"_"+mod+"_xchi="+str(xchi),
                 do_plot = False,
                 plot_save= True,
                 event_threshold=0.1)

        CES.plot_mvsg(title="U(1)X_xH="+str(xH)+"_"+mod+"_xchi="+str(xchi)
                      ,plot_save=True)