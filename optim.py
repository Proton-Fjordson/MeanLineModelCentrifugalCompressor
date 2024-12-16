import numpy as np
import geometry
import losses
import physics
import mechanics
import compressor
import sys

import CoolProp
from scipy.optimize import minimize

def find_optimal_compressor(arguments, *additional_args):
    param_D1tip_D2, param_D1hub_D1tip, param_D3_D2, param_b2_D2, param_flsplit, param_B2g, param_B1gtip = arguments

    save, show, speed, massflow, gas, D2, tau, w_tau, w_eff = additional_args

    N = speed #rpm
    omega = N * 2* np.pi/60 #rad/s
    G = massflow #kg/s
    mygas = CoolProp.AbstractState('REFPROP', gas)

    compressor_1 = compressor.Compressor([], G, omega, gas=mygas, Tinit=15.8, Pinit=1)
    compressor_1._max_iteration = 25
    compressor_1._criterion_convergence = 5e-10
    compressor_1._save_txt_blade = False

    compressor_1._parts = \
        [compressor.IGV(compressor_1), #IGV
        compressor.Impeller(compressor_1, mechanics.Materials(name="AlSi10Mg0.6", Youngmodulus= 0, poisson= 0.33, yieldstress= 440e6, density= 2700)),
        compressor.VanelessDiffuser(compressor_1), #Vaneless diffuser
        compressor.VanedDiffuser(compressor_1), #Vaned diffuser
        compressor.Volute(compressor_1)] #Volute

    compressor_1._parts[1]._geometry = geometry.Impeller()
    compressor_1._parts[1]._geometry._inlet_diameters = np.array([max(5.5e-3, param_D1hub_D1tip*param_D1tip_D2*D2), 0, param_D1tip_D2*D2]) #m
    compressor_1._parts[1]._geometry._inlet_blade_angle = [np.deg2rad(param_B1gtip/2), 0, np.deg2rad(param_B1gtip)]
    compressor_1._parts[1]._geometry._outlet_diameter = D2 #m
    compressor_1._parts[1]._geometry._outlet_blade_height = param_b2_D2*D2 #m
    compressor_1._parts[1]._geometry._outlet_blade_angle = np.deg2rad(param_B2g)
    compressor_1._parts[1]._geometry._axial_extension = 0.4*(compressor_1._parts[1]._geometry._outlet_diameter - 0.5*(compressor_1._parts[1]._geometry._inlet_diameters[2] + compressor_1._parts[1]._geometry._inlet_diameters[0])) #m
    compressor_1._parts[1]._geometry._blade_running_clearance = 0.5e-3 #m
    compressor_1._parts[1]._geometry._splitter_blade_length_fraction = (param_flsplit if param_flsplit >= 0.5 else 0)
    compressor_1._parts[1]._geometry._number_blade_full = np.round(
        2*np.pi*np.cos((compressor_1._parts[1]._geometry._inlet_blade_angle[2] + compressor_1._parts[1]._geometry._outlet_blade_angle)/2) \
        / (0.4*np.log(compressor_1._parts[1]._geometry._outlet_diameter/compressor_1._parts[1]._geometry._inlet_diameters[2])) / (4 if param_flsplit >= 0.5 else 2)
        )
    compressor_1._parts[1]._geometry._parameter_angle = 0.5
    compressor_1._parts[1]._geometry._inlet_blade_thickness = np.array([0, 0, 0.75])*1e-3 #m
    compressor_1._parts[1]._geometry._outlet_blade_thickness = 0.75*1e-3 #m
    compressor_1._parts[1]._geometry._Ra = 10*1e-6 #m
    compressor_1._parts[1]._geometry._taperratio = 1.5
    compressor_1._parts[1]._geometry._tapertype = 'parabolic'
    compressor_1._parts[1]._geometry.compute_inlet_average_blade_angle()
    compressor_1._parts[1]._geometry.update_geometry(set_angle=False)

    #Find _parameter_angle to have vertical blade at outlet
    res = minimize(fun= compressor_1._parts[1]._geometry.compare_phi_find_param_angle,
            x0= [compressor_1._parts[1]._geometry._parameter_angle], bounds= [(0, 1)], tol= 1e-10, method= 'SLSQP')
    if not res.get('success'): 
        raise RuntimeError("The parameter angle to ensure outlet shape did not converge {}, fun = {}".format(
            compressor_1._parts[1]._geometry._phi[-1, 0] - compressor_1._parts[1]._geometry._phi[-1, 0], res.get('fun')))
    elif (compressor_1._parts[1]._geometry._phi[-1, 0] - compressor_1._parts[1]._geometry._phi[-1, 0]) > 1e-10:
        raise RuntimeError("The parameter angle to ensure outlet shape did not converge {}".format(
            compressor_1._parts[1]._geometry._phi[-1, 0] - compressor_1._parts[1]._geometry._phi[-1, 0]))
    compressor_1._parts[1]._geometry.compute_blades(nb_additional_lines= 0, nb_theta= 200, adjustthickness= False)

    compressor_1._parts[2]._geometry = geometry.VanelessDiffuser()
    compressor_1._parts[2]._geometry._outlet_diameter = param_D3_D2*D2 #m
    compressor_1._parts[2]._geometry._outlet_height = compressor_1._parts[1]._geometry._outlet_blade_height
    compressor_1._parts[2]._geometry.compute_area()

    compressor_1._parts[4]._geometry = geometry.Volute()
    compressor_1._parts[4]._geometry.solve_diameter(
        compressor_1._parts[2]._geometry._outlet_area, 
        compressor_1._parts[2]._geometry._outlet_height)
    #update diffuser end diameter to take into account volute modeling
    temp_d = compressor_1._parts[2]._geometry._outlet_diameter
    compressor_1._parts[2]._geometry.update_outlet_diameter(compressor_1._parts[4]._geometry.find_offset_diffuser(compressor_1._parts[2]._geometry._outlet_height))

    compressor_1._parts[1]._losses_models = [
        losses.IncidenceLoss(), #0
        losses.ShockLoss(), #1
        losses.DiffusionLoss(), #2
        losses.ChokingLoss(), #3
        losses.SupercriticalLoss(), #4
        losses.BladeloadingLoss(), #5
        losses.SkinfrictionLoss(), #6
        losses.ClearanceLoss(), #7
        losses.RecirculationLoss(), #8
        losses.LeakageLoss(), #9
        losses.DiscfrictionLoss() #10
    ]

    compressor_1._parts[2]._losses_models = [
        losses.MixingLoss(), #0
        losses.VanelessdiffuserLoss() #1
    ]

    compressor_1._parts[4]._losses_models = [losses.VoluteLoss()]

    compressor_1.initialize_thermodynamics()
    r = 0
    k = 0
    for p in compressor_1._parts:
        r = p.solve()
        if -1 == r:
            print(arguments)
            print(additional_args)
            raise RuntimeError("Failure in solve for {}".format(p._name))

    if -1 != r:
        compressor_1.compute_NsDs()
        compressor_1.compute_efficiencies(show=False)

    discstress, MOSds = compressor_1._parts[1].get_discstress()
    bladestress, MOSbs = compressor_1._parts[1].get_bladestress()

    if save: 
        compressor_1.save_compressor()
        compressor_1._parts[1].save_impeller_geometry(nb_additional_lines= 5, adjustthickness= False)
    if show:
        compressor_1.plot_meridionalview(nb_additional_lines= 5, show=True, force= True, hidedetails= False)
        compressor_1.plot_3Dimpeller_convergence(nb_additional_lines= 5, show=False, force= True, adjustthickness= False)

    to_minimize = (w_tau*np.abs((compressor_1._compression_ratio - tau)/tau) + w_eff*(1 - compressor_1._efficiency_compressor))/(w_tau + w_eff)
    #to_minimize = (w_tau*np.abs((compressor_1._P[-1]/compressor_1._P[0] - tau)/tau)+ w_eff*(1 - compressor_1._efficiency_compressor))/(w_tau + w_eff)
    return max(to_minimize, -1e20*r)

if __name__ == "__main__":
    opti = False
    if len(sys.argv) > 1:
        if 'False' == sys.argv[1]:
            opti = False
        elif 'True' == sys.argv[1]:
            opti = True

    speed = 100000
    massflow = 4 *1e-3
    D2 = 34.4e-3
    tau = 3
    w_tau = 7
    w_eff = 3
    x0=[0.5, #param_D1tip_D2,
        0.3, #param_D1hub_D1tip,
        1.34, #param_D3_D2,
        0.07, #param_b2_D2,
        0.75, #param_flsplit,
        -65, #param_B2g,
        -17] #param_B1gtip
    bounds= [(0.5, 0.75), #param_D1tip_D2
            (0.3, 0.8), #param_D1hub_D1tip
            (1 + 1/3., 1.5), #param_D3_D2
            (0.03, 0.08), #param_b2_D2
            (0, 0.75), #param_flsplit
            (-65, -10), #param_B2g
            (-65, -5)] #param_B1gtip

    if opti:
        res = minimize(
            fun= find_optimal_compressor,
            args=(False, False, speed, massflow, 'Nitrogen', D2, tau, w_tau, w_eff),
            x0= x0,
            bounds= bounds,
            tol= 1e-10,
            method= 'SLSQP')
        print("args = {}\nx0 = {}\nbounds ={}\n{}".format([False, False, speed, massflow, 'Nitrogen', D2, tau, w_tau, w_eff], x0, bounds, res))
        print("arguments = {}".format(res.get('x')))
        find_optimal_compressor(res.get('x'), *(True, True, speed, massflow, 'Nitrogen', D2, tau, w_tau, w_eff))
    else:
        #param_D1tip_D2, param_D1hub_D1tip, param_D3_D2, param_b2_D2, param_flsplit, param_B2g, param_B1gtip, nb_blade
        arguments = [0.5, 0.3, 1.33333333, 0.06908892, 0.75, -45., -16.58120607]
        result = find_optimal_compressor(arguments, *(False, True, speed, massflow, 'Nitrogen', D2, tau, w_tau, w_eff))
        print(result)

