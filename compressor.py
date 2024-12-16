import numpy as np
import geometry
import losses
import physics
import mechanics

import CoolProp
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import collections.abc
from scipy.optimize import minimize

class Compressor:
    _g = 9.80665 #m/s^2 standard gravity

    """
        Class representing the compressor.

        Class representing the compressor (speed, mass flow) and parts (inlet, inlet guide vane, impeller etc.)

        Attributes
        ----------
        _parts : list of Compressorpart
            self explanatory
        _massflow : float
            mass flow in kg/s
        _speed : float 
            rotation speed in rad/s

        Methods
        -------
        __init__(parts=[], massflow=0, speed=0)
            Creates a compressor with parts, massflow and speed
        return_speed_Hz
            Return the rotation speed of the compressor in Hz
        return_speed_rpm
            Return the rotation speed of the compressor in rpm (rotation per second)
    """
    def __init__(self, parts=[], massflow=0, speed=0, gas=None, Tinit=20, Pinit=1):
        """
            Constructor of Compressor class

            Constructor of Compressor class with optionnal definition of _massflow and _speed

            Parameters
            ----------
            parts : list of Compressorpart, optionnal
                self explanatory
            massflow : float, optionnal
                mass flow of the compressor in kg/s
            speed : float, optionnal
                rotation speed of the compressor in rad/s
        """
        self._parts = parts
        self._massflow = massflow
        self._speed = speed
        self._gas = gas

        len = 8
        self._T = [physics.C2K(Tinit)]*len
        self._P = [physics.bar2Pa(Pinit)]*len
        self._Tisentropic = [physics.C2K(Tinit)]*len
        self._densities = [0]*len
        self._enthalpy = [0]*len
        self._enthalpy_isentropic = [0]*len
        self._entropy = [0]*len

        self._max_iteration = 50
        self._criterion_convergence = 1e-10

        self._compression_ratio = 0
        self._compression_ratio_total = 0
        self._dh_isentropic_impeller = 0
        self._dh_isentropic = 0
        self._dh_total = 0
        self._efficiency_impeller = 0
        self._efficiency_compressor = 0
        self._power = 0
        self._torque = 0
        self._Ns = 0
        self._Ds = 0

        self._save_txt_blade = False
    
    def return_speed_Hz(self):
        """Simple conversion of the rotation speed of the compressor _speed from rad/s to Hz"""
        return self._speed/(2*np.pi)
    
    def return_speed_rpm(self):
        """Simple conversion of the rotation speed of the compressor _speed from rad/s to rpm (rotation per minute)"""
        return self.return_speed_Hz()*60

    def compute_NsDs(self):
        
        if 0 == self._compression_ratio: self.compute_efficiencies(show= False)

        self._gas.update(CoolProp.PT_INPUTS, self._P[self._parts[1]._partindex[0] - 1], self._T[self._parts[1]._partindex[0] - 1])
        a_sound_1 = self._gas.speed_sound()
        kpv1 = CoolProp.CoolProp.PropsSI(
            'ISENTROPIC_EXPANSION_COEFFICIENT', 
            'T', self._T[self._parts[1]._partindex[0] - 1], 
            'P', self._P[self._parts[1]._partindex[0] - 1], "REFPROP::" + self._gas.fluid_names()[0])
        mach_1 = np.linalg.norm(self._parts[1]._inlet_velocity[0, 1])/a_sound_1
        Tt1 = self._T[self._parts[1]._partindex[0] - 1]*(1 + (kpv1 - 1)/2*mach_1**2)
        dh = self._gas.cpmass()*Tt1*(self._compression_ratio_total**(kpv1/(kpv1 - 1)) -1)
        rhot1 = self._densities[1]*(1 + (kpv1 - 1)/2*mach_1**2)**(1/(kpv1 - 1))

        q = self._massflow/rhot1
        self._Ns = self._speed*q**0.5/dh**0.75
        self._Ds = self._parts[1]._geometry._outlet_diameter*dh**0.75/q**0.5

    def compute_efficiencies(self, show=True, withoutvolute= False):
        self._compression_ratio = self._P[-1]/self._P[0]     

        sum_internal = np.sum([k._value if "internal" == k._type else 0 for k in self._parts[1]._losses_models])
        sum_parastic = np.sum([k._value if "parasitic" == k._type else 0 for k in self._parts[1]._losses_models])
        sum_outside = 0
        if not withoutvolute:
            sum_outside = np.sum([np.sum([k._value if "outside" == k._type else 0 for k in i._losses_models]) for i in self._parts[2:]])
        else:
            print("Compute efficiencies without a volute")
            sum_outside = np.sum([np.sum([k._value if "outside" == k._type else 0 for k in i._losses_models]) for i in self._parts[2:3]])

        self._dh_isentropic_impeller = self._parts[1]._Leul - sum_internal
        self._dh_isentropic = self._dh_isentropic_impeller - sum_outside
        self._dh_total = self._parts[1]._Leul + sum_parastic
        self._efficiency_impeller = self._dh_isentropic_impeller/self._dh_total
        self._efficiency_compressor = self._dh_isentropic/self._dh_total
        self._power = self._massflow*self._dh_total
        self._torque = self._power/self._speed


        self._gas.update(CoolProp.PT_INPUTS, self._P[self._parts[1]._partindex[0] - 1], self._T[self._parts[1]._partindex[0] - 1])
        a_sound_1 = self._gas.speed_sound()
        kpv1 = CoolProp.CoolProp.PropsSI(
            'ISENTROPIC_EXPANSION_COEFFICIENT', 
            'T', self._T[self._parts[1]._partindex[0] - 1], 
            'P', self._P[self._parts[1]._partindex[0] - 1], "REFPROP::" + self._gas.fluid_names()[0])
        mach_1 = np.linalg.norm(self._parts[1]._inlet_velocity[0, 1])/a_sound_1
        Tt1 = self._T[self._parts[1]._partindex[0] - 1]*(1 + (kpv1 - 1)/2*mach_1**2)

        self._gas.update(CoolProp.PT_INPUTS, self._P[self._parts[1]._partindex[1]], self._T[self._parts[1]._partindex[1]])
        a_sound_2 = self._gas.speed_sound()
        mach_2 = np.linalg.norm(self._parts[1]._outlet_velocity[0, 1])/a_sound_2
        kpv2 = CoolProp.CoolProp.PropsSI(
            'ISENTROPIC_EXPANSION_COEFFICIENT', 
            'T', self._T[self._parts[1]._partindex[1]], 
            'P', self._P[self._parts[1]._partindex[1]], "REFPROP::" + self._gas.fluid_names()[0])
        Tt2 = self._T[self._parts[1]._partindex[1]]*(1 + (kpv2 - 1)/2*mach_2**2)
        self._compression_ratio_total = (1 + self._efficiency_compressor*(Tt2 - Tt1)/Tt1)**(kpv2/(kpv2 - 1))

        if show:
            print("----------------------------")
            print("\tTotal Compression Ratio = {:.2f}".format(self._compression_ratio_total))
            print("\tCompressor efficiency = {:.2f} %".format(self._efficiency_compressor*100))
            print("\tImpeller efficiency = {:.2f} %".format(self._efficiency_impeller*100))
            print("\tCompressor power = {:.2f} kW".format(self._power/1000))
            print("\tCompressor torque = {:.2f} Nm".format(self._torque))
            print("delta H internal/(Leul + deltaH parasitic) = {:.2f}".format(sum_internal/self._dh_total*100))
            print("delta H parasitic/(Leul + deltaH parasitic) = {:.2f}".format(sum_parastic/self._dh_total*100))
            print("delta H outside/(Leul + deltaH parasitic) = {:.2f}".format(sum_outside/self._dh_total*100))
            print("deltaH parasitic = {:.2f}".format(sum_parastic/1000))
            print("deltaH internal = {:.2f}".format(sum_internal/1000))
            print("deltaH outside = {:.2f}".format(sum_outside/1000))
            print("Leul = {}".format(self._parts[1]._Leul/1000))
            print("----------------------------")

    def initialize_thermodynamics(self):
        self._gas.update(CoolProp.PT_INPUTS, self._P[0], self._T[0])
        self._entropy[0] = self._gas.smass() #entropy J/kg/K
        self._enthalpy[0] = self._gas.hmass() #enthalpy J/kg
        self._densities[0] = self._gas.rhomass() #kg/m^3

        self._gas.update(CoolProp.PT_INPUTS, self._P[0], self._Tisentropic[0])
        self._enthalpy_isentropic[0] = self._gas.hmass()

    def save_compressor(self, filename = "./DUMP save/" + "DUMP compressor" + str(time.time()) +".txt"):
        """
            Method to save all data allowing to re-do the computation and save the results
        """
        with open(filename, 'a') as file:
            file.write('_g = {}\n'.format(self._g))
            file.write('_massflow = {}\n'.format(self._massflow))
            file.write('_speed = {}\n'.format(self._speed))
            file.write('_gas = {}\n'.format(self._gas.fluid_names()[0]))
            file.write('_compression_ratio = {}\n'.format(self._compression_ratio))
            file.write('_compression_ratio_total = {}\n'.format(self._compression_ratio_total))
            file.write('_efficiency_compressor = {}\n'.format(self._efficiency_compressor))
            file.write('_efficiency_impeller = {}\n'.format(self._efficiency_impeller))
            file.write('_power = {}\n'.format(self._power))
            file.write('_torque = {}\n'.format(self._torque))
            file.write('_Ns = {}\n'.format(self._Ns))
            file.write('_Ds = {}\n'.format(self._Ds))

            file.write('_max_iteration = {}\n'.format(self._max_iteration))
            file.write('_criterion_convergence = {}\n'.format(self._criterion_convergence))

            file.write('_T = {}\n'.format(self._T))
            file.write('_P = {}\n'.format(self._P))
            file.write('_Tisentropic = {}\n'.format(self._Tisentropic))
            file.write('_densities = {}\n'.format(self._densities))
            file.write('_enthalpy = {}\n'.format(self._enthalpy))
            file.write('_enthalpy_isentropic = {}\n'.format(self._enthalpy_isentropic))
            file.write('_entropy = {}\n'.format(self._entropy))
            file.write('_dh_isentropic_impeller = {}\n'.format(self._dh_isentropic_impeller))
            file.write('_dh_isentropic = {}\n'.format(self._dh_isentropic))
            file.write('_dh_total = {}\n'.format(self._dh_total))
            file.write('\n')

            for i in self._parts:
                i.save_compressorpart(file)
                file.write('\n')

    def plot_meridionalview(self, nb_additional_lines= 1, show=False, hidedetails=False, force= False):
        mycolors = ['#7E2F8E', '#EDB120', '#D95319', '#0072BD']
        if (type(self._parts[1]._geometry._Xt) is not np.ndarray) or (force): self._parts[1]._geometry.compute_blades(nb_additional_lines= nb_additional_lines, nb_theta= 200, adjustthickness= False)
        if 0 == self._compression_ratio: self.compute_efficiencies(show= False)

        fig2 = plt.figure()
        gs2 = gridspec.GridSpec(3,2, hspace=0.7, wspace=0.2)


        #T, P
        ax2_00 = fig2.add_subplot()
        ax2_00.set_position(gs2[0, 0].get_position(fig2))
        ax2_01 = ax2_00.twinx()
        ax2_01.set_position(gs2[0, 0].get_position(fig2))
        ax2_02 = ax2_00.twiny()
        ax2_02.set_position(gs2[0, 0].get_position(fig2))

        ax2_00.set_xlabel("r [mm]") ; ax2_00.set_ylabel("T [K]")
        ax2_01.set_ylabel("P [bar]")
        ax2_00.grid(axis='both')
        xtoplot = [0,
                self._parts[1]._geometry._inlet_diameters[1]/2,
                (self._parts[1]._geometry._inlet_diameters[1] + self._parts[1]._geometry._inlet_diameters[2])/4,
                self._parts[1]._geometry._outlet_diameter/2,
                self._parts[2]._geometry._outlet_diameter/2,
                self._parts[2]._geometry._outlet_diameter/2,
                self._parts[2]._geometry._outlet_diameter/2
                ]
        if (self._parts[-1]._name == 'Volute'):
            xtoplot = xtoplot + [self._parts[2]._geometry._outlet_diameter/2 + self._parts[4]._geometry._D/2]
        else:
            xtoplot = xtoplot + [self._parts[2]._geometry._outlet_diameter/2*1.3]
        line_T = ax2_00.plot(xtoplot, self._T, linewidth=1.2, label="Temperature", marker='+', color=mycolors[0])
        line_P = ax2_01.plot(xtoplot, np.array(self._P)/physics.bar2Pa(1), linewidth=1.2, label="Pressure", marker='+', color=mycolors[1])
        lines = line_T + line_P
        labels = [l.get_label() for l in lines]
        ax2_00.legend(lines, labels)
        ax2_00.text(xtoplot[0], self._T[0], "{:.2f} K\n{:.2f} bara".format(self._T[0], self._P[0]/physics.bar2Pa(1)), 
                    rotation_mode='anchor', fontsize= 9, ha='left', va='bottom', rotation= 45)
        ax2_00.text(xtoplot[-1], self._T[-1], "{:.2f} K\n{:.2f} bara".format(self._T[-1], self._P[-1]/physics.bar2Pa(1)), 
                    rotation_mode='anchor', fontsize= 9, ha='right', va='top', rotation= 45)
        
        xforticks = xtoplot[0:5] + xtoplot[-1:]
        ax2_02.set_xticks(xforticks)
        ax2_02.set_xticklabels(['inlet', 'imp inlet', 'imp throat', 'imp\noutlet', 'vnd diff\noutlet', 'volute\noutlet'], rotation= 45, ha= 'left')
        ax2_02.set_xlim(ax2_00.get_xlim())
        for i in xforticks: ax2_02.axvline(i, linewidth=1, linestyle=(0, (5, 10)), color='grey')


        #s, h
        ax2_10 = fig2.add_subplot()
        ax2_10.set_position(gs2[1, 0].get_position(fig2))
        ax2_11 = ax2_10.twinx()
        ax2_11.set_position(gs2[1, 0].get_position(fig2))
        ax2_12 = ax2_10.twiny()
        ax2_12.set_position(gs2[1, 0].get_position(fig2))

        ax2_10.set_xlabel("r [mm]") ; ax2_10.set_ylabel("h [kJ/kg]")
        ax2_11.set_ylabel("s [kJ/kg/K]")
        ax2_10.grid(axis='both')
        line_h = ax2_10.plot(xtoplot, np.array(self._enthalpy)/1000, linewidth=1.2, label="Enthalpy", marker='+', color=mycolors[0])
        line_s = ax2_11.plot(xtoplot, np.array(self._entropy)/1000, linewidth=1.2, label="Entropy", marker='+', color=mycolors[1])
        lines = line_h + line_s
        labels = [l.get_label() for l in lines]
        ax2_10.legend(lines, labels)
        ax2_10.text(xtoplot[0], self._enthalpy[0]/1000, "{:.2e} kJ/kg\n{:.2e} kJ/kg/K".format(self._enthalpy[0]/1000, self._entropy[0]/1000), 
                    rotation_mode='anchor', fontsize= 9, ha='left', va='bottom', rotation= 45)
        ax2_11.text(xtoplot[-1], self._entropy[-1]/1000, "{:.2e} kJ/kg\n{:.2e} kJ/kg/K".format(self._enthalpy[-1]/1000, self._entropy[-1]/1000), 
                    rotation_mode='anchor', fontsize= 9, ha='right', va='top', rotation= 45)
        
        xforticks = xtoplot[0:5] + xtoplot[-1:]
        ax2_12.set_xticks(xforticks)
        ax2_12.set_xticklabels(['inlet', 'imp inlet', 'imp throat', 'imp\noutlet', 'vnd diff\noutlet', 'volute\noutlet'], rotation= 45, ha= 'left')
        ax2_12.set_xlim(ax2_10.get_xlim())
        for i in xforticks: ax2_12.axvline(i, linewidth=1, linestyle=(0, (5, 10)), color='grey')


        #Losses
        ax2_01 = fig2.add_subplot()
        ax2_01.set_position(gs2[0:2, 1].get_position(fig2))
        sum_internal = np.sum([k._value if "internal" == k._type else 0 for k in self._parts[1]._losses_models])
        sum_parastic = np.sum([k._value if "parasitic" == k._type else 0 for k in self._parts[1]._losses_models])
        sum_outside_vaneless = np.sum([k._value if "outside" == k._type else 0 for k in self._parts[2]._losses_models])
        sum_total = sum_internal + sum_parastic + sum_outside_vaneless
        losses_type = ['internal', 'parasitic', 'outside']
        losses_bars = [
            "internal\n\n{:.1f}% (total)\n{:.2f}% (Leul + par)".format(sum_internal/sum_total*100, sum_internal/self._dh_total*100), 
            "parasitic\n\n{:.1f}% (total)\n{:.2f}% (Leul + par)".format(sum_parastic/sum_total*100, sum_parastic/self._dh_total*100), 
            "outside\nvaneless diffuser\n{:.1f}% (total)\n{:.2f}% (Leul + par)".format(sum_outside_vaneless/sum_total*100, sum_outside_vaneless/self._dh_total*100), 
            ]
        values_bars = [
            [i._value/1000 for i in self._parts[1]._losses_models if losses_type[0] == i._type], #losses[0]
            [i._value/1000 for i in self._parts[1]._losses_models if losses_type[1] == i._type], #losses[1]
            [i._value/1000 for i in self._parts[2]._losses_models], #losses[2]
            ] #losses[3]
        labels_bars = [
            [i._name for i in self._parts[1]._losses_models if losses_type[0] == i._type], #losses[0]
            [i._name for i in self._parts[1]._losses_models if losses_type[1] == i._type],
            [i._name for i in self._parts[2]._losses_models],
            ]
        if (self._parts[-1]._name == 'Volute'):
            sum_outside_volute = np.sum([k._value if "outside" == k._type else 0 for k in self._parts[4]._losses_models])
            sum_total = sum_total + sum_outside_volute
            losses_type = losses_type + ['outside']
            losses_bars = losses_bars + ["outside\nvolute\n{:.1f}% (total)\n{:.2f}% (Leul + par)".format(sum_outside_volute/sum_total*100, sum_outside_volute/self._dh_total*100)]
            values_bars = values_bars + [[i._value/1000 for i in self._parts[4]._losses_models]]
            labels_bars = labels_bars + [[i._name for i in self._parts[4]._losses_models]]
            
        colors = plt.colormaps['Pastel1'](np.linspace(0, 1, len(labels_bars[0])))
        if hidedetails:
            colors = [mycolors[1]]*len(values_bars[0])
        bottom = np.zeros(len(losses_bars))
        for idx in range(0, len(values_bars[0])):
            b = ax2_01.bar(
                losses_bars, 
                [0 if idx >= len(i) else i[idx] for i in values_bars],
                width=0.5, bottom=bottom, color= colors[idx])
            bottom = bottom + np.array([0 if idx >= len(i) else i[idx] for i in values_bars])
            if not hidedetails:
                ax2_01.bar_label(b, labels= ['' if idx >= len(i) else i[idx] for i in labels_bars], label_type='center')
        ax2_01.set_title("Losses")
        ax2_01.set_ylabel("enthalpy loss [kJ/kg]")
        ax2_01.grid(axis='y', linestyle='dotted')

        #meridional view
        ax2_20 = fig2.add_subplot()
        ax2_20.set_position(gs2[-1, 0].get_position(fig2))
        ax2_20.set_xlabel("r [mm]") ; ax2_20.set_ylabel("z [mm]")
        facteur_plot_clearance= 2
        backface = [np.array([0, (self._parts[2]._geometry._outlet_diameter)/2])*1000,
                    np.array([self._parts[1]._geometry._z[-1,0] - self._parts[1]._geometry._blade_running_clearance*facteur_plot_clearance]*2)*1000]
        diffuser = [np.array([(np.array(self._parts[1]._geometry._r[:, -1]) + self._parts[1]._geometry._blade_running_clearance*facteur_plot_clearance)[-1], 
                              self._parts[2]._geometry._outlet_diameter/2])*1000,
                    np.array([(np.array(self._parts[1]._geometry._z[:, -1]) + self._parts[1]._geometry._blade_running_clearance*facteur_plot_clearance)[-1], 
                              self._parts[1]._geometry._z[-1,0] - self._parts[1]._geometry._blade_running_clearance*facteur_plot_clearance + self._parts[2]._geometry._outlet_height])*1000]
        ax2_20.plot(
            (np.array(self._parts[1]._geometry._r[:, -1]) + self._parts[1]._geometry._blade_running_clearance*facteur_plot_clearance)*1000, 
            (np.array(self._parts[1]._geometry._z[:, -1]) + self._parts[1]._geometry._blade_running_clearance*facteur_plot_clearance)*1000, 
            linewidth=1.2, color=mycolors[1], label='front impeller')
        if (self._parts[-1]._name == 'Volute'):
            x0 = (self._parts[2]._geometry._outlet_diameter + self._parts[4]._geometry._D)/2
            y0 = self._parts[1]._geometry._z[-1,0] - self._parts[1]._geometry._blade_running_clearance*facteur_plot_clearance + self._parts[4]._geometry._D/2
            lim_theta = np.pi - np.arcsin((self._parts[1]._geometry._z[-1,0] - self._parts[1]._geometry._blade_running_clearance*facteur_plot_clearance + self._parts[2]._geometry._outlet_height - y0)/(self._parts[4]._geometry._D/2)) #so it joins with diffuser
            thetas = np.linspace(-np.pi/2, lim_theta,  num=50)
            volute_x = x0 + self._parts[4]._geometry._D/2*np.cos(thetas)
            volute_y = y0 + self._parts[4]._geometry._D/2*np.sin(thetas)
            backface = [np.array([0, (self._parts[2]._geometry._outlet_diameter + self._parts[4]._geometry._D)/2])*1000,
                        np.array([self._parts[1]._geometry._z[-1,0] - self._parts[1]._geometry._blade_running_clearance*facteur_plot_clearance]*2)*1000]
            diffuser = [np.array([(np.array(self._parts[1]._geometry._r[:, -1]) + self._parts[1]._geometry._blade_running_clearance*facteur_plot_clearance)[-1], volute_x[-1]])*1000,
                        np.array([(np.array(self._parts[1]._geometry._z[:, -1]) + self._parts[1]._geometry._blade_running_clearance*facteur_plot_clearance)[-1], volute_y[-1]])*1000]
            ax2_20.plot(volute_x*1000, volute_y*1000, linewidth=1.2, color=mycolors[2], label='volute')
            ax2_20.plot(
                np.array([self._parts[1]._geometry._r[0, -1] + self._parts[1]._geometry._blade_running_clearance*facteur_plot_clearance]*2)*1000, 
                np.array([self._parts[1]._geometry._z[0, 0] + self._parts[1]._geometry._blade_running_clearance*facteur_plot_clearance, 
                    self._parts[1]._geometry._z[0, 0] + np.max([np.abs(self._parts[1]._geometry._z[-1, 0])/2, np.max(volute_y)])])*1000, 
                linewidth=1.2, color=mycolors[1], label='inlet')
            

        ax2_20.plot(diffuser[0], diffuser[1], linewidth=1.2, color=mycolors[1], label='diffuser')
        ax2_20.plot(backface[0], backface[1], linewidth=1.2, color=mycolors[1], label='backface')
        
        ax2_20.plot(
            np.array([0, self._parts[1]._geometry._r[0,0], self._parts[1]._geometry._r[0, -1]])*1000, 
            np.array([self._parts[1]._geometry._z[0, 0]]*3)*1000, 
            linewidth=1.2, color=mycolors[0], label='impeller')
        ax2_20.plot(
            np.array([0, self._parts[1]._geometry._r[-1,0], self._parts[1]._geometry._r[-1, -1]])*1000, 
            np.array([self._parts[1]._geometry._z[-1, 0]]*3)*1000, 
            linewidth=1.2, color=mycolors[0], label='impeller')
        ax2_20.plot(self._parts[1]._geometry._r[:,0]*1000, self._parts[1]._geometry._z[:,0]*1000, linewidth=1.2, color=mycolors[0], label='impeller', linestyle='dashed')
        ax2_20.plot(self._parts[1]._geometry._r[:,-1]*1000, self._parts[1]._geometry._z[:,-1]*1000, linewidth=1.2, color=mycolors[0], label='impeller')
        ax2_20.plot(
            np.array([self._parts[1]._geometry._r[-1, -1]]*2)*1000, 
            np.array([self._parts[1]._geometry._z[-1, 0], self._parts[1]._geometry._z[-1, -1]])*1000, 
            linewidth=1.2, color=mycolors[0], label='impeller')
        ax2_20.axvline(0, linewidth=1.2, color='k', label='symmetry axis', linestyle="dashdot")
        ax2_20.set_aspect('equal')


        #Text
        textgraph = "Gas: {}\tMassflow = {} g/s\tN = {:.0f} rpm\tU2 = {:.1f} m/s".format(self._gas.fluid_names()[0], self._massflow*1000, np.round(self.return_speed_rpm()), self._parts[1]._outlet_velocity[2, 1, 1]) +\
                    "\n" + "Ns = {:.2f}       Ds = {:.2f}".format(self._Ns, self._Ds) +\
                    "\n" + r"Compression ratio $\tau$ = {:.2f}".format(self._compression_ratio) +\
                    "\t" + r"Compression ratio total $\beta$ = {:.2f}".format(self._compression_ratio_total) +\
                    "\n" + r"Compressor efficiency $\eta$ = {:.2f} %".format(self._efficiency_compressor*100) +\
                    "\n" + r"Impeller efficiency $\eta_{{imp}}$ = {:.2f} %".format(self._efficiency_impeller*100) +\
                    "\t" + "Impeller reaction = {:.2f}".format(self._parts[1]._R) +\
                    "\n" + r"Compressor power = {:.2f} kW".format(self._power/1000) +\
                    "\t" + r"Compressor torque = {:.2f} Nm".format(self._torque) +\
                    "\n" + r"Disc stress $\sigma_{{disc}}$ = {:.0f} MPa".format(self._parts[1].get_discstress()[0]/1e6) +\
                    "\t" + r"MOS $\sigma_{{disc}}$ = {:.0f} % ({})".format(self._parts[1]._MOSdisc*100, self._parts[1]._materials._name) +\
                    "\n" + r"Blade stress $\sigma_{{blade}}$ = {:.0f} MPa".format(self._parts[1].get_bladestress()[0]/1e6) +\
                    "\t" + r"MOS $\sigma_{{blade}}$ = {:.0f} % ({})".format(self._parts[1]._MOSblade*100, self._parts[1]._materials._name) +\
                    "\n" + r"$D_{{1 tip}}/D_2$ = {:.2f}".format(self._parts[1]._geometry._inlet_diameters[-1]/self._parts[1]._geometry._outlet_diameter) +\
                    "\t" + r"$D_{{1 hub}}/D_{{1 tip}}$ = {:.2f}".format(self._parts[1]._geometry._inlet_diameters[0]/self._parts[1]._geometry._inlet_diameters[-1]) +\
                    "\t" + r"$D_3/D_2$ = {:.2f}".format(self._parts[2]._geometry._outlet_diameter/self._parts[1]._geometry._outlet_diameter) +\
                    "\t" + r"$b_2/D_2$ = {:.2f}".format(self._parts[1]._geometry._outlet_blade_height/self._parts[1]._geometry._outlet_diameter) +\
                    "\n" + "blade thickness inlet = {} mm".format(np.round(np.array(self._parts[1]._geometry._inlet_blade_thickness)*1000, 2)) +\
                    "\t" + "average blade thickness outlet = {} mm".format(self._parts[1]._geometry._outlet_average_blade_thickness*1000)
        if (self._parts[-1]._name == 'Volute'):
            textgraph = textgraph + "\n" + "Volute diameter = {:.3f} mm".format(self._parts[4]._geometry._D*1000)
        textgraph = textgraph.replace('\t', ' '*10)
        plt.figtext(gs2[-1, 1].get_position(fig2).x0, gs2[-1, 1].get_position(fig2).y1, textgraph, wrap=True, fontsize=12, verticalalignment="top")

        if show: plt.show()

    def plot_3Dimpeller_convergence(self, nb_additional_lines= 1, show=False, force= False, ignoretext= False, adjustthickness= False):
        mycolors = ['#7E2F8E', '#EDB120', '#D95319', '#0072BD']
        if (type(self._parts[1]._geometry._Xt) is not np.ndarray) or (force): self._parts[1]._geometry.compute_blades(nb_additional_lines= nb_additional_lines, nb_theta= 200, adjustthickness= adjustthickness)


        fig = plt.figure()
        gs = gridspec.GridSpec(4,2, hspace=0.5, wspace=0.2)

        #Text
        if (0 == self._compression_ratio) and (not ignoretext): self.compute_efficiencies(show= False)
        textgraph = "Gas: {}      Massflow = {} g/s      N = {:.0f} rpm".format(self._gas.fluid_names()[0], self._massflow*1000, np.round(self.return_speed_rpm())) +\
                    "\n" + "Impeller reaction = {:.2f}".format(self._parts[1]._R) +\
                    "\n" + "Ns = {:.2f}       Ds = {:.2f}".format(self._Ns, self._Ds) +\
                    "\n" + r"Compression ratio $\tau$ = {:.2f}".format(self._compression_ratio) +\
                    r"    Compression ratio total $\beta$ = {:.2f}".format(self._compression_ratio_total) +\
                    "\n" + r"Impeller efficiency $\eta_{{imp}}$ = {:.2f} %".format(self._efficiency_impeller*100) +\
                    "\n" + r"Compressor efficiency $\eta$ = {:.2f} %".format(self._efficiency_compressor*100) +\
                    "\n" + r"Compressor power = {:.2f} kW".format(self._power/1000) +\
                    "\n" + r"Compressor torque = {:.2f} Nm".format(self._torque)
        plt.figtext(gs[0, 0].get_position(fig).x0, gs[0, 0].get_position(fig).y1, textgraph, fontsize=12, verticalalignment="top")
        

        #3D impeller
        ax11 = fig.add_subplot(projection='3d')
        ax11.set_position(gs[0:3, 1].get_position(fig))
        my_col = cm.viridis(self._parts[1]._geometry._mt_adim)
        
        for i, val in enumerate(self._parts[1]._geometry._phi_allblades):
            X, Y = self._parts[1]._geometry._r*np.cos(val), self._parts[1]._geometry._r*np.sin(val)
            if not (np.abs(self._parts[1]._geometry._outlet_blade_angle) < 1e-12):
                indexes = np.argwhere(np.abs(val) >= 1e-10)
            else:
                indexes = indexes = np.argwhere(np.abs(val) >= -1)
            ax11.plot_surface(X[indexes[0, 0]:indexes[-1, 0] + 1, :], Y[indexes[0, 0]:indexes[-1, 0] + 1, :], self._parts[1]._geometry._z[indexes[0, 0]:indexes[-1, 0] + 1, :], facecolors=my_col, alpha=1)

        ax11.set_aspect('equal')
        ax11.set_title("Impeller 3D view") ; ax11.set_xlabel("x [mm]") ; ax11.set_ylabel('y [mm]') ; ax11.set_zlabel('z [mm]')

        xtoplot = range(0, len(self._T))
        
        #velocities
        ax10 = fig.add_subplot()
        ax10.set_position(gs[1, 0].get_position(fig))
        ax101 = ax10.twinx()
        ax101.set_position(gs[1, 0].get_position(fig))
        
        ax10.set_xlabel("Point") ; ax10.set_ylabel("Relative velocity meridional [m/s]")
        ax101.set_ylabel("Relative velocity radial [m/s]")
        ax10.grid(axis='both')
        line_meridional_y = [
                self._parts[1]._inlet_velocity[1,1,0], #inlet
                self._parts[1]._inlet_velocity[1,1,0], #impeller inlet
                self._parts[1]._inlet_velocity[1,1,0], #impeller throat
                self._parts[1]._outlet_velocity[1,1,0], #impeller outlet
                self._parts[2]._outlet_velocity[0], #vaneless diffuser outlet
                self._parts[2]._outlet_velocity[0], #vaned diffuser throat
                self._parts[2]._outlet_velocity[0], #vaned diffuser outlet
            ]
        line_tan_y = np.abs([
                self._parts[1]._inlet_velocity[1,1,1], #inlet
                self._parts[1]._inlet_velocity[1,1,1], #impeller inlet
                self._parts[1]._inlet_velocity[1,1,1], #impeller throat
                self._parts[1]._outlet_velocity[1,1,1], #impeller outlet
                self._parts[2]._outlet_velocity[1], #vaneless diffuser outlet
                self._parts[2]._outlet_velocity[1], #vaned diffuser throat
                self._parts[2]._outlet_velocity[1], #vaned diffuser outlet
                ])
        if (self._parts[-1]._name == 'Volute'):
            line_meridional_y = line_meridional_y + [self._parts[4]._outlet_velocity[0]] #volute outlet
            line_tan_y = np.append(line_tan_y, np.abs(self._parts[4]._outlet_velocity[1])) #volute outlet
        else:
            line_meridional_y = line_meridional_y + [self._parts[2]._outlet_velocity[0]] #copy last one when no volute
            line_tan_y = np.append(line_tan_y, np.abs(self._parts[2]._outlet_velocity[1])) #copy last one when no volute
        line_meridional = ax10.plot(xtoplot, line_meridional_y, linewidth=1.2, label="Meridional mid", marker='+', color=mycolors[0])
        line_meridional_hub = ax10.plot(
            xtoplot[1:4],
            [self._parts[1]._inlet_velocity[1,0,0], #impeller inlet
            self._parts[1]._inlet_velocity[1,0,0], #impeller throat
            self._parts[1]._outlet_velocity[1,0,0]], #impeller outlet
            linewidth=1.2, label="Meridional hub", marker='+', color=mycolors[0], linestyle="dotted"
        )
        line_meridional_tip = ax10.plot(
            xtoplot[1:4],
            [self._parts[1]._inlet_velocity[1,2,0], #impeller inlet
            self._parts[1]._inlet_velocity[1,2,0], #impeller throat
            self._parts[1]._outlet_velocity[1,2,0]], #impeller outlet
            linewidth=1.2, label="Meridional tip", marker='+', color=mycolors[0], linestyle="dashed"
        )
        line_tan = ax101.plot(xtoplot, line_tan_y, linewidth=1.2, label="Tangential mid", marker='+', color=mycolors[1])
        line_tan_hub = ax101.plot(
            xtoplot[1:4],
            np.abs([self._parts[1]._inlet_velocity[1,0,1], #impeller inlet
            self._parts[1]._inlet_velocity[1,0,1], #impeller throat
            self._parts[1]._outlet_velocity[1,0,1]]), #impeller outlet
            linewidth=1.2, label="Tangential hub", marker='+', color=mycolors[1], linestyle="dotted"
        )
        line_tan_tip = ax101.plot(
            xtoplot[1:4],
            np.abs([self._parts[1]._inlet_velocity[1,2,1], #impeller inlet
            self._parts[1]._inlet_velocity[1,2,1], #impeller throat
            self._parts[1]._outlet_velocity[1,2,1]]), #impeller outlet
            linewidth=1.2, label="Tangential tip", marker='+', color=mycolors[1], linestyle="dashed"
        )
        lines = line_meridional + line_tan
        labels = [l.get_label() for l in lines]
        ax10.legend(lines, labels)

        #P = f(H, s)
        ax20 = fig.add_subplot()
        ax20.set_position(gs[2:, 0].get_position(fig))
        
        ax20.set_xlabel("Entropy [kJ/kg/K]") ; ax20.set_ylabel("Enthalpy [kJ/kg]")
        ax20.grid(axis='both')
        ax20.set_xlim((np.array(self._entropy)[0]/1000*0.998, np.array(self._entropy)[-1]/1000*1.005))
        ax20.set_ylim((np.array(self._enthalpy)[0]/1000*0.995, np.array(self._enthalpy)[-1]/1000*1.005))
        ax20.plot(np.array(self._entropy)/1000, np.array(self._enthalpy)/1000, linestyle='dotted', marker='o', fillstyle='none', color=mycolors[2])

        Tmin = np.min(self._T)*1; Tmax = np.max(self._T)*1.0 ; step = 10 ; nb_values = int((Tmax - Tmin)/step) + 1
        Ttoplot = np.linspace(Tmin, Tmax, nb_values)
        isoP_s = np.zeros((len(self._T), len(Ttoplot)))
        isoP_h = np.zeros((len(self._T), len(Ttoplot)))
        references = ['inlet', 'imp inlet', 'imp throat', 'imp outlet', 'vnl diff outlet', 'vaned diff throat', 'vaned diff outlet', 'volute\noutlet']
        for i in range(0, len(self._T)):
            if (5 == i) or (6 == i): continue
            for j in range(0, len(Ttoplot)):
                self._gas.update(CoolProp.PT_INPUTS, self._P[i], Ttoplot[j])
                isoP_s[i, j] = self._gas.smass()/1000
                isoP_h[i, j] = self._gas.hmass()/1000
            ax20.plot(isoP_s[i], isoP_h[i], linestyle='-', fillstyle='none', color='grey')
            """ax20.text(isoP_s[i, -1], isoP_h[i, -1], r"  $P_{{{}}}$".format(references[i]), 
                      rotation_mode='anchor', fontsize= 10, ha='left', va='center', rotation= -45)"""
            ax20.text(self._entropy[i]/1000, self._enthalpy[i]/1000, "{}   ".format(references[i]), fontsize= 9, ha='right', va='center')

        #h, convergence = f(D)
        ax40 = fig.add_subplot()
        ax40.set_position(gs[3, 1].get_position(fig))

        ax40.set_xlabel("Number of iterations for convergence") ; ax40.set_ylabel(r"|h - $\widetilde{h}$|")
        ax40.grid(axis='both')
        xtoplot = range(0, len(self._parts[1]._h_convergence_impeller))
        xtoplot2 = range(0, len(self._parts[2]._h_convergence_vanelessdiffuser))
        ax40.plot(xtoplot, self._parts[1]._h_convergence_impeller, linewidth=1, marker='+', label="Impeller", color=mycolors[0])
        ax40.plot(xtoplot2, self._parts[2]._h_convergence_vanelessdiffuser, linewidth=1, marker='x', label="Diffuser", color=mycolors[1])
        if (self._parts[-1]._name == 'Volute'):
            xtoplot3 = range(0, len(self._parts[4]._h_convergence_volute))
            ax40.plot(xtoplot3, self._parts[4]._h_convergence_volute, linewidth=1, marker='x', label="Volute", color=mycolors[2])
        ax40.axhline(y=self._criterion_convergence, linewidth=1.5, label="Convergence critertion", color="k")
        ax40.legend()
        ax40.set_yscale('log')
        ax40.set_ylim(top=10)

        if show: plt.show()

class Compressorpart():
    _varmax = 10/100.
    
    def __init__(self, compressor=Compressor(),
                 losses_models=[], geometry=None):
        self._compressor = compressor
        self._losses_models = losses_models
        self._geometry = geometry

    def update_thermodynamics(self):
        if isinstance(self._partindex, collections.abc.Sequence):
            for idx in self._partindex:
                if (1000 < self._compressor._T[idx]) or (1e-2 > self._compressor._P[idx]):
                    return -1
                self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[idx], self._compressor._T[idx])
                self._compressor._entropy[idx] = self._compressor._gas.smass() #entropy J/kg/K
                self._compressor._enthalpy[idx] = self._compressor._gas.hmass() #enthalpy J/kg
                self._compressor._densities[idx] = self._compressor._gas.rhomass() #kg/m^3

                self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[idx], self._compressor._Tisentropic[idx])
                self._compressor._enthalpy_isentropic[idx] = self._compressor._gas.hmass()
        else:
            if (1000 < self._compressor._T[self._partindex]) or (1e-6 > self._compressor._P[self._partindex]):
                return -1
            self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex], self._compressor._T[self._partindex])
            self._compressor._entropy[self._partindex] = self._compressor._gas.smass() #entropy J/kg/K
            self._compressor._enthalpy[self._partindex] = self._compressor._gas.hmass() #enthalpy J/kg
            self._compressor._densities[self._partindex] = self._compressor._gas.rhomass() #kg/m^3

            self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex], self._compressor._Tisentropic[self._partindex])
            self._compressor._enthalpy_isentropic[self._partindex] = self._compressor._gas.hmass()

class Volute(Compressorpart):
    _name = "Volute"
    def __init__(self, compressor=Compressor()):
        super().__init__(compressor)
        self._outlet_velocity = np.zeros((2)) #meridionnal (radial) and tangential
        self._h_convergence_volute = []
        self._partindex = 7

        self._isconverged = [False, False] #isentropic loop, given pressure loop

    def compute_outlet_velocity(self, V4u):
        self._outlet_velocity[0] = 0
        self._outlet_velocity[1] = V4u
        self._outlet_velocity[1] = self._compressor._massflow/(self._compressor._densities[self._partindex]*self._geometry._area)

    def solve(self):
        self._compressor._T[self._partindex] = self._compressor._T[self._partindex - 1]*1.05
        self._compressor._Tisentropic[self._partindex] = self._compressor._T[self._partindex]
        self._compressor._P[self._partindex] = self._compressor._P[self._partindex - 1]*1.05
        #compute losses
        self._losses_models[0].compute_loss(self._compressor._parts[2]._outlet_velocity[0])

        #given pressure loop for the real point
        for j in range(0, self._compressor._max_iteration):
            #compute thermodynamic quantities
            self._isconverged[0] = False

            #isentropic loop for the isentropic point
            for i in range(0, self._compressor._max_iteration):
                #compute thermodynamic quantities
                #Real point
                r = self.update_thermodynamics()
                if -1 == r: return r
                #Isentropic point
                self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex], self._compressor._Tisentropic[self._partindex])
                s = self._compressor._gas.smass() #entropy J/kg/K

                #compute speed
                self.compute_outlet_velocity(self._compressor._parts[2]._outlet_velocity[1])            

                #compute corrected thermodynamic quantities for the isentropic point
                s_tilde = self._compressor._entropy[self._partindex - 1] #isentropic
                h_tilde = self._compressor._enthalpy[self._partindex] + 0.5*np.linalg.norm(self._outlet_velocity)**2 \
                        - self._losses_models[0]._value

                ds = s_tilde - s
                dh = h_tilde - (self._compressor._enthalpy_isentropic[self._partindex] + 0.5*np.linalg.norm(self._outlet_velocity)**2)

                if (np.abs(dh) < self._compressor._criterion_convergence) and (np.abs(ds) < self._compressor._criterion_convergence):
                    self._isconverged[0] = True
                    break

                #change T, P
                self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex], self._compressor._Tisentropic[self._partindex])
                temp_p = self._compressor._P[self._partindex]
                temp_t = self._compressor._Tisentropic[self._partindex]
                P_tilde = self._compressor._P[self._partindex] + \
                    dh*self._compressor._gas.first_partial_deriv(CoolProp.iP, CoolProp.iHmass,CoolProp.iSmass) +\
                    ds*self._compressor._gas.first_partial_deriv(CoolProp.iP, CoolProp.iSmass,CoolProp.iHmass)
                self._compressor._Tisentropic[self._partindex] = self._compressor._Tisentropic[self._partindex] + \
                    dh*self._compressor._gas.first_partial_deriv(CoolProp.iT, CoolProp.iHmass,CoolProp.iP) +\
                    (P_tilde - self._compressor._P[self._partindex])*self._compressor._gas.first_partial_deriv(CoolProp.iT, CoolProp.iP,CoolProp.iHmass)
                self._compressor._P[self._partindex] = P_tilde

            h_tilde = self._compressor._enthalpy[self._partindex - 1] + 0.5*(
                np.linalg.norm(self._compressor._parts[2]._outlet_velocity)**2 -\
                np.linalg.norm(self._outlet_velocity)**2
            )
            dh = h_tilde - self._compressor._enthalpy[self._partindex]
            
            self._h_convergence_volute = self._h_convergence_volute + [np.abs(dh)]

            if (np.abs(dh) < self._compressor._criterion_convergence):
                self._isconverged[1] = True
                break

            self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex], self._compressor._T[self._partindex])
            temp_t = self._compressor._T[self._partindex]
            self._compressor._T[self._partindex] = self._compressor._T[self._partindex] + \
                    dh*self._compressor._gas.first_partial_deriv(CoolProp.iT, CoolProp.iHmass,CoolProp.iP)
        
        return 0

    def save_compressorpart(self, file):
        file.write(self._name + "\n")
        file.write("\t" + "_outlet_velocity = ")
        np.savetxt(file, self._outlet_velocity, delimiter=' ', newline='\t')
        file.write("\n\t" + "_h_convergence_volute = {}\n".format(self._h_convergence_volute))
        file.write("\t" + "_isconverged = {}\n".format(self._isconverged))
        file.write("\t" + "_geometry.\n")
        file.write("\t"*2 + "_D = {}\n".format(self._geometry._D))
        file.write("\t"*2 + "_area = {}\n".format(self._geometry._area))
        file.write("\t" + "_losses_models.\n")
        for i in self._losses_models:
            file.write("\t"*2 + i._name + "\n")
            file.write("\t"*3 + "_value = {}".format(i._value) + "\n")
            file.write("\t"*3 + "_type = {}".format(i._type) + "\n")
            if "Volute" == i._name:
                continue
            else:
                file.write("\t"*3 + "Loss model found but dump save not implemented\n")

class VanelessDiffuser(Compressorpart):
    _name = "Vaneless Diffuser"
    def __init__(self, compressor=Compressor()):
        super().__init__(compressor)
        self._outlet_velocity = np.zeros((2)) #for meridionnal (radial) and tangential
        self._h_convergence_vanelessdiffuser = []
        self._partindex = 4

        self._isconverged = [False, False] #isentropic loop, given pressure loop

    def compute_outlet_velocity_meridional(self):
        self._outlet_velocity[0] = self._compressor._massflow/(self._compressor._densities[self._partindex]*self._geometry._outlet_area)

    def solve(self):
        self._compressor._T[self._partindex] = self._compressor._T[self._partindex - 1]*1.02
        self._compressor._Tisentropic[self._partindex] = self._compressor._T[self._partindex]
        self._compressor._P[self._partindex] = self._compressor._P[self._partindex - 1]*1.02

        #given pressure loop for the real point
        for j in range(0, self._compressor._max_iteration):
            #compute thermodynamic quantities
            self._isconverged[0] = False

            #isentropic loop for the isentropic point
            for i in range(0, self._compressor._max_iteration):
                #compute thermodynamic quantities
                #Real point
                r = self.update_thermodynamics()
                if -1 == r: return r
                #Isentropic point
                self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex], self._compressor._Tisentropic[self._partindex])
                s = self._compressor._gas.smass() #entropy J/kg/K

                #compute speed
                self.compute_outlet_velocity_meridional()

                #impeller output
                self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex - 1], self._compressor._T[self._partindex - 1])
                visc = self._compressor._gas.viscosity()/self._compressor._gas.rhomass()
                kpv = CoolProp.CoolProp.PropsSI('ISENTROPIC_EXPANSION_COEFFICIENT', 'T', self._compressor._T[self._partindex - 1], 'P', self._compressor._P[self._partindex - 1], "REFPROP::" + self._compressor._gas.fluid_names()[0])
                soundspeed2 = self._compressor._gas.speed_sound()
                cp2 = self._compressor._gas.cpmass()
                #diffuser output
                self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex], self._compressor._T[self._partindex])
                visc = (visc + self._compressor._gas.viscosity()/self._compressor._gas.rhomass())/2
                #compute losses
                self._losses_models[0].compute_loss(
                    self._compressor._parts[1]._losses_models[4]._wmax,
                    self._compressor._parts[1]._inlet_velocity,
                    self._compressor._parts[1]._outlet_velocity,
                    self._compressor._parts[1]._geometry._outlet_area,
                    self._compressor._parts[1]._geometry._outlet_diameter,
                    self._compressor._parts[1]._geometry._outlet_blade_height
                ) #mixing
                """
                self._outlet_velocity[1] = self._losses_models[1].compute_loss(
                    type= "Stanitz",
                    args_minimal= [
                        self._compressor._parts[1]._outlet_velocity[0, 1],
                        self._outlet_velocity,
                        self._compressor._parts[1]._geometry._outlet_blade_height,
                        self._geometry._outlet_height,
                        visc,
                        self._compressor._parts[1]._geometry._outlet_diameter,
                        self._geometry._outlet_diameter,
                        self._compressor._massflow
                    ],
                    args= [
                        kpv, 
                        np.linalg.norm(self._compressor._parts[1]._outlet_velocity[1, 1])/soundspeed2,
                        cp2,
                        self._compressor._T, self._compressor._P, self._compressor._densities, 
                        self._partindex - 1,
                        self._partindex
                    ]
                ) #vaneless diffuser
                """
                
                self._outlet_velocity[1] = self._losses_models[1].compute_loss(
                    type= "Coppage",
                    args_minimal= [
                        self._compressor._parts[1]._outlet_velocity[0, 1],
                        self._outlet_velocity,
                        self._compressor._parts[1]._geometry._outlet_blade_height,
                        self._geometry._outlet_height,
                        visc,
                        self._compressor._parts[1]._geometry._outlet_diameter,
                        self._geometry._outlet_diameter,
                        self._compressor._massflow
                    ],
                    args= [
                        np.arctan(self._compressor._parts[1]._tanalpha2)
                    ]
                ) #vaneless diffuser
                

                sum_losses = self._losses_models[0]._value + self._losses_models[1]._value

                #compute corrected thermodynamic quantities for the isentropic point
                s_tilde = self._compressor._entropy[self._partindex - 1] #isentropic
                h_tilde = self._compressor._enthalpy[self._partindex] - sum_losses

                ds = s_tilde - s
                dh = h_tilde - self._compressor._enthalpy_isentropic[self._partindex]

                if (np.abs(dh) < self._compressor._criterion_convergence) and (np.abs(ds) < self._compressor._criterion_convergence):
                    self._isconverged[0] = True
                    break

                #change T, P
                self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex], self._compressor._Tisentropic[self._partindex])
                temp_p = self._compressor._P[self._partindex]
                temp_t = self._compressor._Tisentropic[self._partindex]
                P_tilde = self._compressor._P[self._partindex] + \
                    dh*self._compressor._gas.first_partial_deriv(CoolProp.iP, CoolProp.iHmass,CoolProp.iSmass) +\
                    ds*self._compressor._gas.first_partial_deriv(CoolProp.iP, CoolProp.iSmass,CoolProp.iHmass)
                self._compressor._Tisentropic[self._partindex] = self._compressor._Tisentropic[self._partindex] + \
                    dh*self._compressor._gas.first_partial_deriv(CoolProp.iT, CoolProp.iHmass,CoolProp.iP) +\
                    (P_tilde - self._compressor._P[self._partindex])*self._compressor._gas.first_partial_deriv(CoolProp.iT, CoolProp.iP,CoolProp.iHmass)
                self._compressor._P[self._partindex] = P_tilde

            h_tilde = self._compressor._enthalpy[self._partindex - 1] + 0.5*(
                np.linalg.norm(self._compressor._parts[1]._outlet_velocity[0, 1])**2 -\
                np.linalg.norm(self._outlet_velocity)**2)
            dh = h_tilde - self._compressor._enthalpy[self._partindex]
            
            self._h_convergence_vanelessdiffuser = self._h_convergence_vanelessdiffuser + [np.abs(dh)]

            if (np.abs(dh) < self._compressor._criterion_convergence):
                self._isconverged[1] = True
                break

            self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex], self._compressor._T[self._partindex])
            temp_t = self._compressor._T[self._partindex]
            self._compressor._T[self._partindex] = self._compressor._T[self._partindex] + \
                    dh*self._compressor._gas.first_partial_deriv(CoolProp.iT, CoolProp.iHmass,CoolProp.iP)
        
        return 0

    def save_compressorpart(self, file):
        file.write(self._name + "\n")
        file.write("\t" + "_outlet_velocity =" + '\n'+2*'\t')
        np.savetxt(file, self._outlet_velocity, delimiter=' ', newline='\n'+2*'\t')
        file.write("\n\t" + "_h_convergence_vanelessdiffuser = {}\n".format(self._h_convergence_vanelessdiffuser))
        file.write("\t" + "_isconverged = {}\n".format(self._isconverged))
        file.write("\t" + "_geometry.\n")
        file.write("\t"*2 + "_outlet_diameter_old = {}\n".format(self._geometry._outlet_diameter_old))
        file.write("\t"*2 + "_outlet_diameter = {}\n".format(self._geometry._outlet_diameter))
        file.write("\t"*2 + "_outlet_height = {}\n".format(self._geometry._outlet_height))
        file.write("\t"*2 + "_outlet_area = {}\n".format(self._geometry._outlet_area))
        file.write("\t" + "_losses_models.\n")
        for i in self._losses_models:
            file.write("\t"*2 + i._name + "\n")
            file.write("\t"*3 + "_value = {}".format(i._value) + "\n")
            file.write("\t"*3 + "_type = {}".format(i._type) + "\n")
            if "Mixing" == i._name:
                continue
            elif "Vaneless diffuser" == i._name:
                file.write("\t"*3 + "_equation = {}".format(i._equation) + "\n")
                file.write("\t"*3 + "_kcfvld = {}".format(i._kcfvld) + "\n")
                file.write("\t"*3 + "_cf = {}".format(i._cf) + "\n")
                file.write("\t"*3 + "_Re = {}".format(i._Re) + "\n")
            else:
                file.write("\t"*3 + "Loss model found but dump save not implemented\n")

class IGV(Compressorpart):
    _name = "IGV"
    def __init__(self, compressor=Compressor()):
        super().__init__(compressor)
        self._partindex = 1
    
    def solve(self):
        self._compressor._T[self._partindex] = self._compressor._T[self._partindex - 1]
        self._compressor._Tisentropic[self._partindex] = self._compressor._T[self._partindex]
        self._compressor._P[self._partindex] = self._compressor._P[self._partindex - 1]
        r = self.update_thermodynamics()
        if -1 == r: return r
        return 0

    def save_compressorpart(self, file):
        file.write(self._name + "\n")
        file.write("\t" + "not implemented yet")

class VanedDiffuser(Compressorpart):
    _name = "VanedDiffuser"
    def __init__(self, compressor=Compressor()):
        super().__init__(compressor)
        self._partindex = [5, 6]

    def solve(self):
        #Throat
        self._compressor._T[self._partindex[0]] = self._compressor._T[self._partindex[0] - 1]
        self._compressor._Tisentropic[self._partindex[0]] = self._compressor._T[self._partindex[0]]
        self._compressor._P[self._partindex[0]] = self._compressor._P[self._partindex[0] - 1]
        #Vaned diffuser
        self._compressor._T[self._partindex[1]] = self._compressor._T[self._partindex[1] - 1]
        self._compressor._Tisentropic[self._partindex[1]] = self._compressor._T[self._partindex[1]]
        self._compressor._P[self._partindex[1]] = self._compressor._P[self._partindex[1] - 1]
        r = self.update_thermodynamics()
        if -1 == r: return r
        return 0

    def save_compressorpart(self, file):
        file.write(self._name + "\n")
        file.write("\t" + "not implemented yet")

class Impeller(Compressorpart):
    _name = "Impeller"
    def __init__(self, compressor=Compressor(), materials= mechanics.Materials()):
        super().__init__(compressor)
        self._inlet_velocity = np.zeros((3,3,2)) #first index for V W U ; second index hub mid tip ; last index for meridionnal (radial) and tangential
        self._outlet_velocity = np.zeros((3,3,2)) #first index for V W U ; second index hub mid tip ; last index for meridionnal (radial) and tangential
        self._throat_velocity = 0 #W throat, others not useful?
        self._Leul = 0
        self._R = 0
        self._tanalpha2 = 0 #outlet flow angle

        self._h_convergence_throat = []
        self._h_convergence_impeller = []
        self._isconverged_throat = False
        self._isconverged_impeller = [False, False] #isentropic, given pressure loop

        self._partindex = [2, 3] #throat and outlet

        self._materials = materials

        self._MOSdisc = 0
        self._MOSblade = 0

    # _inlet_velocity : 3x3x2 array of floats
    #                   velocities V, W, U for hub, mid and tip line.
    #                   V is the absolute velocity. W is the relative velocity. U is the impeller tangential velocity.\n
    #                   _inlet_velocity[0, 0, 0] = V hub meridional\n
    #                   _inlet_velocity[1, 0, 0] = W hub meridional\n
    #                   _inlet_velocity[1, 2, 1] = W tip tangential\n
    # _outlet_velocity : 3x3x2 array of floats
    #                   velocities V, W, U for hub, mid and tip line.
    #                   V is the absolute velocity. W is the relative velocity. U is the impeller tangential velocity.\n
    #                   _outlet_velocity[0, 0, 0] = V hub meridional\n
    #                   _outlet_velocity[1, 0, 0] = W hub meridional\n
    #                   _outlet_velocity[1, 2, 1] = W tip tangential\n

    def compute_outlet_velocity_impeller(self):
        """
        Compute the velocities of the impeller

        Compute the velocities of the impeller at inlet, outlet and throat using the velocity triangle and equations from S. Parisi PhD thesis (see methods called)
        """
        
        self._outlet_velocity[2, :, 1] = np.array(self._geometry._outlet_diameter)/2*self._compressor._speed #U outlet
        self._outlet_velocity[2, :, 0] = 0  #U outlet
        self.compute_velocity_massflow(density= self._compressor._densities[self._partindex[1]], zone="outlet") #V, W outlet radial
        self.compute_relative_velocity_outlet() #W outlet tan
        self._outlet_velocity[0, :, 1] = self._outlet_velocity[1, :, 1] + self._outlet_velocity[2, :, 1] #V outlet tan

        self._R = 1 - 0.5*self._outlet_velocity[0, 1, 1]/self._outlet_velocity[2, 1, 1]
        self._tanalpha2 = self._outlet_velocity[0, 1, 1]/self._outlet_velocity[0, 1, 0]

    def compute_inlet_velocity_impeller(self):
        """
        Compute the velocities of the impeller at inlet

        Compute the velocities of the impeller at inlet using the velocity triangle and equations from S. Parisi PhD thesis (see methods called)
        No induced swirl considered
        """
        self._inlet_velocity[2, :, 1] = np.array(self._geometry._inlet_diameters)/2*self._compressor._speed #U inlet
        self._inlet_velocity[2, :, 0] = 0 #U meridional = 0
        self.compute_velocity_massflow(density= self._compressor._densities[self._partindex[0] - 1], zone="inlet") #V, W inlet radial
        self._inlet_velocity[0, :, 1] = 0 #V inlet tan V purely radial
        self._inlet_velocity[1, :, 1] = self._inlet_velocity[0, :, 1] - self._inlet_velocity[2, :, 1] #W inlet tan

    def compute_throat_velocity_impeller(self):
        """
        Compute the velocities of the impeller at inlet

        Compute the velocities of the impeller at inlet using the velocity triangle and equations from S. Parisi PhD thesis (see methods called)
        """
        self.compute_velocity_massflow(density= self._compressor._densities[self._partindex[0]], zone="throat")

    def compute_velocity_massflow(self, density, zone="inlet"):
        """
        Compute the radial component of the absolute velocity 

        Equation 2.2 of S. Parisi PhD thesis

        Parameters
        ----------
        massflow : float
            mass flow of the impeller
        density : float
            density of the mixture or gas
        zone : str, optional
            "inlet" or "outlet", zone where the velocity is computed, by default "inlet"

        Raises
        ------
        NotImplementedError
            When using with a not valid zone, will raise the exception
        """
        if "inlet" == zone:
            self._inlet_velocity[0:2, :, 0] = self._compressor._massflow/(density * self._geometry._inlet_area)
        elif "outlet" == zone:
            self._outlet_velocity[0:2, :, 0] = self._compressor._massflow/(density * self._geometry._outlet_area)
        elif "throat" == zone:
            self._throat_velocity = self._compressor._massflow/(density*self._geometry._throat_area)
        else:
            raise NotImplementedError("compute_absolute_velocity_radial for inlet, throat or outlet only")

    def compute_slip_factor(self):
        """
        Compute the slip factor of the impeller

        Equation 2.6 to 2.8 of S. Parisi PhD thesis

        Returns
        -------
        float
            Slip factor according to equation 2.6 to 2.8 of S. Parisi PhD thesis
        """
        sigma = 1 - np.sqrt(np.cos(self._geometry._outlet_blade_angle))/self._geometry._effective_number_blade**0.7
        eps = np.exp(-8.16*np.sin(self._geometry._outlet_blade_angle)/self._geometry._effective_number_blade)
        if (self._geometry._inlet_diameters[1]/self._geometry._outlet_diameter > eps):
            sigma = sigma*(1 - ((self._geometry._inlet_diameters[1]/self._geometry._outlet_diameter - eps)/(1 - eps))**3)
        return sigma

    def compute_relative_velocity_outlet(self):
        """
        Compute the relative speed at the outlet of the impeller

        Equations 2.4 to 2.9 of S. Parisi PhD thesis
        """
        w2inf = self._outlet_velocity[1, :, 0]*np.tan(self._geometry._outlet_blade_angle)
        sigma = self.compute_slip_factor()
        self._outlet_velocity[1, :, 1] = w2inf - self._outlet_velocity[2, :, 1]*(1 - sigma)

    def compute_eulerianwork(self):
        self._Leul = self._outlet_velocity[2, 2, 1]*self._outlet_velocity[0, 2, 1] - self._inlet_velocity[2, 2, 1]*self._inlet_velocity[0, 2, 1]

    def update_losses(self, show=False):
        """
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
        """
        self._losses_models[0].compute_loss(self._inlet_velocity, self._geometry._inlet_blade_angle, self._geometry._inlet_optimal_angle)

        #processgas.iisentropic_expansion_coefficient() NOT WORKING
        #shock loss computed with inlet data
        self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex[0] - 1], self._compressor._T[self._partindex[0] - 1])
        kpv = CoolProp.CoolProp.PropsSI(
            'ISENTROPIC_EXPANSION_COEFFICIENT', 
            'T', self._compressor._T[self._partindex[0] - 1], 
            'P', self._compressor._P[self._partindex[0] - 1], "REFPROP::" + self._compressor._gas.fluid_names()[0])
        self._losses_models[1].compute_loss(
            kpv, 
            self._compressor._T[self._partindex[0] - 1],
            self._inlet_velocity,
            self._compressor._gas.speed_sound(),
            self._compressor._gas.first_partial_deriv(CoolProp.iHmass, CoolProp.iT,CoolProp.iP)
        )

        self._losses_models[2].compute_loss(
            np.linalg.norm(self._inlet_velocity[1, 1]),
            np.linalg.norm(self._inlet_velocity[1, -1]),
            self._throat_velocity,
            self._losses_models[0]._value,
            self._losses_models[0]._finc
        )

        #ChokingLoss computed with inlet data
        self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex[0]], self._compressor._T[self._partindex[0]])
        self._losses_models[3].compute_loss(
            np.abs(self._throat_velocity)/self._compressor._gas.speed_sound(),
            kpv, np.linalg.norm(self._inlet_velocity[1, 1]))

        #SupercriticalLoss computed with inlet data
        self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex[0]], self._compressor._T[self._partindex[0]])
        athroat = self._compressor._gas.speed_sound()
        self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex[0] - 1], self._compressor._T[self._partindex[0] - 1])
        self._losses_models[4].compute_loss(
            self._geometry._outlet_diameter,
            self._Leul,
            self._geometry._effective_number_blade,
            self._geometry._blade_length,
            self._inlet_velocity,
            self._outlet_velocity,
            np.linalg.norm(self._inlet_velocity[1, 1])/self._compressor._gas.speed_sound(),
            athroat
        )

        #BladeloadingLoss
        self._losses_models[5].compute_loss('Whitfield', #Whitfield, Coppage
            np.linalg.norm(self._outlet_velocity[1, 1]),
            np.linalg.norm(self._inlet_velocity[1, -1]),
            self._Leul,
            self._geometry._effective_number_blade,
            self._geometry._outlet_diameter,
            self._geometry._inlet_diameters[-1],
            self._outlet_velocity[-1, 1, 1],
            self._geometry._inlet_diameters[0],
            self._geometry._outlet_blade_height,
            self._geometry._blade_length
        )


        #SkinfrictionLoss
        self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex[0] - 1], self._compressor._T[self._partindex[0] - 1])
        nu = np.array([self._compressor._gas.viscosity()/self._compressor._densities[self._partindex[0] - 1]])
        self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex[1]], self._compressor._T[self._partindex[1]])
        nu[-1] = self._compressor._gas.viscosity()/self._compressor._densities[self._partindex[1]]
        self._losses_models[6].compute_loss(
            self._geometry._blade_length,
            self._geometry._hydraulic_diameter,
            self._geometry._Ra,
            nu,
            self._inlet_velocity,
            self._outlet_velocity
        )
        
        #ClearanceLoss
        self._losses_models[7]._clearance = self._geometry._blade_running_clearance
        self._losses_models[7].compute_loss(
            self._outlet_velocity,
            self._inlet_velocity[0, 1, 0],
            self._geometry._outlet_diameter,
            self._geometry._inlet_diameters[-1],
            self._geometry._inlet_diameters[0],
            self._compressor._densities[self._partindex[1]], 
            self._compressor._densities[self._partindex[0] - 1],
            self._geometry._effective_number_blade,
            self._geometry._outlet_blade_height
        )

        #RecirculationLoss
        bll = losses.BladeloadingLoss()
        diffusion_factor = 0
        if self._losses_models[5]._equation == "Coppage":
            diffusion_factor = self._losses_models[5]._d
        else:
            diffusion_factor = bll.compute_diffusion_Coppage(
                np.linalg.norm(self._outlet_velocity[1, 1]),
                np.linalg.norm(self._inlet_velocity[1, -1]),
                self._Leul, self._geometry._effective_number_blade,
                self._geometry._outlet_diameter,
                self._geometry._inlet_diameters[-1],
                self._outlet_velocity[-1, 1, 1])
        del bll
        self._losses_models[8].compute_loss('Coppage', #Coppage, Oh
            self._tanalpha2,
            diffusion_factor,
            self._outlet_velocity[2, 1, 1]
        )

        #LeakageLoss
        self._losses_models[9].compute_loss("open",
            [
                self._compressor._massflow,
                self._inlet_velocity,
                self._outlet_velocity,
                self._geometry._inlet_diameters,
                self._geometry._outlet_diameter,
                self._geometry._outlet_blade_height,
                self._geometry._blade_length_meanline,
                self._geometry._effective_number_blade,
                self._compressor._densities[self._partindex[1]],
                self._geometry._blade_running_clearance
            ]
        )

        #DiscfrictionLoss
        self._losses_models[10].compute_loss(
            self._outlet_velocity,
            self._geometry._outlet_diameter,
            self._compressor._densities[self._partindex[1]],
            self._compressor._massflow,
            nu[-1]
        )

        if show:
            for i in self._losses_models:
                print("{} value = {:.0f}".format(i._name, i._value))

    def solve(self, show=False):
        self.compute_inlet_velocity_impeller()
        r = self.solve_throat()
        if -1 == r:
            print("Error in solve_throat()\n\tWthroat = {:.2f}\n\tT = {:.2f} K\tP = {:.2f} bara\n\tdh = {} J/kg/K".format(
                  self._throat_velocity, self._compressor._T[self._partindex[0]], self._compressor._P[self._partindex[0]]/physics.bar2Pa(1), self._h_convergence_throat))
            return r
        r = self.solve_outlet(show)
        if -1 == r: 
            print("Error in solve_outlet()\n\tT = {:.2f} K\n\tT_isentropic = {:.2f}\n\tP = {:.2f} bara\n\tdh = {} J/kg/K".format(
                self._compressor._T[self._partindex[1]], 
                self._compressor._Tisentropic[self._partindex[1]], 
                self._compressor._P[self._partindex[1]]/physics.bar2Pa(1), self._h_convergence_impeller))
            return r
        return 0

    def solve_throat(self):
        self._compressor._T[self._partindex[0]] = self._compressor._T[self._partindex[0] - 1]
        self._compressor._Tisentropic[self._partindex[0]] = self._compressor._T[self._partindex[0]]
        self._compressor._P[self._partindex[0]] = self._compressor._P[self._partindex[0] - 1]

        for i in range(0, self._compressor._max_iteration):
            #compute thermodynamic quantities
            r = self.update_thermodynamics()
            if -1 == r: return r
            #compute speed
            self.compute_throat_velocity_impeller()
            #compute corrected thermodynamic quantities
            s_tilde = self._compressor._entropy[self._partindex[0] - 1] #isentropic
            h_tilde = self._compressor._enthalpy[self._partindex[0] - 1] + 0.5*np.linalg.norm(self._inlet_velocity[1, 1])**2 \
                    - 0.5*self._throat_velocity**2

            ds = s_tilde - self._compressor._entropy[self._partindex[0]]
            dh = h_tilde - self._compressor._enthalpy[self._partindex[0]]

            self._h_convergence_throat = self._h_convergence_throat + [np.abs(dh)]

            if (np.abs(dh) < self._compressor._criterion_convergence) and (np.abs(ds) < self._compressor._criterion_convergence):
                self._isconverged_throat = True
                break

            if np.abs(dh/self._compressor._enthalpy[self._partindex[0]]) > self._varmax:
                dh = np.sign(dh)*self._varmax*self._compressor._enthalpy[self._partindex[0]]
            if np.abs(ds/self._compressor._entropy[self._partindex[0]]) > self._varmax:
                ds = np.sign(ds)*self._varmax*self._compressor._entropy[self._partindex[0]]

            if (self._throat_velocity < np.linalg.norm(self._inlet_velocity[1, 1])):
                #change T, P
                self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex[0]], self._compressor._T[self._partindex[0]])
                P_tilde = self._compressor._P[self._partindex[0]] + \
                    dh*self._compressor._gas.first_partial_deriv(CoolProp.iP, CoolProp.iHmass,CoolProp.iSmass) +\
                    ds*self._compressor._gas.first_partial_deriv(CoolProp.iP, CoolProp.iSmass,CoolProp.iHmass)
                self._compressor._T[self._partindex[0]] = self._compressor._T[self._partindex[0]] + \
                    dh*self._compressor._gas.first_partial_deriv(CoolProp.iT, CoolProp.iHmass,CoolProp.iP) +\
                    (P_tilde - self._compressor._P[self._partindex[0]])*self._compressor._gas.first_partial_deriv(CoolProp.iT, CoolProp.iP,CoolProp.iHmass)
                self._compressor._P[self._partindex[0]] = P_tilde
            else:
                d = self._compressor._densities[self._partindex[0]]
                self._compressor._densities[self._partindex[0]] = self._compressor._massflow/(np.linalg.norm(self._inlet_velocity[1, 1]) * 0.9 * self._geometry._throat_area)
                self._compressor._entropy[self._partindex[0]] = self._compressor._entropy[self._partindex[0] - 1]
                """print("Throat velocity too high (Wthroat = {} = {}/({}*{}) vs W1mid = {}). Adjust density = {}".format(
                    self._throat_velocity, self._compressor._massflow, d, self._geometry._throat_area,
                    np.linalg.norm(self._inlet_velocity[1, 1]), self._compressor._densities[self._partindex[0]]))"""

                self._compressor._gas.update(CoolProp.DmassSmass_INPUTS, self._compressor._densities[self._partindex[0]], self._compressor._entropy[self._partindex[0]])
                self._compressor._T[self._partindex[0]] = self._compressor._gas.T()
                self._compressor._P[self._partindex[0]] = self._compressor._gas.p()
        
        return 0

    def solve_outlet(self, show=False):
        self._compressor._T[self._partindex[1]] = self._compressor._T[self._partindex[0] - 1]
        self._compressor._Tisentropic[self._partindex[1]] = self._compressor._T[self._partindex[1]]
        self._compressor._P[self._partindex[1]] = self._compressor._P[self._partindex[0] - 1]

        #given pressure loop for the real point
        for j in range(0, self._compressor._max_iteration):
            #compute thermodynamic quantities
            self._isconverged_impeller[0] = False


            #isentropic loop for the isentropic point
            for i in range(0, self._compressor._max_iteration):
                #compute thermodynamic quantities
                #Real point
                r = self.update_thermodynamics()
                if -1 == r: 
                    print("Error in Impeller.solve_outlet")
                    print("\t_T = {}\n\t_P = {}".format(self._compressor._T, self._compressor._P))
                    return r
                #Isentropic point
                self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex[1]], self._compressor._Tisentropic[self._partindex[1]])
                s = self._compressor._gas.smass() #entropy J/kg/K

                #compute speed
                self.compute_outlet_velocity_impeller()
                self.compute_eulerianwork()

                #compute losses
                self.update_losses(show=show)
                sum_internal = np.sum([k._value if "internal" == k._type else 0 for k in self._losses_models])
                sum_parasitic = np.sum([k._value if "parasitic" == k._type else 0 for k in self._losses_models])

                #compute corrected thermodynamic quantities for the isentropic point
                rhs = 0.5*np.linalg.norm(self._inlet_velocity[0, 1])**2 \
                    + self._Leul - sum_internal \
                    - 0.5*np.linalg.norm(self._outlet_velocity[0, 1])**2
                s_tilde = self._compressor._entropy[self._partindex[0] - 1] #isentropic
                h_tilde = self._compressor._enthalpy[self._partindex[0] - 1] + rhs

                ds = s_tilde - s
                dh = h_tilde - self._compressor._enthalpy_isentropic[self._partindex[1]]
                
                if (np.abs(dh) < self._compressor._criterion_convergence) and (np.abs(ds) < self._compressor._criterion_convergence):
                    self._isconverged_impeller[0] = True
                    break

                if np.abs(dh/self._compressor._enthalpy[self._partindex[0]]) > self._varmax:
                    dh = np.sign(dh)*self._varmax*self._compressor._enthalpy[self._partindex[0]]
                if np.abs(ds/self._compressor._entropy[self._partindex[0]]) > self._varmax:
                    ds = np.sign(ds)*self._varmax*self._compressor._entropy[self._partindex[0]]

                #change T, P
                self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex[1]], self._compressor._Tisentropic[self._partindex[1]])
                temp_p = self._compressor._P[self._partindex[1]]
                temp_t = self._compressor._Tisentropic[self._partindex[1]]
                P_tilde = self._compressor._P[self._partindex[1]] + \
                    dh*self._compressor._gas.first_partial_deriv(CoolProp.iP, CoolProp.iHmass,CoolProp.iSmass) +\
                    ds*self._compressor._gas.first_partial_deriv(CoolProp.iP, CoolProp.iSmass,CoolProp.iHmass)
                self._compressor._Tisentropic[self._partindex[1]] = self._compressor._Tisentropic[self._partindex[1]] + \
                    dh*self._compressor._gas.first_partial_deriv(CoolProp.iT, CoolProp.iHmass,CoolProp.iP) +\
                    (P_tilde - self._compressor._P[self._partindex[1]])*self._compressor._gas.first_partial_deriv(CoolProp.iT, CoolProp.iP,CoolProp.iHmass)
                self._compressor._P[self._partindex[1]] = P_tilde

                     
            h_tilde = self._compressor._enthalpy[self._partindex[0] - 1] + 0.5*np.linalg.norm(self._inlet_velocity[0, 1])**2 +\
                    self._Leul + \
                    sum_parasitic - \
                    0.5*np.linalg.norm(self._outlet_velocity[0, 1])**2

            dh = h_tilde - self._compressor._enthalpy[self._partindex[1]]

            if np.abs(dh/self._compressor._enthalpy[self._partindex[1]]) > self._varmax:
                    dh = np.sign(dh)*self._varmax*self._compressor._enthalpy[self._partindex[1]]
            
            self._h_convergence_impeller = self._h_convergence_impeller + [np.abs(dh)]

            """if (dh < 0):
                print("dh = {:.3f}".format(dh))
                print("sum parasitic = {:.3f}".format(sum_parasitic/1000))
                print("Leul = {:.3f}".format(self._Leul/1000))
                print("0.5*V2**2 = {:.3f}".format(0.5*np.linalg.norm(self._outlet_velocity[0, 1])**2/1000))
                print("h1 = {:.3f}".format(self._compressor._enthalpy[self._partindex[0] - 1]/1000))
                print("0.5*V1**2 = {:.3f}".format(0.5*np.linalg.norm(self._inlet_velocity[0, 1])**2/1000))
                print("\th1t = {:.3f}".format((self._compressor._enthalpy[self._partindex[0] - 1] + 0.5*np.linalg.norm(self._inlet_velocity[0, 1])**2)/1000))
                print("self._compressor._enthalpy[self._partindex[1]] = {}".format(self._compressor._enthalpy[self._partindex[1]]/1000))
                print("T = {}\tP = {}".format(self._compressor._T[self._partindex[1]], self._compressor._P[self._partindex[1]]/physics.bar2Pa(1)))
                print("j = {}".format(j))
                print("self._h_convergence_impeller = {}".format(self._h_convergence_impeller))
                for i in range(0, len(self._losses_models)):
                    if "parasitic" == self._losses_models[i]._type:
                        print("{} = {}".format(self._losses_models[i]._name, self._losses_models[i]._value/1000))"""
                

            if (np.abs(dh) < self._compressor._criterion_convergence):
                self._isconverged_impeller[1] = True
                break

            self._compressor._gas.update(CoolProp.PT_INPUTS, self._compressor._P[self._partindex[1]], self._compressor._T[self._partindex[1]])
            temp_t = self._compressor._T[self._partindex[1]]
            self._compressor._T[self._partindex[1]] = self._compressor._T[self._partindex[1]] + \
                    dh*self._compressor._gas.first_partial_deriv(CoolProp.iT, CoolProp.iHmass,CoolProp.iP)
        
        return 0

    def get_discstress(self):
        """
            Computes the disc stress
            
            Return
            -------
            float
                Disc stress
            float
                Margin of safety for the materials in comparison with the tensile yield stress
        """
        ds = mechanics.DiscStress()
        stress = ds.compute_stress(self._materials._density, self._outlet_velocity[2, 1, 1], self._materials._poisson)
        del ds
        self._MOSdisc = self._materials.get_MOS(stress)
        return stress, self._MOSdisc

    def get_bladestress(self):
        """
        """
        bs = mechanics.BladeStress()
        s = bs.compute_stress(
            self._geometry._outlet_blade_height,
            self._materials._density,
            self._geometry._outlet_diameter,
            self._compressor._speed,
            self._geometry._outlet_blade_angle,
            self._geometry._outlet_average_blade_thickness,
            self._geometry._taperratio,
            self._geometry._tapertype
            )
        del bs
        self._MOSblade = self._materials.get_MOS(s)
        return s, self._MOSblade

    def save_compressorpart(self, file):
        file.write(self._name + "\n")
        file.write("\t" + "_inlet_velocity V =" + '\n'+2*'\t')
        np.savetxt(file, self._inlet_velocity[0], delimiter=' ', newline='\n'+2*'\t')
        file.write("\n")
        file.write("\t" + "_inlet_velocity W =" + '\n'+2*'\t')
        np.savetxt(file, self._inlet_velocity[1], delimiter=' ', newline='\n'+2*'\t')
        file.write("\n")
        file.write("\t" + "_inlet_velocity U =" + '\n'+2*'\t')
        np.savetxt(file, self._inlet_velocity[2], delimiter=' ', newline='\n'+2*'\t')
        file.write("\n")
        file.write("\t" + "_outlet_velocity V =" + '\n'+2*'\t')
        np.savetxt(file, self._outlet_velocity[0], delimiter=' ', newline='\n'+2*'\t')
        file.write("\n")
        file.write("\t" + "_outlet_velocity W =" + '\n'+2*'\t')
        np.savetxt(file, self._outlet_velocity[1], delimiter=' ', newline='\n'+2*'\t')
        file.write("\n")
        file.write("\t" + "_outlet_velocity U =" + '\n'+2*'\t')
        np.savetxt(file, self._outlet_velocity[2], delimiter=' ', newline='\n'+2*'\t')
        file.write("\n")
        file.write("\t" + "_throat_velocity = {}\n".format(self._throat_velocity))
        file.write("\t" + "_Leul = {}\n".format(self._Leul))
        file.write("\t" + "_R = {}\n".format(self._R))
        file.write("\t" + "_tanalpha2 = {}\n".format(self._tanalpha2))
        file.write("\t" + "_h_convergence_throat = {}\n".format(self._h_convergence_throat))
        file.write("\t" + "_isconverged_throat = {}\n".format(self._isconverged_throat))
        file.write("\t" + "_h_convergence_impeller = {}\n".format(self._h_convergence_impeller))
        file.write("\t" + "_isconverged_impeller = {}\n".format(self._isconverged_impeller))
        file.write("\t" + "_MOSdisc = {}\n".format(self._MOSdisc))
        file.write("\t" + "_MOSblade = {}\n".format(self._MOSblade))

        file.write("\t" + "_materials.\n")
        file.write("\t"*2 + "_name = {}".format(self._materials._name) + "\n")
        file.write("\t"*2 + "_youngmodulus = {}".format(self._materials._youngmodulus) + "\n")
        file.write("\t"*2 + "_poisson = {}".format(self._materials._poisson) + "\n")
        file.write("\t"*2 + "_yieldstress = {}".format(self._materials._yieldstress) + "\n")
        file.write("\t"*2 + "_density = {}".format(self._materials._density) + "\n")
        file.write("\t"*2 + "_ksigma = {}".format(mechanics.DiscStress()._ksigma) + "\n")

        file.write("\t" + "_geometry.\n")
        file.write("\t"*2 + "_inlet_diameters = {}\n".format(self._geometry._inlet_diameters))
        file.write("\t"*2 + "_inlet_blade_thickness = {}\n".format(self._geometry._inlet_blade_thickness))
        file.write("\t"*2 + "_inlet_blade_angle = {}\n".format(np.rad2deg(self._geometry._inlet_blade_angle)))
        file.write("\t"*2 + "_outlet_diameter = {}\n".format(self._geometry._outlet_diameter))
        file.write("\t"*2 + "_outlet_blade_height = {}\n".format(self._geometry._outlet_blade_height))
        file.write("\t"*2 + "_outlet_blade_thickness = {}\n".format(self._geometry._outlet_blade_thickness))
        file.write("\t"*2 + "_outlet_average_blade_thickness = {}\n".format(self._geometry._outlet_average_blade_thickness))
        file.write("\t"*2 + "_outlet_blade_angle = {}\n".format(np.rad2deg(self._geometry._outlet_blade_angle)))
        file.write("\t"*2 + "_axial_extension = {}\n".format(self._geometry._axial_extension))
        file.write("\t"*2 + "_blade_running_clearance = {}\n".format(self._geometry._blade_running_clearance))
        file.write("\t"*2 + "_number_blade_full = {}\n".format(self._geometry._number_blade_full))
        file.write("\t"*2 + "_splitter_blade_length_fraction = {}\n".format(self._geometry._splitter_blade_length_fraction))
        file.write("\t"*2 + "_blade_length = {}\n".format(self._geometry._blade_length))
        file.write("\t"*2 + "_effective_number_blade = {}\n".format(self._geometry._effective_number_blade))
        file.write("\t"*2 + "_full_number_blade_w_splitters = {}\n".format(self._geometry._full_number_blade_w_splitters))
        file.write("\t"*2 + "_hydraulic_diameter = {}\n".format(self._geometry._hydraulic_diameter))
        file.write("\t"*2 + "_inlet_area = {}\n".format(self._geometry._inlet_area))
        file.write("\t"*2 + "_throat_pitch_blade = {}\n".format(self._geometry._throat_pitch_blade))
        file.write("\t"*2 + "_throat_width = {}\n".format(self._geometry._throat_width))
        file.write("\t"*2 + "_throat_area = {}\n".format(self._geometry._throat_area))
        file.write("\t"*2 + "_outlet_area = {}\n".format(self._geometry._outlet_area))
        file.write("\t"*2 + "_inlet_optimal_angle = {}\n".format(self._geometry._inlet_optimal_angle))
        file.write("\t"*2 + "_parameter_angle = {}\n".format(self._geometry._parameter_angle))
        file.write("\t"*2 + "_Ra = {}\n".format(self._geometry._Ra))
        file.write("\t"*2 + "_taperratio = {}\n".format(self._geometry._taperratio))
        file.write("\t"*2 + "_tapertype = {}\n".format(self._geometry._tapertype))

        file.write("\t" + "_losses_models.\n")
        for i in self._losses_models:
            file.write("\t"*2 + i._name + "\n")
            file.write("\t"*3 + "_type = {}".format(i._type) + "\n")
            file.write("\t"*3 + "_value = {}".format(i._value) + "\n")
            if i._name in ["Shock", "Choking", "Recirculation"]:
                continue
            elif "Incidence" == i._name:
                file.write("\t"*3 + "_finc = {}".format(i._finc) + "\n")
            elif "Diffusion" == i._name:
                file.write("\t"*3 + "{}".format(i._difforstall) + "\n")
            elif "Supercritical" == i._name:
                file.write("\t"*3 + "_fsup = {}".format(i._fsup) + "\n")
                file.write("\t"*3 + "_wmax = {}".format(i._wmax) + "\n")
                file.write("\t"*3 + "_Mcr = {}".format(i._Mcr) + "\n")
                file.write("\t"*3 + "_mach = {}".format(i._mach) + "\n")
            elif "Blade loading" == i._name:
                file.write("\t"*3 + "_d = {}".format(i._d) + "\n")
                file.write("\t"*3 + "_equation = {}".format(i._equation) + "\n")
            elif "Skin friction" == i._name:
                file.write("\t"*3 + "_Ksf = {}".format(i._Ksf) + "\n")
            elif "Clearance" == i._name:
                file.write("\t"*3 + "_clearance = {}".format(i._clearance) + "\n")
            elif "Leakage" == i._name:
                file.write("\t"*3 + "_equation = {}".format(i._equation) + "\n")
                if "open" == i._type:
                    file.write("\t"*3 + "_clearancemassflow = {}".format(i._clearancemassflow) + "\n")
                elif "closed" == i._type:
                    file.write("\t"*3 + "_cd = {}".format(i._cd) + "\n")                              
            elif "Disc friction" == i._name:
                file.write("\t"*3 + "_Re = {}".format(i._Re) + "\n")
                file.write("\t"*3 + "_frictionfactor = {}".format(i._frictionfactor) + "\n")
            else:
                file.write("\t"*3 + "Loss model found but dump save not implemented\n")

    def save_impeller_geometry(self, nb_additional_lines= 1, filename="./DUMP save/" + "Impeller geometry" + str(time.time()) + ".txt", force= False, adjustthickness= False):
        if (type(self._geometry._Xt) is not np.ndarray) or (force): self._geometry.compute_blades(nb_additional_lines= nb_additional_lines, nb_theta=200, adjustthickness= adjustthickness)
        t = np.zeros(self._geometry._r.shape)
        for idx, val in enumerate(self._geometry._mt_adim[:-1,]):
            t[idx] = self._geometry.compute_tx(1 - (self._geometry._r[idx,:] - self._geometry._r[idx,0])/(self._geometry._r[idx,-1] - self._geometry._r[idx,0]))*((1 - val)*self._geometry._inlet_blade_thickness[-1] + val*self._geometry._outlet_blade_thickness)
        t[-1] = self._geometry._outlet_blade_thickness

        with open(filename, 'a') as file:
            X, Y = self._geometry._r*np.cos(self._geometry._phi_allblades[0]), self._geometry._r*np.sin(self._geometry._phi_allblades[0])

            file.write('Full blade = \n')
            file.write('X = \n')
            np.savetxt(file, X, delimiter=' ', newline='\n')
            file.write('Y = \n')
            np.savetxt(file, Y, delimiter=' ', newline='\n')
            file.write('Z = \n')
            np.savetxt(file, self._geometry._z, delimiter=' ', newline='\n')

            file.write('\nFull blade thickness distribution in radial direction = \n')
            file.write('Thickness = \n')
            np.savetxt(file, t, delimiter=' ', newline='\n')
            file.write('Radius lines = \n')
            np.savetxt(file, self._geometry._r, delimiter=' ', newline='\n')

            if 1e-15 < self._geometry._splitter_blade_length_fraction and 1 > self._geometry._splitter_blade_length_fraction:
                X, Y = self._geometry._r*np.cos(self._geometry._phi_allblades[1]), self._geometry._r*np.sin(self._geometry._phi_allblades[1])
                indexes = np.argwhere(np.abs(self._geometry._phi_allblades[1]) > 1e-15)

                file.write('\nSplitter = \n')
                file.write('X = \n')
                np.savetxt(file, X[indexes[0, 0]:indexes[-1, 0] + 1, :], delimiter=' ', newline='\n')
                file.write('Y = \n')
                np.savetxt(file, Y[indexes[0, 0]:indexes[-1, 0] + 1, :], delimiter=' ', newline='\n')
                file.write('Z = \n')
                np.savetxt(file, self._geometry._z[indexes[0, 0]:indexes[-1, 0] + 1, :], delimiter=' ', newline='\n')

if __name__ == "__main__":
    N = 134.1 * 1000 #rpm
    omega = N * 2* np.pi/60 #rad/s
    G = 0.13 #kg/s

    gas = 'air'
    mygas = CoolProp.AbstractState('REFPROP', gas)
    Tinit = 288.15 - physics.C2K(0)
    pinit = 101325/physics.bar2Pa(1)

    compressor_1 = Compressor([], G, omega, gas=mygas, Tinit=Tinit, Pinit= pinit)
    compressor_1._max_iteration = 50
    compressor_1._criterion_convergence = 5e-10
    compressor_1._save_txt_blade = True

    print("Vitesse = {:g} rpm".format(compressor_1.return_speed_rpm()))

    #0 = IGV = None
    #1 = Impeller
    #2 = Vaneless diffuser
    #3 = Vaned diffuser = None
    #4 = Volute
    compressor_1._parts = \
        [IGV(compressor_1), #IGV
        Impeller(compressor_1, mechanics.Materials(name="TA6V", Youngmodulus= 111e9, poisson= 0.31, yieldstress= 1220e6, density= 4410)),
        VanelessDiffuser(compressor_1), #Vaneless diffuser
        VanedDiffuser(compressor_1), #Vaned diffuser
        Volute(compressor_1)] #Volute

    compressor_1._parts[1]._geometry = geometry.Impeller()
    compressor_1._parts[1]._geometry._inlet_diameters = np.array([12, 0, 38])*1e-3 #m
    compressor_1._parts[1]._geometry._inlet_blade_angle = [np.deg2rad(-30), 0, np.deg2rad(-62)]
    compressor_1._parts[1]._geometry._outlet_diameter = 52 * 1e-3 #m
    compressor_1._parts[1]._geometry._outlet_blade_height = 3.8 * 1e-3 #m
    compressor_1._parts[1]._geometry._outlet_blade_angle = np.deg2rad(-30)
    compressor_1._parts[1]._geometry._axial_extension = 14 * 1e-3 #m
    compressor_1._parts[1]._geometry._blade_running_clearance = 0.1 * 1e-3 #m
    compressor_1._parts[1]._geometry._number_blade_full = 6
    compressor_1._parts[1]._geometry._splitter_blade_length_fraction = 0.6

    compressor_1._parts[1]._geometry._inlet_blade_thickness = np.array([0, 0, 0.5])*1e-3 #m
    compressor_1._parts[1]._geometry._outlet_blade_thickness = 0.5*1e-3 #m
    compressor_1._parts[1]._geometry._parameter_angle = 0.5
    compressor_1._parts[1]._geometry._Ra = 3.5*1e-6 #m
    compressor_1._parts[1]._geometry._taperratio = 2
    compressor_1._parts[1]._geometry._tapertype = 'parabolic'
    compressor_1._parts[1]._geometry.compute_inlet_average_blade_angle()
    compressor_1._parts[1]._geometry.update_geometry(set_angle= False)
    
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
    compressor_1._parts[2]._geometry._outlet_diameter = 140 * 1e-3 #m
    compressor_1._parts[2]._geometry._outlet_height = compressor_1._parts[1]._geometry._outlet_blade_height
    compressor_1._parts[2]._geometry.compute_area()

    if (compressor_1._parts[-1]._name == 'Volute'):
        compressor_1._parts[4]._geometry = geometry.Volute()
        compressor_1._parts[4]._geometry.solve_diameter(
            compressor_1._parts[2]._geometry._outlet_area, 
            compressor_1._parts[2]._geometry._outlet_height)
        #update diffuser end diameter to take into account volute modeling
        temp_d = compressor_1._parts[2]._geometry._outlet_diameter
        compressor_1._parts[2]._geometry.update_outlet_diameter(compressor_1._parts[4]._geometry.find_offset_diffuser(compressor_1._parts[2]._geometry._outlet_height))
        print("Volute diameter = {} m".format(compressor_1._parts[4]._geometry._D))
        print("Volute area (diffuser area) = {} m^2 ({} m^2)".format(
            compressor_1._parts[4]._geometry._area,
            compressor_1._parts[2]._geometry._outlet_area))
        print("Updated diffuser outlet diameter = {} (old = {}, delta = {})".format(compressor_1._parts[2]._geometry._outlet_diameter, temp_d, compressor_1._parts[2]._geometry._outlet_diameter - temp_d))
        

    print("Optimal blade angle = {}\nActual blade angle = {}".format(np.rad2deg(compressor_1._parts[1]._geometry._inlet_optimal_angle), np.rad2deg(compressor_1._parts[1]._geometry._inlet_blade_angle)))
    print("Thicknesses inlet = {}\nAverage outlet thickness = {}".format(compressor_1._parts[1]._geometry._inlet_blade_thickness, compressor_1._parts[1]._geometry._outlet_average_blade_thickness))

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

    if (compressor_1._parts[-1]._name == 'Volute'):
        compressor_1._parts[4]._losses_models = [losses.VoluteLoss()]


    compressor_1.initialize_thermodynamics()
    for p in compressor_1._parts:
        print(p._name)
        r = p.solve()
        if (-1 == r):
            raise RuntimeError("Failure in solve for {}\n\tT = {}\n\tP = {}\n\td = {}\n\tOutlet velocities = {}".format(
                p._name, compressor_1._T, compressor_1._P, compressor_1._densities, p._outlet_velocity))
        print("\t OK {}".format(p._name))

    compressor_1.compute_NsDs()
    print("Ns = {}\tDs = {}".format(compressor_1._Ns, compressor_1._Ds))
    compressor_1.compute_efficiencies(withoutvolute= True)

    discstress, MOSds = compressor_1._parts[1].get_discstress()
    print("Disc stress = {:.2f} MPa\tMOS = {:.2f} %".format(discstress/1e6, MOSds*100))
    bladestress, MOSbs = compressor_1._parts[1].get_bladestress()
    print("Blade stress = {:.2f} MPa\tMOS = {:.2f} %".format(bladestress/1e6, MOSbs*100))

    #compressor_1.save_compressor()

    #if compressor_1._save_txt_blade: compressor_1._parts[1].save_impeller_geometry(nb_additional_lines= 1, force= True)

    #compressor_1.plot_meridionalview(nb_additional_lines= 0, show=True, force= True)
    #compressor_1.plot_3Dimpeller_convergence(nb_additional_lines= 0, show=True, force= False)