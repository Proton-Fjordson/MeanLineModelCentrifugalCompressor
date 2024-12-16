import geometry
import numpy as np
from scipy import optimize 

import matplotlib.pyplot as plt
import matplotlib

class IGVLoss():
    """Container of the Inlet Guide Vane (IGV) loss

        Attributes
        ----------
        _type : string, defaults to ``'outside'``
                Class attribute and not instance attribute. Define the type of loss
        _name : string, defaults to ``'IGV'``
                Class attribute and not instance attribute. Define the name of the loss

        Note
        ----
        To implement as needed, no IGV used yet
    """
    _type = "outside"
    _name = "IGV"
    pass

class IncidenceLoss():
    """Container of the Incidence loss for an impeller

        The incidence loss is due to the adjustement of the flow direction at the inlet of the impeller (:cite:t:`2012:Parisi`).

        Attributes
        ----------
        _type : string, defaults to ``'internal'``
            Class attribute and not instance attribute. Define the type of loss
        _name : string, defaults to ``'Incidence'``
                Class attribute and not instance attribute. Define the name of the loss
        _value : float, defaults to ``0``
                 computed massic enthalpy loss value
        _finc : float, defaults to ``0.6``
                fraction of the velocity component perpendical to the optimal flow direction, main parameter of the model\n
                usually between 0.5 and 0.7 (:cite:t:`2012:Parisi`)
    """
    _type = "internal"
    _name = "Incidence"

    def __init__(self):
        """Constructor of IncidenceLoss class

            Sets ``_finc`` to default value of ``0.6`` and ``_value`` to default value of ``0``.
        """
        self._finc = 0.6
        self._value = 0

    def compute_loss(self, inlet_velocity, inlet_blade_angle, inlet_optimal_angle):
        """Computes and updates the incidence loss value ``_value``

            The incidence loss is computed as explained in :cite:t:`2012:Parisi` and :cite:t:`2000:Aungier`.
            First the non optimal component of the speed :math:`W_{\\mathrm{nonopt}\ i}` is computed for the hub, mid and tip positions given by the subscript :math:`i` by :
            :math:`W_{\\mathrm{nonopt}\ i} = W_{1\ i} \\mathrm{sin}\\left| \\beta_{1\ i} - \\beta_{1\ \\mathrm{opti}\ i} \\right|` 
            where :math:`\\beta_{1\ \\mathrm{opti}\ i}` is the inlet optimal blade angle and :math:`\\beta_{1\ i}` is the actual inlet blade angle.\n
            Then the enthalpy loss is computed as :math:`\\Delta h_i = \\frac{1}{2} f_{\\mathrm{inc}} W_{\\mathrm{nonopt}\ i}^2`
            with :math:`f_{\\mathrm{inc}}` the incidence parameter given by the class attribute ``_finc``.
            Finally, the total enthalpy loss is given by :math:`\\frac{\\Delta h_{\\mathrm{hub}} + 10\\Delta h_{\\mathrm{mid}} + \\Delta h_{\\mathrm{tip}}}{12}`

            Parameters
            ----------
            inlet_velocity : 3 x 3 x 2 array of float
                             V, W, U velocities for hub, mid and tip
            inlet_blade_angle : 3 x 1 array of float
                                hub, mid and tip blade angles at the inlet
            inlet_optimal_angle : 3 x 1 array of float
                                  hub, mid and tip optimal blade angles at the inlet

            Returns
            -------
            _value : float
                     computed massic enthalpy loss value
        """
        wnonopt = np.zeros((3,))
        h = np.zeros((3,))
        for i in range(0, 3):
            wnonopt[i] = np.linalg.norm(inlet_velocity[1, i]) * \
                np.sin(np.abs(inlet_blade_angle[i] - inlet_optimal_angle[i]))
        h = 0.5*self._finc*wnonopt**2
        self._value = 1/12.*(h[0] + 10*h[1] + h[2])
        return self._value
    
class ShockLoss():
    """Container of the Shock loss for an impeller

        Shock loss is considered when the inlet relative speed is sonic and shock-waves develop at the inlet.
    
        Attributes
        ----------
        _type : string, defaults to ``'internal'``
            Class attribute and not instance attribute. Define the type of loss
        _name : string, defaults to ``'Shock'``
                Class attribute and not instance attribute. Define the name of the loss
        _value : float, defaults to ``0``
                 computed massic enthalpy loss value
    """
    _type = "internal"
    _name = "Shock"

    def __init__(self):
        """Constructor of ShockLoss class

            Sets ``_value`` to default value of ``0``.
        """
        self._value = 0

    def compute_loss(self, kpv, inlet_temperature, inlet_velocity, inlet_speedofsound, dhdT):
        """Computes the loss caused by a shockwave in the impeller section

            The shock loss is based on :cite:t:`2012:Parisi` assumptions ie.:

            1. Normal shock-wave
            2. Compressibility factor, heat capacity and isentropic exponent are constant across the shock-wave.

            The total temperature in the shock :math:`T_{t1\ \\mathrm{sh}}` is computed as a function of 
            the inlet total temperature :math:`T_{t1}`, 
            the inlet Mach number :math:`M_{w1}` and 
            the isentropic expansion coefficient :math:`k`:\n
            :math:`\\frac{T_{t1\ \\mathrm{sh}}}{T_{t1}} = \\frac{(k + 1)M_{w1}^2}{(k-1)M_{w1}^2 + 2} \\left( \\frac{k + 1}{2kM_{w1}^2 - (k - 1)} \\right)^{\\frac{1}{k}}`\n
            Then the enthalpy loss is given by :math:`\\Delta h = \\left( \\frac{\\partial h}{\\partial T} \\right)_{P\ 1} \\left( T_{t1} - T_{t1\ \\mathrm{sh}} \\right)`
            \n
            Similarly as :func:`~losses.IncidenceLoss.compute_loss`, this computation is done at hub, mid and tip line then averaged:
            :math:`\\frac{\\Delta h_{\\mathrm{hub}} + 10\\Delta h_{\\mathrm{mid}} + \\Delta h_{\\mathrm{tip}}}{12}`


            Parameters
            ----------
            kpv : array of floats of length 3
                  Isentropic expansion coefficient (Cp/Cv) values for [hub, mid, tip] line at the inlet of the impeller

            Returns
            -------
            _value : float
                     Massic enthalpy loss from shockwave\n
                     0 if all the inlet relative Mach number are less than 1
        """
        mach = [np.linalg.norm(inlet_velocity[1, i])/inlet_speedofsound for i in range(0, 3)]
        h = np.zeros(3,)
        for i in range(0, 3):
            if mach[i] < 1: continue
            Tt1sh_Tt1 = (kpv + 1)*mach[i]**2/((kpv - 1)*mach[i]**2 + 2)*((kpv + 1)/(2*kpv*mach[i]**2 - (kpv - 1)))**(1/kpv)
            Tt1 = inlet_temperature*(1 + (kpv - 1)/2*mach[i])
            h[i] = dhdT*Tt1*(1 - Tt1sh_Tt1)
        self._value = 1/12.*(h[0] + 10*h[1] + h[2])
        return self._value

class DiffusionLoss():
    """Container of the Diffusion loss for an impeller

        Diffusion loss is a loss occuring between the inlet and the throat due to a slowdown of the fluid. 
        Stall is also considered when the diffusion loss is excessive.
    
        Attributes
        ----------
        _type : string, defaults to ``'internal'``
            Class attribute and not instance attribute. Define the type of loss
        _name : string, defaults to ``'Diffusion'``
                Class attribute and not instance attribute. Define the name of the loss
        _value : float, defaults to ``0``
                 computed massic enthalpy loss value
        _difforstall : string, defaults to ``None``
                       string indicating if diffusion, stall or incidence is prevalent
    """
    _type = "internal"
    _name = "Diffusion"

    def __init__(self):
        """Constructor of ShockLoss class

            Sets ``_value`` to default value of ``0`` and ``_difforstall`` to default value of ``None``.
        """
        self._value = 0
        self._difforstall = None

    def compute_diffusion_loss_component(self, impeller_inlet_velocity_W1mid, throat_velocity, incidenceloss_param):
        """Computes and returns the diffusion loss component for the diffusion loss

            The diffusion loss is given by :cite:t:`2000:Aungier` as: :math:`\\Delta h_{\\mathrm{diff}} = \\mathrm{max}\\left( 0.5 f_{\\mathrm{inc}}\ ;\ 0.4 \\right) \\left( W_{1\ \\mathrm{mid}} - W_{\\mathrm{throat}} \\right)^2`
            where :math:`W_{1\ \\mathrm{mid}}` is the relative velocity of the fluid at the mean line of the inlet of the impeller,
            :math:`W_{\\mathrm{throat}}` the relative velocity at the throat of the impeller.
            The :math:`\\mathrm{max}` is added since some authors consider the incidence factor while some others consider a factor of ``0.4``.

            Parameters
            ----------
            impeller_inlet_velocity_W1mid : float
                                            Norm of impeller's relative velocity at the mean line of the inlet
            throat_velocity : float
                              Relative velocity in the throat section given by the continuity equation at the throat assuming that the flow is perpendicular to the throat area
            incidenceloss_param : float
                                  See ``IncidenceLoss._finc``

            Returns
            -------
            dh_diff : float
                      computed massic enthalpy loss for the diffusion component
        """
        d = impeller_inlet_velocity_W1mid - throat_velocity
        return max(0.5*incidenceloss_param, 0.4)*d**2 if d > 0 else 0
    
    def compute_stall_loss_component(self, impeller_inlet_velocity_W1tip, throat_velocity):
        """Computes the stall loss component of the diffusion loss model

            The stall loss is given by :cite:t:`2000:Aungier` as: :math:`\\Delta h_{\\mathrm{stall}} = 0.5 \\left( W_{1\ \\mathrm{mid}} - 1.75 W_{\\mathrm{throat}} \\right)^2`
            if :math:`W_{1\ \\mathrm{mid}} > 1.75 W_{\\mathrm{throat}}`
            where :math:`W_{1\ \\mathrm{mid}}` is the relative velocity of the fluid at the mean line of the inlet of the impeller,
            :math:`W_{\\mathrm{throat}}` the relative velocity at the throat of the impeller.

            Parameters
            ----------
            impeller_inlet_velocity_W1tip : float
                                            Norm of impeller's relative velocity at the tip line of the inlet
            throat_velocity : float
                              Relative velocity in the throat section given by the continuity equation at the throat assuming that the flow is perpendicular to the throat area

            Returns
            -------
            dh_stall : float
                       computed massic enthalpy loss for the stall component
        """
        d = impeller_inlet_velocity_W1tip - 1.75*throat_velocity
        return 0.5*d**2 if d > 0 else 0

    def compute_loss(self, impeller_inlet_velocity_W1mid, impeller_inlet_velocity_W1tip, throat_velocity, incidenceloss, incidenceloss_param):
        """Computes and returns the diffusion loss of the impeller ; stall is also considered. The value is stored in ``_value``.

            Diffusion, stall and incidence are considered. Only the part above the incidence loss from :func:`~losses.IncidenceLoss` is considered.
            Effectively, this means that the enthalpy loss is computed as 
            :math:`\\mathrm{max}\\left( h_{\\mathrm{diff}}\ ;\ h_{\\mathrm{stall}}\ ;\ h_{\\mathrm{inc}} \\right) - h_{\\mathrm{inc}}`

            Parameters
            ----------
            impeller_inlet_velocity_W1mid : float
                                            Impeller's relative velocity (norm) at the mean line of the inlet
            impeller_inlet_velocity_W1tip : float
                                            Impeller's relative velocity (norm) at the tip line of the inlet
            throat_velocity : float
                              Relative velocity in the throat section given by the continuity equation at the throat assuming that the flow is perpendicular to the throat area
            incidenceloss : float
                            Incidence loss, usuallu computed by :func:`~losses.IncidenceLoss.compute_incidence_loss()`
            incidenceloss_param : float
                                  Incidence loss parameter from :func:`~losses.IncidenceLoss._finc`
        
            Returns
            -------
            _value : float
                     massic enthalpy loss
        """
        diff = self.compute_diffusion_loss_component(impeller_inlet_velocity_W1mid, throat_velocity, incidenceloss_param)
        stall =  self.compute_stall_loss_component(impeller_inlet_velocity_W1tip, throat_velocity)
        if diff > stall:
            self._difforstall = 'Diffusion or incidence prevalent'
        else:
            self._difforstall = 'Stall or incidence prevalent'
        self._value =  max(diff, stall, incidenceloss) - incidenceloss
        return self._value

class ChokingLoss():
    """Container of the Choking loss for an impeller

        Choking loss is a loss occuring when the fluid approaches sonic conditions near the throat.
        The computation is done following the formula of :cite:t:`1995:Aungier`:\n
        :math:`\\Delta h = 0.5 W_{1\ \\mathrm{mid}}^2 \\left(0.05 x + x^7 \\right)`\nwith
        :math:`W_{1\ \\mathrm{mid}}^2` the impeller inlet relative velocity for the mean line and 
        :math:`x = 10 \\left( 1.1 - A_{\\mathrm{throat}}/A^* \\right)`
    
        Attributes
        ----------
        _type : string, defaults to ``'internal'``
            Class attribute and not instance attribute. Define the type of loss
        _name : string, defaults to ``'Choking'``
                Class attribute and not instance attribute. Define the name of the loss
        _value : float, defaults to ``0``
                 computed massic enthalpy loss value
    """
    _type = "internal"
    _name = "Choking"
    
    def __init__(self):
        """Constructor of ShockLoss class

            Sets ``_value`` to default value of ``0``.
        """
        self._value = 0

    def compute_ratioarea(self, mach, kpv):
        """Computes the ratio of sonic area over the area for the throat and returns it

            The ratio of sonic area over the area is given by :cite:t:`1961:Dixon` equation 80:
            :math:`\\frac{A^*}{A_{\\mathrm{throat}}} = \\left( \\left( \\frac{k + 1}{2} \\right)^{\\frac{k + 1}{2(k - 1)}} Ma \\left( 1 + \\frac{k - 1}{2} Ma^2 \\right) \\right)^{- \\frac{k + 1}{2(k - 1)}}`

            Parameters
            ----------
            mach : float
                   Mach number at the mid line of the throat of the impeller
            kpv : float
                  Isentropic expansion coefficient at the throat of the impeller mid line

            Returns
            -------
            a*/a : float
                   A*/Athroat ratio of sonic area over the area according to :cite:t:`1961:Dixon` equation 80.
        """
        exp = (kpv + 1)/(2*(kpv - 1))
        return ((kpv + 1)/2)**exp * mach * (1 + (kpv - 1)/2 * mach**2)**(-exp)
    
    def compute_loss(self, mach, kpv, w1mid):
        """Computes the choking loss, stores it in ``_value`` and returns it

            The computation is done following the formula of :cite:t:`1995:Aungier`:\n
            :math:`\\Delta h = 0.5 W_{1\ \\mathrm{mid}}^2 \\left(0.05 x + x^7 \\right)`\nwith
            :math:`W_{1\ \\mathrm{mid}}^2` the impeller inlet relative velocity for the mean line and 
            :math:`x = 10 \\left( 1.1 - A_{\\mathrm{throat}}/A^* \\right)`

            Parameters
            ----------
            mach : float
                   Mach number at the mid line of the throat of the impeller
            kpv : float
                  Isentropic expansion coefficient at the throat of the impeller mid line
            w1mid : float
                    Norm relative velocity at the mid line of the inlet of the compressor

            Returns
            -------
            dh : float
                 massic enthalpy loss
        """
        x = 10*(1.1 - 1/self.compute_ratioarea(mach, kpv))
        self._value = (1 if x > 0 else 0)*0.5*w1mid**2*(0.05*x + x**7)
        return self._value

class SupercriticalLoss():
    """Container of the Supercritical loss for an impeller

        Supercritical loss is a loss corresponding to additional shocks inside the blade passage due to the local acceleration on the suction side.
        The supercritical loss is computed from :cite:t:`2012:Parisi` and :cite:t:`1995:Aungier`.
        First the maximal relative speed is computed as: :math:`W_\\mathrm{max} = \\left( W_{1\ \\mathrm{mid}} + W_2 + \\Delta W \\right)/2`
        with :math:`W_{1\ \\mathrm{mid}}` the impeller's inlet relative velocity for the mid line,
        :math:`W_2` the impeller's outlet relative velocity and the optimal speed distribution leading to
        :math:`\\Delta W = \\frac{2 \\pi D_2 L_\\mathrm{eul}}{N_{\\mathrm{b\ eff}} L_\\mathrm{b} U_2}`.
        Using usual nomenclature, :math:`D_2` is the impeller's outlet diameter, 
        :math:`L_\\mathrm{eul}` the impeller's Eulerian work,
        :math:`N_{\\mathrm{b\ eff}` the number of effective blades (includes splitters effects),
        :math:`L_\\mathrm{b}` the blade length and :math:`U_2` the impeller's outlet tangential speed.\n
         
        The critical mach number is then computed with
        :math:`M_{\\mathrm{crit}} = M_{\\mathrm{W}\ 1\ \\mathrm{mid}} W^*/W_\\mathrm{max}`
        where :math:`W^*` is the speed of sound for the throat and
        :math:`M_{\\mathrm{W}\ 1\ \\mathrm{mid}}` the impeller's inlet Mach number for the relative speed
        at the mid line.\n
        
        Finally, the enthalpy loss is computed with:
        :math:`\\Delta h = f_\\mathrm{sup} \\frac{W_\\mathrm{max}^2}{2}\\left( M_{\\mathrm{W}\ 1\ \\mathrm{mid}} - M_\\mathrm{crit}\\right)`
        
    
        Attributes
        ----------
        _type : string, defaults to ``'internal'``
            Class attribute and not instance attribute. Define the type of loss
        _name : string, defaults to ``'Supercritical'``
                Class attribute and not instance attribute. Define the name of the loss
        _value : float, defaults to ``0``
                 computed massic enthalpy loss value
        _fsup : float, defaults to ``0.4``
                Experimental coefficient
        _wmax : float, defaults to ``0``
                Maximum speed point considering an optimal velocity distribution
        _Mcr : float, defaults to ``0``
               Critical Mach value at the inlet condition causing sonic conditions for the maximum velocity point
        _mach : float, defaults to ``0``
                Mach number for the inlet relative speed at the mean line
    """
    _type = "internal"
    _name = "Supercritical"
    
    def __init__(self):
        """Constructor of ShockLoss class

            Sets ``_value``, ``_wmax``, ``_Mcr`` and ``_mach`` to default value of ``0`` ; ``_fsup`` to default value of 0.4.
        """
        self._value = 0
        self._fsup = 0.4
        self._wmax = 0
        self._Mcr = 0
        self._mach = 0

    def compute_max_velocity(self, 
                             outlet_diameter, eulerian_work, effective_number_blade, blade_length,
                             inlet_velocities, outlet_velocities):
        """Computes and stores the max velocity ``_wmax`` to be used for the supercritical loss computation

            From :cite:t:`2012:Parisi`: 
            :math:`W_\\mathrm{max} = \\left( W_{1\ \\mathrm{mid}} + W_2 + \\Delta W \\right)/2`
            with :math:`W_{1\ \\mathrm{mid}}` the impeller's inlet relative velocity for the mid line,
            :math:`W_2` the impeller's outlet relative velocity and the optimal speed distribution leading to
            :math:`\\Delta W = \\frac{2 \\pi D_2 L_\\mathrm{eul}}{N_{\\mathrm{b\ eff}} L_\\mathrm{b} U_2}`.
            Using usual nomenclature, :math:`D_2` is the impeller's outlet diameter, 
            :math:`L_\\mathrm{eul}` the impeller's Eulerian work,
            :math:`N_{\\mathrm{b\ eff}` the number of effective blades (includes splitters effects),
            :math:`L_\\mathrm{b}` the blade length and :math:`U_2` the impeller's outlet tangential speed.\n

            Parameters
            ----------
            outlet_diameter : float
                              Impeller's outlet diameter :math:`D_2`
            eulerian_work : float
                            Eulerian work of the impeller :math:`L_\\mathrm{eul}`
            effective_number_blade : float
                                     Number of effective blades according :math:`N_{\\mathrm{b\ eff}`
            blade_length : float
                           Length of the blade :math:`L_\\mathrm{b}`
            inlet_velocities : 3x2x2 array of float
                               velocities V, W, U for hub, mid and tip line with radial and tangential coordinates
            outlet_velocities : 3x3x2 array of floats
                                Velocities V, W, U for hub, mid and tip line.
        """
        dw = 2*np.pi*outlet_diameter*eulerian_work/(effective_number_blade*blade_length*outlet_velocities[2, 2, 1])
        self._wmax = 0.5*(np.linalg.norm(inlet_velocities[1, 1]) + np.linalg.norm(outlet_velocities[1, 1]) + dw)

    def compute_loss(self,
                    outlet_diameter, 
                    eulerian_work, 
                    effective_number_blade, 
                    blade_length,
                    inlet_velocities, 
                    outlet_velocities,
                    mach_number, 
                    speedsound_throat):
        """Computes the supercritical loss, stores it in ``_value`` and returns ``_value``

            From :cite:t:`1995:Aungier` and :cite:t:`2012:Parisi`:
            the critical mach number is computed with
            :math:`M_{\\mathrm{crit}} = M_{\\mathrm{W}\ 1\ \\mathrm{mid}} W^*/W_\\mathrm{max}`
            where :math:`W^*` is the speed of sound for the throat and
            :math:`M_{\\mathrm{W}\ 1\ \\mathrm{mid}}` the impeller's inlet Mach number for the relative speed (parameter also stored in ``_mach``)
            at the mid line. The result is stored in ``_Mcr``\n
            
            Finally, the enthalpy loss is computed with:
            :math:`\\Delta h = f_\\mathrm{sup} \\frac{W_\\mathrm{max}^2}{2}\\left( M_{\\mathrm{W}\ 1\ \\mathrm{mid}} - M_\\mathrm{crit}\\right)`
            and stored in ``_value`` before being returned.

            Parameters
            ----------
            outlet_diameter : float
                              Impeller's outlet diameter :math:`D_2`
            eulerian_work : float
                            Eulerian work of the impeller :math:`L_\\mathrm{eul}`
            effective_number_blade : float
                                     Number of effective blades according :math:`N_{\\mathrm{b\ eff}`
            blade_length : float
                           Length of the blade :math:`L_\\mathrm{b}`
            inlet_velocities : 3x2x2 array of float
                               velocities V, W, U for hub, mid and tip line with radial and tangential coordinates
            outlet_velocities : 3x3x2 array of floats
                                Velocities V, W, U for hub, mid and tip line.
            mach_number : float
                          Mach number at inlet mid line of the impeller :math:`M_{\\mathrm{W}\ 1\ \\mathrm{mid}}`
            speedsound_throat : float
                                Speed of sound at the throat :math:`W^*`

            Returns
            -------
            dh : float
                 massic enthalpy loss
        """
        self.compute_max_velocity(outlet_diameter, eulerian_work, effective_number_blade, blade_length, inlet_velocities, outlet_velocities)
        self._Mcr = mach_number*speedsound_throat/self._wmax
        self._mach = mach_number
        self._value = self._fsup*0.5*self._wmax**2*(mach_number - self._Mcr)**2*(1 if mach_number > self._Mcr else 0)
        return self._value

class BladeloadingLoss():
    """
        Compute the blade loading loss in the impeller section due to the growth of a boundary layer in an adverse velocity gradient

        Equations 2.85, 2.86 and 2.87 of S. Parisi PhD thesis.
    """
    _type = "internal"
    _name = "Blade loading"
    
    def __init__(self):
        self._value = 0
        self._d = 0
        self._equation = None

    def compute_diffusion_Coppage(self,
            impeller_outlet_speed, impeller_inlet_tip_speed,
            eulerian_work, number_efficient_blades,
            impeller_outlet_diameter, impeller_inlet_tip_diameter, impeller_outlet_tangential_speed):
        """
            Compute the diffusion factor for the loss model according to Coppage

            Equation 2.86 of S. Parisi PhD thesis. Best for low specific speed with no splitter.

            Parameters
            ----------
            impeller_outlet_speed : float
                Relative speed (norm ; W) at the outlet of the impeller.
            impeller_inlet_tip_speed : float
                Relative speed (norm ; W) at the tip line of the impeller.
            eulerian_work : float
                Eulerian work of the impeller. Equation 2.12 of S. Parisi PhD thesis.
            number_efficient_blades : float
                Number of efficient blades of the impeller. Equation 2.9 of S. Parisi PhD thesis.
            impeller_outlet_diameter : float
                Diameter of the outlet of the impeller.
            impeller_inlet_tip_diameter : float
                Diameter of the impeller at the inlet for the tip line.
            impeller_outlet_tangential_speed : float
                U2

            Returns
            -------
            float
                Coefficient of diffusion from Coppage model. Equation 2.86 of S. Parisi PhD thesis.
        """
        return 1 - impeller_outlet_speed/impeller_inlet_tip_speed + 0.75*eulerian_work/((
            impeller_inlet_tip_speed/impeller_outlet_speed*(
                number_efficient_blades/np.pi*(1 - impeller_inlet_tip_diameter/impeller_outlet_diameter) +\
                2*impeller_inlet_tip_diameter/impeller_outlet_diameter
                )
            )*impeller_outlet_tangential_speed**2)
    
    def compute_diffusion_Whitfield(self,
            impeller_outlet_speed, impeller_inlet_tip_speed,
            eulerian_work, number_efficient_blades,
            impeller_outlet_diameter, impeller_inlet_tip_diameter,
            impeller_inlet_hub_diameter,
            outlet_blade_height, blade_length,
            impeller_outlet_tangential_speed):
        """
            Compute the diffusion factor for the loss model according to Whitfield.

            Equation 2.87 of S. Parisi PhD thesis. Best for high specific speed with splitters.

            Parameters
            ----------
            impeller_outlet_speed : float
                Relative speed (norm ; W) at the outlet of the impeller.
            impeller_inlet_tip_speed : float
                Relative speed (norm ; W) at the tip line of the impeller.
            eulerian_work : float
                Eulerian work of the impeller. Equation 2.12 of S. Parisi PhD thesis.
            number_efficient_blades : float
                Number of efficient blades of the impeller. Equation 2.9 of S. Parisi PhD thesis.
            impeller_outlet_diameter : float
                Diameter of the outlet of the impeller.
            impeller_inlet_tip_diameter : float
                Diameter of the impeller at the inlet for the tip line.
            impeller_inlet_hub_diameter : float
                Diameter of the impeller at the inlet for the hub line.
            outlet_blade_height : float
                Blade height at the outlet (:math:`b_2` in S. Parisi PhD thesis)
            blade_length : float
                Length of a blade (equation 2.34 of S. Parisi PhD thesis)
            impeller_outlet_tangential_speed : float
                Speed of the impeller (U)

            Returns
            -------
            float
                Coefficient of diffusion from Whitfield model. Equation 2.87 of S. Parisi PhD thesis.
        """
        return 1 - impeller_outlet_speed/impeller_inlet_tip_speed + \
            np.pi*impeller_outlet_diameter*eulerian_work/(2*number_efficient_blades*blade_length*impeller_inlet_tip_speed*impeller_outlet_tangential_speed) + \
            0.1*(impeller_inlet_tip_diameter - impeller_inlet_hub_diameter + 2*outlet_blade_height)/((impeller_outlet_diameter - impeller_inlet_tip_diameter)*(1 + impeller_outlet_speed/impeller_inlet_tip_speed))

    def compute_loss(self, type,
            impeller_outlet_speed, impeller_inlet_tip_speed,
            eulerian_work, number_efficient_blades,
            impeller_outlet_diameter, impeller_inlet_tip_diameter,
            impeller_outlet_tangential_speed,
            impeller_inlet_hub_diameter=0,
            outlet_blade_height=0, blade_length=0):
        """
            Compute the blade loading loss

            Equation 2.85 of S. Parisi PhD thesis.

            impeller_outlet_speed : float
                Tangential relative speed (norm ; W2) at the outlet of the impeller.
            impeller_inlet_tip_speed : float
                Tangential relative speed at the tip line of the impeller.
            eulerian_work : float
                Eulerian work of the impeller. Equation 2.12 of S. Parisi PhD thesis.
            number_efficient_blades : float
                Number of efficient blades of the impeller. Equation 2.9 of S. Parisi PhD thesis.
            impeller_outlet_diameter : float
                Diameter of the outlet of the impeller.
            impeller_inlet_tip_diameter : float
                Diameter of the impeller at the inlet for the tip line.
            impeller_outlet_tangential_speed : float
                Speed of the impeller (U)
            impeller_inlet_hub_diameter : float, optional
                Diameter of the impeller at the inlet for the hub line, by default 0
            outlet_blade_height : float, optional
                Blade height at the outlet (:math:`b_2` in S. Parisi PhD thesis), by default 0
            blade_length : float, optional
                Length of a blade (equation 2.34 of S. Parisi PhD thesis), by default 0

            Returns
            -------
            float
                Blade loading loss according to equation 2.85 of S. Parisi PhD thesis.

            Raises
            ------
            NotImplementedError
                if type is not Coppage or Whitfield, raises the implementation error
        """
        self._equation = type
        if type == "Coppage":
            self._d = self.compute_diffusion_Coppage(
                impeller_outlet_speed, impeller_inlet_tip_speed,
                eulerian_work, number_efficient_blades,
                impeller_outlet_diameter, impeller_inlet_tip_diameter, impeller_outlet_tangential_speed)
        elif type == "Whitfield":
            self._d = self.compute_diffusion_Whitfield(
                impeller_outlet_speed, impeller_inlet_tip_speed,
                eulerian_work, number_efficient_blades,
                impeller_outlet_diameter, impeller_inlet_tip_diameter,
                impeller_inlet_hub_diameter,
                outlet_blade_height, blade_length,
                impeller_outlet_tangential_speed)
        else:
            raise NotImplementedError("Diffusion factor implemented are Coppage and Whitfield")
        self._value = 0.05*self._d**2*impeller_outlet_tangential_speed**2
        return self._value

class SkinfrictionLoss():
    """
        Compute the skin friction loss component

        Equation 2.88 of S. Parisi PhD thesis.\n
        Colebrook-White equation solved with brentq xtol=1e-10 and rtol=1e-15\n

        Attribute
        -------
        _Ksf : float
            Fanning friction factor. Default is 4
    """
    _type = "internal"
    _name = "Skin friction"

    def __init__(self):
        self._value = 0
        self._Ksf = 4 #Fanning friction factor ; could be higher

    def compute_skinfrictionloss_speed(self, inlet_velocities, outlet_velocities):
        """
            Compute the skin friction speed used for the skin friction loss

            Equation 2.89 of S. Parisi PhD thesis

            Parameters
            ----------
            inlet_velocities : array of float
                3x3x2 array of floats
                velocities V, W, U for hub, mid and tip line.
            outlet_velocities : array of float
                3x3x2 array of floats
                velocities V, W, U for hub, mid and tip line.

            Returns
            -------
            float
                Meridional relative speed Wsf described by equation 2.89 of S. Parisi PhD thesis
        """
        return 1/8.*(inlet_velocities[0, 2, 0] + outlet_velocities[0, 1, 0] + \
            inlet_velocities[1, 2, 0] + 2*inlet_velocities[1, 0, 0] + 3*outlet_velocities[0, 1, 0])

    def compute_loss(self, bladelength, hydraulicdiameter,
                                 averageroughness, kinematicvisc, #array [inlet, outlet] 
                                 inlet_velocities, outlet_velocities):
        """
            Compute the skin friction loss

            Equation 2.88 of S. Parisi PhD thesis

            Parameters
            ----------
            bladelength : float
                Blade length defined by equation 2.34 of S. Parisi PhD thesis
            hydraulicdiameter : float
                Hydraulic diameter defined by equation 2.35 of S. Parisi PhD thesis
            averageroughness : float
                Average roughness of the surface after machining and eventual treatments also called Ra. Ra is usually around 3.2 µm for standard machining.
            kinematicvisc : array of floats
                Kinematic viscosities of the inlet and outlet
            outlet_velocities : array of float
                3x3x2 array of floats
                velocities V, W, U for hub, mid and tip line.

            Returns
            -------
            float  
                Loss of enthalpy due to the skin friction component defined by equation 2.88 of S. Parisi PhD thesis
        """
        Wsf = self.compute_skinfrictionloss_speed(inlet_velocities=inlet_velocities, outlet_velocities=outlet_velocities)
        Re = self.compute_Reynolds(Wsf, hydraulicdiameter, np.mean(kinematicvisc))
        relativeroughnesseps = averageroughness/hydraulicdiameter
        fanningfactor = self.compute_fanningfactor(Re, relativeroughnesseps)
        self._value = 0.5*self._Ksf*fanningfactor*bladelength/hydraulicdiameter*Wsf**2
        return self._value
    
    def compute_Reynolds(self, speed, length, kinematicvisc):
        """
            Compute the Reynolds number

            Parameters
            ----------
            speed : float
                Speed to take into account for the Reynolds number
            length : float
                Representative length to take into account for the Reynolds number
            kinematicvisc : float
                Kinematic viscosity of the fluid to compute the Reynolds number

            Returns
            -------
            float
                Reynolds number
        """
        return speed*length/kinematicvisc

    def compute_fanningfactor(self, Reynolds, relativeroughnesseps):
        """
            Compute the fanning factor cf

            Equation 2.91 of S. Parisi PhD thesis

            Parameters
            ----------
            Reynolds : float
                Reynolds number
            relativeroughnesseps : float
                Relative roughness equals to the average roughness of the surface divided by the hydraulic diameter

            Returns
            -------
            float
                Fanning factor cf according to equation 2.91 of S. Parisi PhD thesis
        """
        if Reynolds >= 2300.:
            return self.solve_Colebrook_White(Reynolds= Reynolds, relativeroughnesseps= relativeroughnesseps)/4
        else:
            return max(16./Reynolds, self.solve_Colebrook_White(Reynolds= 2300., relativeroughnesseps= relativeroughnesseps)/4)

    def solve_Colebrook_White(self, Reynolds, relativeroughnesseps):
        """
            Find the root value of the Colebrook White equation to find the friction factor :math:`\lambda`

            Equation 2.92 of S. Parisi PhD thesis = 0

            Parameters
            ----------
            Reynolds : float
                Reynolds number of the flow
            relativeroughnesseps : float
                Average roughness / hydraulic diameter

            Returns
            -------
            float
                Friction factor given by the Colebrook-White equation, equation 2.92 of S. Parisi PhD thesis
        """
        return optimize.brentq(self.Colebrook_White_equation, args= (relativeroughnesseps, Reynolds), a=1e-5, b=100, xtol=1e-10, rtol=1e-15)

    def Colebrook_White_equation(self, frictionfactorlambda, relativeroughnesseps, Reynolds):
        """
            Colebrook White function

            Equation 2.92 of S. Parisi PhD thesis

            Parameters
            ----------
            frictionfactorlambda : float
                Friction factor of the Colebrook-White equation, :math:`\lambda`
            relativeroughnesseps : float
                Relative roughness :math:`\epsilon` computed as the ratio of the average roughness of the surface and the hydraulic diameter
            Reynolds : float
                Reynolds number of the flow

            Returns
            -------
            float
                Rest of the Colebrook White function re-written to be equal to 0 from equation 2.92 of S. Parisi PhD thesis
        """
        return 1./np.sqrt(frictionfactorlambda) + 2*np.log10(relativeroughnesseps/3.7 + 2.51/(Reynolds*np.sqrt(frictionfactorlambda)))

class ClearanceLoss():
    """
        Compute the clearance loss due to the clearance in front of the impeller

        Equation 2.93 of S. Parisi PhD thesis

        Attributes
        ----------
        _clearance : float
                clearance in front of the impeller, main parameter of the model\n
                usually between 0.1 mm and 1 mm
    """
    _type = "internal"
    _name = "Clearance"

    def __init__(self):
        self._value = 0
        self._clearance = 2e-4 #metres

    def compute_loss(self,
        outlet_velocities,
        inlet_mid_absolute_speed_radial,
        outlet_diameter, inlet_tip_diameter, inlet_hub_diameter,
        outlet_density, inlet_density,
        effective_number_blade, outlet_blade_height):
        """
            Compute the clearance loss of the impeller

            Equation 2.93 of S. Parisi PhD thesis

            Parameters
            ----------
            outlet_velocities : array of float
                3x3x2 array of floats
                velocities V, W, U for hub, mid and tip line.
            inlet_mid_absolute_speed_radial : float
                Radial component of the absolute speed at the inlet of the impeller for the mid line Vm1mid
            outlet_diameter : float
                Diameter of the impeller at the outlet
            inlet_tip_diameter : float
                Diameter of the impeller at the inlet for the tip line
            inlet_hub_diameter : float
                Diameter of the impeller at the inlet for the hub line
            outlet_density : float
                Density of the fluid at the outlet of the impeller
            inlet_density : float
                Density of the fluid at the inlet of the impeller
            effective_number_blade : float
                Effective number of blades for the impeller for instance equation 2.9 of S. Parisi PhD thesis
            outlet_blade_height : float
                Blade height at the outlet of the impeller

            Returns
            -------
            float
                clearance loss component according to 2.93 of S. Parisi PhD thesis
        """
        self._value = 0.6* self._clearance/outlet_blade_height * outlet_velocities[0, 1, 1] *\
        np.sqrt(
            4*np.pi/(outlet_blade_height*effective_number_blade)*\
            ((inlet_tip_diameter**2 - inlet_hub_diameter**2)/(2*(outlet_diameter - inlet_tip_diameter)*(1 + outlet_density/inlet_density)))*\
            outlet_velocities[0, 1, 1]*inlet_mid_absolute_speed_radial)
        return self._value

class RecirculationLoss():
    """
        Compute the recirculation loss that models the backflow from the vaneless diffuser to the impeller

        Equation 2.94 of S. Parisi PhD thesis
    """
    _type = "parasitic"
    _name = "Recirculation"

    def __init__(self):
        self._value = 0

    def compute_loss(self, 
        type, tanflow_angle, diffusion_factor, impeller_linearspeed):
        """
            Compute the recirculation loss (back flow from the diffuser to the impeller)

            Equation 2.94 of S. Parisi PhD thesis

            Parameters
            ----------
            type : string
                Equation applied. Should be Coppage or Oh
            tanflow_angle : float
                tan of exit absolute flow angle
            diffusion_factor : float
                Diffusion factor, see BladeloadingLoss class
            impeller_linearspeed : float
                Speed of the impeller (U2)

            Returns
            -------
            float
                Recirculation loss according to Equation 2.94 of S. Parisi PhD thesis for Coppage type or equation 9 of https://doi.org/10.1016/j.cja.2017.08.002 for Oh

            Raises
            ------
            NotImplementedError
                Raised if type is not Coppage or Oh
        """
        h = 0
        if type == "Coppage":
            h = 0.02*np.sqrt(np.abs(tanflow_angle))*diffusion_factor**2*impeller_linearspeed**2
        elif type == "Oh":
            h = 8e-5*np.sinh(3.5*np.arctan(tanflow_angle)**3)*diffusion_factor**2*impeller_linearspeed**2
        else:
            raise NotImplementedError("Recirculation Loss implemented are Coppage and Oh")
        self._value = h
        return self._value

class LeakageLoss():
    """
        Compute the leakage loss for open or closed impellers

        For open impeller equation 2.98 of S. Parisi PhD thesis. Leakage from pressure side to suction side of the blade.\n
        For closed impeller equation 2.100 of S. Parisi PhD thesis. Leakage from high pressure to low pressure

        Attributes
        ----------
        _cd : float
            Experimental correction factor used for shrouded impellers, default is 0.9
    """
    _type = "parasitic"
    _name = "Leakage"

    def __init__(self):
        self._value = 0
        self._cd = 0.9 #experimental correction factor
        self._equation = None
        self._clearancemassflow = 0

    def compute_DP_open(self, massflow,
                        inlet_velocities, outlet_velocities, 
                        inlet_diameters, outlet_diameter,
                        outlet_blade_height, blade_length, 
                        numbereffectiveblade):
        """
            Compute the pressure difference between suction and pressure side

            Equation 2.95 of S. Parisi PhD thesis

            Parameters
            ----------
            massflow : float
                massflow of the compressor
            inlet_velocities : array of float
                3x3x2 array of floats
                velocities V, W, U for hub, mid and tip line.
            outlet_velocities : array of float
                3x3x2 array of floats
                velocities V, W, U for hub, mid and tip line.
            inlet_diameters : array of float
                Array of [hub mid tip] line diameters
            outlet_diameter : float
                Outlet diameter of the impeller
            outlet_blade_height : float
                Height of the blade at the outlet of the impeller
            blade_length : float
                Blade length defined by equation 2.34 of S. Parisi PhD thesis
            numbereffectiveblade : float
                Number of effective blade defined by equation 2.9 of S. Parisi PhD thesis

            Returns
            -------
            float
                pressure difference between the pressure side and suction side according to equation 2.95 of S. Parisi PhD thesis
        """
        return 4*massflow*(outlet_diameter*outlet_velocities[0, 1, 1] - inlet_diameters[1]*inlet_velocities[0, 1, 1])/\
            (numbereffectiveblade*blade_length*(inlet_diameters[1] + outlet_diameter)*(inlet_diameters[2] - inlet_diameters[0] + outlet_blade_height))

    def compute_loss(self,
        type="open",
        array_input=[]):
        """
            Wrapper to compute the Leakage loss

            type : string
                open or closed depending on impeller's geometry
            array_input_open : array
                array with the inputs for LeakageLoss.compute_leakageloss_open
            array_input_closed : array
                array with the inputs for LeakageLoss.compute_leakageloss_closed

            Returns
            -------
            float
                Leakage loss depending on the type used.

            Raises
            ------
            NotImplementedError
                if type is not open or closed, raises the implementation error
        """
        self._equation = type
        if "open" == type:
            self._value = self.compute_leakageloss_open(*array_input)
            return self._value
        elif "closed" == type:
            self._value = self.compute_leakageloss_closed(*array_input)
            return self._value
        else:
            raise NotImplementedError("Impeller can only be open or closed")

    def compute_leakageloss_open(self, massflow,
                        inlet_velocities, outlet_velocities, 
                        inlet_diameters, outlet_diameter,
                        outlet_blade_height, blade_length, 
                        numbereffectiveblade,
                        density_outlet, bladerunningclearance):
        """
            Computes the leakage loss of an open impeller

            Equation 2.98 of S. Parisi PhD thesis

            Parameters
            ----------
            massflow : float
                massflow of the compressor
            inlet_velocities : array of float
                3x3x2 array of floats
                velocities V, W, U for hub, mid and tip line.
            outlet_velocities : array of float
                3x3x2 array of floats
                velocities V, W, U for hub, mid and tip line.
            inlet_diameters : array of float
                Array of [hub mid tip] line diameters
            outlet_diameter : float
                Outlet diameter of the impeller
            outlet_blade_height : float
                Height of the blade at the outlet of the impeller
            blade_length : float
                Blade length defined by equation 2.34 of S. Parisi PhD thesis
            numbereffectiveblade : float
                Number of effective blade defined by equation 2.9 of S. Parisi PhD thesis
            density_outlet : float
                density of the fluid at the outlet of the impeller
            bladerunningclearance : float
                blade running clearance of the impeller

            Returns
            -------
            float
                enthalpy lost from the leakage between pressure and suction side of the impeller blade according to equation 2.98 of S. Parisi PhD thesis
        """
        dP = self.compute_DP_open(massflow,
                        inlet_velocities, outlet_velocities, 
                        inlet_diameters, outlet_diameter,
                        outlet_blade_height, blade_length, 
                        numbereffectiveblade)
        u = 0.816*np.sqrt(2*dP/density_outlet)
        self._clearancemassflow = density_outlet*numbereffectiveblade*bladerunningclearance*blade_length*u
        return self._clearancemassflow*u*outlet_velocities[2, 1, 1]/(2*massflow)
    
    def compute_leakageloss_closed(self, massflow, inlet_diameters, density_outlet, bladerunningclearance, numberofteethlaby, deltah_isentropic, Leulerian):
        """
            Computes the leakage loss for a closed impeller

            Equation 2.100 of S. Parisi PhD thesis

            Parameters
            ----------
            massflow : float
                massflow of the compressor
            inlet_diameters : array of float
                Array of [hub mid tip] line diameters
            density_outlet : float
                Density of the fluid at the outlet of the impeller
            bladerunningclearance : float
                Blade running clearance of the impeller
            numberofteethlaby : integer
                Number of teeth in the labyrinth seal in front of the closed impeller
            deltah_isentropic : float
                Isentropic enthalpy of the transformation done by the impeller
            Leulerian : float
                Eulerian work done by the enthalpy, equation 2.12 of S. Parisi PhD thesis

            Returns
            -------
            float
                enthalpy lost from the high pressure side to the low pressure side of the impeller according to equation 2.100 of S. Parisi PhD thesis
        """
        g = density_outlet*bladerunningclearance*np.pi*(inlet_diameters[2] + bladerunningclearance)*self._cd*np.sqrt(2*deltah_isentropic/numberofteethlaby)
        return g*Leulerian/massflow

class DiscfrictionLoss():
    """
        Computes the disc friction loss that models the friction at the back-face of the impeller

        Equations 2.101 to 2.103 of S. Parisi PhD thesis

        Parameters
        ----------
        Loss : _type_
            _description_
    """
    _type = "parasitic"
    _name = "Disc friction"

    def __init__(self):
        self._value = 0
        self._Re = 0
        self._frictionfactor = 0
    

    def compute_friction_factor(self):
        """
            Computes the friction factor

            Equation 2.102 of S. Parisi PhD thesis. Laminar to turbulent boundary layer transition is done at Re = 3e5

            Parameters
            ----------
            Reynolds : float
                Reynolds number of the flow at the outlet of the impeller

            Returns
            -------
            float
                Friction factor according to equation 2.102 of S. Parisi PhD thesis to be used to compute the disc friction loss
        """
        if self._Re < 3e5:
            self._frictionfactor = 2.67/self._Re**0.5
        else:
            self._frictionfactor = 0.0622/self._Re**0.2
        return self._frictionfactor
        
    def compute_loss(self, outlet_velocity, outlet_diameter, outlet_density, massflow, outlet_kinematic_viscosity):
        """
            Computes the disc friction loss of the impeller (back of the impeller)

            Equation 2.103 of S. Parisi PhD thesis

            Parameters
            ----------
            outlet_velocities : array of float
                3x3x2 array of floats
                velocities V, W, U for hub, mid and tip line.
            outlet_diameter : float
                Outlet diameter of the impeller
            outlet_density : float
                Density of the fluid at the outlet of the impeller
            massflow : float
                Massflow of the impeller
            outlet_kinematic_viscosity : float
                Kinematic viscosity at the outlet of the impeller

            Returns
            -------
            _type_
                _description_
        """
        self._Re = outlet_velocity[2, 1, 1]*outlet_diameter/2/outlet_kinematic_viscosity
        self._value = self.compute_friction_factor()*outlet_density*(outlet_diameter/2)**2*outlet_velocity[2, 1, 1]**3/(4*massflow)
        return self._value

class MixingLoss():
    """
        Computes the mixing loss when the flow in the impeller separates

        Equations 2.104 to 2.108 of S. Parisi PhD thesis
    """
    _type = "outside"
    _name = "Mixing"

    def __init__(self):
        self._value = 0

    def compute_separation_velocity(self, max_speed, outlet_relative_velocity):
        """
            Computes the separation velocity

            Equation 2.105 of S. Parisi PhD thesis. Separation velocity

            Parameters
            ----------
            max_speed : float
                Maximum speed obtained from Supercriticalloss.compute_max_velocity
            outlet_relative_velocity : float
                Outlet relative velocity (norm, W2)

            Returns
            -------
            float
                Separation velocity according to equation 2.105 of S. Parisi PhD thesis to be used to compute the mixing loss
        """
        Deq = max_speed/outlet_relative_velocity
        return outlet_relative_velocity*np.max([2, Deq])/2

    def compute_loss(self, max_speed, inlet_velocities, outlet_velocities, impeller_outlet_area, impeller_outlet_diameter, outlet_blade_height):
        """
            Computes the mixing loss

            Equation 2.108 of S. Parisi PhD thesis. 

            Parameters
            ----------
            max_speed : float
                Maximum speed obtained from Supercriticalloss.compute_max_velocity
            outlet_velocities : array of float
                3x3x2 array of floats
                velocities V, W, U for hub, mid and tip line of the impeller
            impeller_outlet_area : float
                impeller outlet area A2 eq. 2.43
            impeller_outlet_diameter : float
                impeller outlet diameter D2
            outlet_blade_height : float
                impeller outlet blade height b2

            Returns
            -------
            float
                Mixing loss 2.108 of S. Parisi PhD thesis
        """
        wsep = self.compute_separation_velocity(max_speed, np.linalg.norm(outlet_velocities[1, 1]))
        wwake = np.sqrt(wsep**2 - outlet_velocities[1, 1, 1]**2)
        wmmix = outlet_velocities[0, 1, 0]*impeller_outlet_area/(np.pi*impeller_outlet_diameter*outlet_blade_height)
        self._value = 0.5*(wwake - wmmix)**2
        """wout = np.sqrt(
            (outlet_velocities[0, 1, 0]*impeller_outlet_area/(np.pi*impeller_outlet_diameter*outlet_blade_height))**2 \
            + (inlet_velocities[1, 1, 1])**2)
        self._value = 0.5*(wsep - wout)**2"""
        return self._value

class VanelessdiffuserLoss():
    _type = "outside"
    _name = "Vaneless diffuser"

    def __init__(self):
        self._value = 0
        self._kcfvld = 0.01
        self._cf = 0
        self._Re = 0
        self._equation = None
    
    def compute_friction_factor(self):
        self._cf = self._kcfvld*(180000/self._Re)**0.2

    def compute_velocitydifftang(self, args):
        """
            args is:
                0 impeller_outlet_velocity V2 [V2mid meridional ; V2mid tangential]
                1 diffuser_outlet_velocity 2 coordinates
                2 blade_height
                3 diffuser_outlet_height
                4 kinematicvisc: average kinematic viscosity used to compute the Reynolds number
                5 impeller_outlet_diameter
                6 diffuser_outlet_diameter
                7 massflow
        """
        return args[0][-1]/\
            (args[6]/args[5] \
             + np.pi*self._cf*args[0][-1]*args[6]\
                *(args[6] - args[5])/(2*args[7]))
    
    def compute_loss_Coppage(self,
                             args_minimal,
                             impeller_outlet_flow_angle):
        """
            args_minimal must have, in every case:
                0 impeller_outlet_velocity V2 [V2mid meridional ; V2mid tangential]
                1 diffuser_outlet_velocity 2 coordinates
                2 blade_height
                3 diffuser_outlet_height
                4 kinematicvisc: average kinematic viscosity used to compute the Reynolds number
                5 impeller_outlet_diameter
                6 diffuser_outlet_diameter
                7 massflow
        """
        self._value = np.min([
            2/3*(args_minimal[5]/args_minimal[6])**(3/2.)*(self._cf*args_minimal[5]/(8*args_minimal[2]))*np.linalg.norm(args_minimal[0])**2/(np.cos(impeller_outlet_flow_angle)**2),
            self._cf*args_minimal[5]\
            *(1 - (args_minimal[5]/args_minimal[6])**1.5)\
                *np.linalg.norm(args_minimal[0])**2/(3*args_minimal[2]*np.cos(impeller_outlet_flow_angle))])

    def compute_loss_Stanitz(self, args_minimal,
                             kpv, mach, cp, 
                             temperature, pressure, densities, idx_impeller_outlet, idx_diffuser_outlet):
        """
            args_minimal must have, in every case:
                0 impeller_outlet_velocity V2 [V2mid meridional ; V2mid tangential]
                1 diffuser_outlet_velocity 2 coordinates
                2 blade_height
                3 diffuser_outlet_height
                4 kinematicvisc: average kinematic viscosity used to compute the Reynolds number
                5 impeller_outlet_diameter
                6 diffuser_outlet_diameter
                7 massflow
            kpv: Isentropic expansion coefficient
            mach: Mach number at impeller's outlet
            cp: caloric capacity at constant pressure at impeller's outlet
            temperature: temperature array _T from Compressor class in compressor.py
            pressure: pressure array from _P Compressor class in compressor.py
            densities: densities array _densities from Compressor class in compressor.py
            idx_impeller_outlet: index to find impeller outlet properties in temperture, pressure or densities
            idx_diffuser_outlet: index to find diffuser's outlet properties in temperture, pressure or densities
        """
        Tt2 = temperature[idx_impeller_outlet]*(1 + (kpv - 1)/2*mach**2)
        pt2 = pressure[idx_impeller_outlet] + 1/2*densities[idx_impeller_outlet]*np.linalg.norm(args_minimal[0])**2
        pt3 = pressure[idx_diffuser_outlet] + 1/2*densities[idx_diffuser_outlet]*np.linalg.norm(args_minimal[1])**2
        exp = (kpv - 1)/kpv
        self._value = cp*Tt2*((pressure[idx_diffuser_outlet]/pt3)**exp - (pressure[idx_diffuser_outlet]/pt2)**exp)

    def compute_loss(self, type= "Stanitz", args_minimal= [0]*8, args= []):
        """
            args_minimal must have, in every case:
                0 impeller_outlet_velocity V2 [V2mid meridional ; V2mid tangential]
                1 diffuser_outlet_velocity 2 coordinates
                2 blade_height
                3 diffuser_outlet_height
                4 kinematicvisc: average kinematic viscosity used to compute the Reynolds number
                5 impeller_outlet_diameter
                6 diffuser_outlet_diameter
                7 massflow
            args for Coppage are
                impeller_outlet_flow_angle
            args for Stanitz are
                kpv: Isentropic expansion coefficient
                mach: Mach number at impeller's outlet
                cp: caloric capacity at constant pressure at impeller's outlet
                temperature: temperature array _T from Compressor class in compressor.py
                pressure: pressure array from _P Compressor class in compressor.py
                densities: densities array _densities from Compressor class in compressor.py
                idx_impeller_outlet: index to find impeller outlet properties in temperture, pressure or densities
                idx_diffuser_outlet: index to find diffuser's outlet properties in temperture, pressure or densities
        """
        self._equation = type
        if len(args_minimal) < 8:
            raise RuntimeError("Vaneless diffuser should have at least 8 arguments for Reynolds and velocity computations")
        
        #iterate to compute diffuser speed and friction coefficient
        old_cf = 1e20
        v3u = 0
        for i in range(0, 200):
            v3u = self.compute_velocitydifftang(args_minimal)
            args_minimal[1][1] = v3u
            
            self._Re = SkinfrictionLoss().compute_Reynolds(
                (np.linalg.norm(args_minimal[0]) + np.linalg.norm(args_minimal[1]))/2,
                args_minimal[2] + args_minimal[3], 
                args_minimal[4])
            self.compute_friction_factor()

            if (np.abs((self._cf - old_cf))/self._cf < 1e-10):
                break

            old_cf = self._cf

        if "Stanitz" == self._equation:
            self.compute_loss_Stanitz(args_minimal, *args)
        elif "Coppage" == self._equation:
            self.compute_loss_Coppage(args_minimal, *args)
        else:
            raise NotImplementedError("Vaneless diffuser Loss is either Coppage or Stanitz")
        return v3u

class VaneddiffuserLoss():
    _type = "outside"
    _name = "Vaned diffuser"

    def __init__(self):
        self._value = 0
    pass

class VoluteLoss():
    """
        Computes the mixing loss when the flow in the impeller separates

        Equations 2.143 of S. Parisi PhD thesis
    """
    _type = "outside"
    _name = "Volute"

    def __init__(self):
        self._value = 0
    
    def compute_loss(self, diffuser_meridional_velocity):
        """
        Computes the separation velocity

        Equation 2.105 of S. Parisi PhD thesis. Separation velocity

        Parameters
        ----------
        diffuser_meridional_velocity : float
            Diffuser's outlet meridional velocity V4m

        Returns
        -------
        float
            Volute loss according to equation 2.143 of S. Parisi PhD thesis
        """
        self._value = 0.5*diffuser_meridional_velocity**2
        return self._value


if __name__ == "__main__()":
    sf = SkinfrictionLoss()
    hydraulicdiameter = 15e-3 #m
    Reynolds = 5000
    relativeroughness = 3.2e-6/hydraulicdiameter

    sensibility_Reynolds = np.logspace(start=2, stop=6, num=100)
    relativeroughness = np.array([1e-2, 3e-3, 4e-4, 2e-4])
    colors = ['purple', 'gold', 'red', 'royalblue']

    cf = np.zeros((relativeroughness.size, sensibility_Reynolds.size))
    Colebrook_White = np.zeros((relativeroughness.size, sensibility_Reynolds.size))
    for i, Re in enumerate(sensibility_Reynolds):
        for j, eps in enumerate(relativeroughness):
            cf[j, i] = sf.compute_fanningfactor(Re, eps)
            Colebrook_White[j, i] = sf.solve_Colebrook_White(Reynolds= Re, relativeroughnesseps= eps)/4

    ref_0_Re = [298.9676907, 1088.581326, 1201.343113, 2330.342407, 2997.873599, 3963.670122, 4988.599018, 6928.925215, 8962.673468, 19936.24131, 39745.37177, 80549.7523, 198817.3227, 600944.9924, 1000000]
    ref_0_cf = [0.053393066, 0.014669802, 0.013770311, 0.013697891, 0.012960097, 0.012294414, 0.011848886, 0.011269953, 0.010890224, 0.010195565, 0.009852036, 0.009671879, 0.009570415, 0.009520083, 0.009520083]

    ref_1_Re = [300.6092763, 1289.978998, 2304.960545, 2997.873599, 5015.990649, 8962.673468, 19936.24131, 29896.76907, 69099.80365, 198817.3227, 597663.3149, 1005490.846]
    ref_1_cf = [0.05325248, 0.012457558, 0.012424757, 0.011540571, 0.010168719, 0.008983597, 0.007894856, 0.007489344, 0.006993156, 0.00672199, 0.006616491, 0.006581694]

    ref_2_Re = [298.9676907, 1297.062075, 1408.097023, 2317.61673, 2997.873599, 5015.990649, 20045.70815, 39745.37177, 89872.46115, 298150.2626, 597663.3149, 994539.1384]
    ref_2_cf = [0.05325248, 0.012229755, 0.01191153, 0.011880167, 0.010976702, 0.009495016, 0.006704291, 0.005814625, 0.00506968, 0.004420175, 0.004215305, 0.004127327]

    ref_3_Re = [298.9676907, 1311.345117, 1423.602765, 2304.960545, 3985.434026, 6009.449924, 9111.123598, 15161.3271, 39963.6075, 99726.58314, 298150.2626, 700522.5446, 1000000]
    ref_3_cf = [0.053112264, 0.012229755, 0.01191153, 0.011817687, 0.010062043, 0.008936351, 0.007999619, 0.007011617, 0.005663325, 0.004758828, 0.004084029, 0.00376351, 0.003684961]

    fig, ax = plt.subplots()
    
    for j, eps in enumerate(relativeroughness):
        ax.plot(sensibility_Reynolds, cf[j], lw='1', color= colors[j], label="{:.0e}".format(relativeroughness[j]))
        ax.plot(sensibility_Reynolds, Colebrook_White[j], color= colors[j], lw='1', linestyle='--')
    
    ax.plot(ref_0_Re, ref_0_cf, linewidth=0, marker='D', color= colors[0], markersize=5, fillstyle="none")
    ax.plot(ref_1_Re, ref_1_cf, linewidth=0, marker='D', color= colors[1], markersize=5, fillstyle="none")
    ax.plot(ref_2_Re, ref_2_cf, linewidth=0, marker='D', color= colors[2], markersize=5, fillstyle="none")
    ax.plot(ref_3_Re, ref_3_cf, linewidth=0, marker='D', color= colors[3], markersize=5, fillstyle="none")

    ax.legend()
    ax.set_xlabel("Re") ; ax.set_ylabel(r'$c_f$')
    ax.set_xscale('log') ; ax.set_yscale('log')
    ax.set(xlim=(100, 10e6), ylim=(0.001, 0.1))
    ax.grid(axis='both') ; ax.grid(axis='both', which='minor', linestyle='--')
    yticks = [0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]
    ax.set_yticks(yticks)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.axvline(x=2300, ymin=0, ymax=1, lw=1, color='k')

    plt.show()