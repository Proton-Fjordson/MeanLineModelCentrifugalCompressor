import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import time
from matplotlib import cm
from scipy.optimize import brentq

class Inlet():
    """Class defining and holding all geometrical data for the inlet of a centrifugal machine
        Currently not in used and to be defined better. For now, copy the impeller inlet.

        Attributes
        ----------
        _hub_diameter : float, defaults to ``0``
            Hub (inner) diameter :math:`D_{\\mathrm{hub}}` of the centrifugal machine's impeller
        _tip_diameter : float, defaults to ``0``
            Tip (inner) diameter :math:`D_{\\mathrm{tip}}` of the centrifugal machine's impeller
        _area : float, defaults to ``0``
            Inlet area for the fluid flow: :math:`\\frac{\\pi}{4} \\left(D_{\\mathrm{tip}}^2 - D_{\\mathrm{hub}}^2\\right)`

        Note
        ----
        TODO Define correctly the class as an inlet and not a derived impeller (in case there's an IGV) and use it in the code.
    """

    def __init__(self):
        """Constructor of Inlet class

            Sets ``_hub_diameter`` :math:`D_{\\mathrm{hub}}`, ``_tip_diameter`` :math:`D_{\\mathrm{tip}}` and _area to ``0``
        """
        self._hub_diameter = 0 #D_0,hub
        self._tip_diameter = 0 #D_0,tip

        self._area = 0 #A_0

    def compute_area(self):
        """Computes the inlet area for the fluid flow as :math:`\\frac{\\pi}{4} \\left(D_{\\mathrm{tip}}^2 - D_{\\mathrm{hub}}^2\\right)` and sets _area

            _tip_diameter :math:`D_{\\mathrm{tip}}` and _hub_diameter :math:`D_{\\mathrm{hub}}` should be set by the user before calling compute_area()
        """
        self._area = np.pi/4*(self._tip_diameter**2 - self._hub_diameter**2)

class IGV():
    """Class defining and holding all geometrical data for the Inlet Guide Vane (IGV) of a centrifugal machine
        Currently not in used and to be defined better. For now, copy the impeller inlet.

        Attributes
        ----------
        _hub_diameter : float, defaults to ``0``
            Hub (inner) diameter :math:`D_{\\mathrm{hub}}` of the centrifugal machine's impeller
        _tip_diameter : float, defaults to ``0``
            Tip (inner) diameter :math:`D_{\\mathrm{tip}}` of the centrifugal machine's impeller
        _area : float, defaults to ``0``
            Inlet area for the fluid flow: :math:`\\frac{\\pi}{4} \\left(D_{\\mathrm{tip}}^2 - D_{\\mathrm{hub}}^2\\right)`
        _position : float, defaults to ``0``
            IGV angle in radians
        _mid_diameter : float, defaults to ``0``
            Mid line diameter :math:`D_{\\mathrm{mid}}`

        Note
        ----
        TODO Define correctly the class as an inlet and not a derived impeller (in case there's an IGV) and use it in the code.
    """

    def __init__(self):
        """Constructor of Inlet class

            Sets ``_hub_diameter``, ``_tip_diameter``, ``_area``, ``_position`` and ``_mid_diameter`` to ``0``
        """
        self._hub_diameter = 0 #D_0,hub
        self._tip_diameter = 0 #D_0,tip
        self._position = 0 #alpha_IGV
        self._mid_diameter = 0 #D_0,mid
        self._area = 0 #A_0

    def compute_mid_diameter(self):
        """Computes the mid line diameter as :math:`\\sqrt{\\frac{D_{\\mathrm{hub}}^2 + D_{\\mathrm{tip}}^2}{2}}` and sets _mid_diameter

            _tip_diameter :math:`D_{\\mathrm{tip}}` and _hub_diameter :math:`D_{\\mathrm{hub}}` should be set by the user before calling compute_mid_diameter()
        """
        self._mid_diameter = np.sqrt((self._hub_diameter**2 + self._tip_diameter**2)/2)
    
    def compute_area(self):
        """Computes the inlet area for the fluid flow as :math:`\\frac{\\pi}{4} \\left(D_{\\mathrm{tip}}^2 - D_{\\mathrm{hub}}^2\\right)` and sets _area

            _tip_diameter :math:`D_{\\mathrm{tip}}` and _hub_diameter :math:`D_{\\mathrm{hub}}` should be set by the user before calling compute_area()
        """
        self._area = np.pi/4*(self._tip_diameter**2 - self._hub_diameter**2)

class Impeller():
    """Class holding all the data to define an impeller geometry

        Attributes
        ----------
        _inlet_diameters : 3x1 array of floats, defaults to ``[0, 0, 0]``
                           inlet diameters of hub :math:`D_{\\mathrm{hub}}`, mid :math:`D_{\\mathrm{mid}}` and tip :math:`D_{\\mathrm{tip}}` line in this order.
        _inlet_blade_thickness : 3 x 1 array of floats, defaults to ``[0, 0, 0]``
                                 inlet blade thickness of hub, mid and tip line in this order.
        _inlet_blade_angle : 3 x 1 array of floats, defaults to ``[0, 0, 0]``
                             inlet blade angle of hub, mid and tip line in this order in radians.
        _outlet_diameter : float, defaults to ``0``
                           outlet diameter of the impeller, usually written :math:`D_2`.
        _outlet_blade_height : float, defaults to ``0``
                               outlet blade height defining the impeller, usually written :math:`b_2`.
        _outlet_blade_thickness : float, defaults to 0
                                  outlet blade thickness.
        _outlet_average_blade_thickness : float, defaults to ``0``
                                          average thickness of the blades at the outlet of the impeller taking into account a taper ratio of the blade (thickness at hub and tip are different)
        _outlet_blade_angle : float, defaults to ``0``
                              outlet blade angle of the impeller in radians, usually written :math:`\\beta_2`.
        _axial_extension : float, defaults to ``0``
                           axial extension (height) of the impeller.
        _blade_running_clearance : float, defaults to ``0``
                                   blade running clearance used for clearance loss, usually written :math:`\epsilon`.
        _number_blade_full : int, defaults to ``0``
                             number of full blades. If there are splitters, full number of blades / 2.
        _splitter_blade_length_fraction : float, defaults to ``0``
                                          splitter blade length fraction ie. length of splitter vs a full blade.\n
                                          Equals 0 or 1 if no splitter are used.
        _blade_length : float, defaults to ``0``
                        length of a blade according to :cite:t:`1970:Jansen`. W. Jansen, Inst. Mech. Eng. Internal Aerodynamics, 1970.
        _blade_length_meanline : float, defaults to 0
                                 length of a blade at the meanline computed by numerically integrating the geometrical definition of the blade.
        _effective_number_blade : float, defaults to ``0``
                                  effective number of blades. Differs from ``_full_number_blade_w_splitters`` insofar as it's not necessarily an integer.
                                  Used for the slip factor.
        _full_number_blade_w_splitters : int, defaults to ``0``
                                         total number of blades including splitters.
        _hydraulic_diameter : float, defaults to ``0``
                              hydraulic diameter of the impeller.
        _inlet_area : float, defaults to ``0``
                      inlet area.
        _throat_pitch_blade : 3x1 array of floats, defaults to ``[0, 0, 0]``
                              pitch of the impeller at the throat for the hub, mid, tip lines in that order.
        _throat_width : 3x1 array of floats, defaults to ``[0, 0, 0]``
                        opening width of the throat for the hub, mid and tip lines in that order.
        _outlet_area : float, defaults to ``0``
                       outlet area
        _inlet_optimal_angle : 3x1 array of floats, defaults to ``[0, 0, 0]``
                               optimal blade angle at inlet for the hub, mid and tip line.\n
        _parameter_angle : float, defaults to ``0``
                           parameter used to define the blade shape using a cubic law.
                           Must be between 0 and 1.
        _Ra : float, defaults to ``0``
              Surface roughness of the impeller and its blades
        _taperratio : float, defaults to ``0``
                      taper ratio of the thickness for the blades.\n
                      Equals :math:`t_{\mathrm{hub}}/t_{\mathrm{tip}}`\n
                      with :math:`t_{\mathrm{tip}}` the thickness of the blade at the tip (``_inlet_blade_thickness[-1]`` for instance)\n
                      and :math:`t_{\mathrm{hub}}` the thickness of the blade at the hub (``_inlet_blade_thickness[0]`` for instance)
        _tapertype : float, defaults to ``'parabolic'``
                     type of thickness evolution along the blade in the radial direction.\n
                     parabolic or linear.\n
                     When parabolic, the thickness :math:`t` along the blade in the radial direction follows :math:`t(r) = 1 + \\left(t_{\mathrm{tip}}/t_{\mathrm{hub}} - 1\\right)r^2`\n
                     When parabolic, the thickness :math:`t` along the blade in the radial direction follows :math:`t(r) = 1 + \\left(t_{\mathrm{tip}}/t_{\mathrm{hub}} - 1\\right)r`\n
        _Xt : array of floats, defaults to ``None``
              normalized radial distances in the radial/axial plane (meridional view) of the blade as an elliptic arc for different lines describing a blade.
              value is 0 for the hub and 1 for the tip of the blade.\n
              1st result of a ``numpy.meshgrid`` with the number of lines to compute (along the radial direction of the blade) and the number of points in the axial direction.
              shape is (number of lines to computes, number of points on a line)
        _Theta : array of floats, defaults to ``None``
                 angles in radians to define the radial/axial plane (meridional view) of the blade as an elliptic arc for different lines describing a blade.
                 value is 0 for the top of the impeller (inlet) and :math:`\\pi/2` for the outlet of the impeller.\n
                 2nd result of a ``numpy.meshgrid`` with the number of lines to compute (along the radial direction of the blade) and the number of points in the axial direction.
                 shape is ``(number of lines to computes, number of points on a line)``
        _r : array of floats, defaults to ``None``
             radial coordinate in the meridional plane of the points to plot a blade from ``_Xt`` and ``_Theta``\n
             shape is ``(number of lines to computes, number of points on a line)``
        _z : array of floats, defaults to ``None``
             axial coordinate in the meridional plane of the points to plot a blade from ``_Xt`` and ``_Theta``\n
             shape is ``(number of lines to computes, number of points on a line)``
        _mt : array of floats, defaults to ``None``
              curvilinear distance along a blade using ``_r`` and ``_z``\n
              shape is ``(number of lines to computes, number of points on a line)``
        _beta : array of floats, defaults to ``None``
                blade angle in radians along the blade\n
                shape is ``(number of lines to computes, number of points on a line)``
        _phi : array of floats, defaults to ``None``
               polar angle in radians of the blade for a 3D blade in cylindrical coordinates\n
               shape is ``(number of lines to computes, number of points on a line)``
        _mt_adim : array of floats, defaults to ``None``
                   normalized curvilinear distance along a blade, ``_mt/_mt[:, -1]``
        _phi_allblades : array of floats, defaults to ``None``
                         polar angle in radians of the blades for 3D blades in cylindrical coordinates (see _phi)\n
                         shape is ``(_full_number_blade_w_splitters, number of lines to computes, number of points on a line)``
    """

    def __init__(self):
        """Constructor of Impeller class

            Sets all attributes to default values
        """
        #inputs
        self._inlet_diameters = [0, 0, 0] #D1 hub mid tip
        self._inlet_blade_thickness = [0, 0, 0] #t1 hub mid tip
        self._inlet_blade_angle = [0, 0, 0] #beta 1 hub mid tip

        self._outlet_diameter = 0 #D_2
        self._outlet_blade_height = 0 #b2
        self._outlet_blade_thickness = 0
        self._outlet_average_blade_thickness = 0 #t2
        self._outlet_blade_angle = 0 #Beta_2g

        self._axial_extension = 0 #L_z
        self._blade_running_clearance = 0 #epsilon_c
        self._number_blade_full = 0 #Nbfull = number of full blade
        self._splitter_blade_length_fraction = 0 #f_Lsplit
        
        #to compute
        self._blade_length = 0 #Lb
        self._blade_length_meanline = 0 #L
        self._effective_number_blade = 0 #Nbeff
        self._full_number_blade_w_splitters = 0 #Nb
        self._hydraulic_diameter = 0 #D_hyd
        self._inlet_area = 0 #A_1

        self._throat_pitch_blade = [0, 0, 0] #s hub mid tip
        self._throat_width = [0, 0, 0] #o hub mid tip
        self._throat_area = 0 #Atheff

        self._outlet_area = 0 #A_2

        self._inlet_optimal_angle = [0, 0, 0] #Beta hub, mid, tip

        self._parameter_angle = 0 #K in [0 ; 1]

        self._Ra = 3.2

        self._taperratio = 1.5 #blade taper ratio = thub/ttip
        self._tapertype = 'parabolic'

        self._Xt = None
        self._Theta = None
        self._r = None
        self._z = None
        self._mt = None
        self._beta = None
        self._phi = None
        self._mt_adim = None
        self._phi_allblades = None

    def update_geometry(self, set_angle=False):
        """Computes impeller geometry data used for computation

            1. sets the thickness of the blades
            2. computes the mid line diameter
            3. computes the optimal inlet blade angles
            4. if ``set_angle`` is ``True``, sets the inlet blade angles to optimal values
            5. computes the blade angle at the mean line (mid) of the inlet
            6. computes the blade length from :cite:t:`1970:Jansen`
            7. computes the effective number of blades and the full number of blades including splitters
            8. computes the hydraulic diameter from :cite:t:`1970:Jansen`
            9. computes the inlet flow area
            10. computes the throat flow area
            11. computes the outlet flow area
            12. computes the geometrical data to 3D plot the hub, mid and tip lines with 200 points along the blade in the axial direction
            13. computes the blade length at mean line (mid) from the 3D data

            Parameters
            ----------
            set_angle : bool, defaults to ``False``
                        define if inlet blade angles are modified to use the optimal angles or not
        """
        self.set_thicknesstaper()
        self.compute_mid_diameter()
        self.compute_optimal_angle()
        if set_angle: self.set_optimal_angle_overblade()
        self.compute_inlet_average_blade_angle()
        self.compute_blade_length()
        self.compute_number_blade()
        self.compute_hydraulic_diameter()
        self.compute_inlet_area()
        self.compute_throat_area()
        self.compute_outlet_area()
        self.compute_blades(nb_additional_lines= 0, nb_theta= 200, adjustthickness= False)
        self._blade_length_meanline = self._mt[-1, 1] - self._mt[0, 1]
    
    def compute_mid_diameter(self):
        """Computes the mean (mid) line diameter and stores it it in the corresponding class attribute.

            Updates ``_inlet_diameters[1]`` with :math:`D_{\\mathrm{mid}} = \\sqrt{\\frac{D_{\\mathrm{tip}}^2 + D_{\\mathrm{hub}}^2}{2}}`
        """
        self._inlet_diameters[1] = np.sqrt((self._inlet_diameters[0]**2 + self._inlet_diameters[-1]**2)/2)

    def compute_inlet_average_blade_angle(self):
        """Computes the average inlet blade angle and stores it in the corresponding class attribute.

            Updates ``_inlet_blade_angle[1]`` with the interpolation of the inlet blade angle at the mean (mid) line diameter 
        """
        a = np.array([[self._inlet_blade_angle[0], 0,self._inlet_blade_angle[-1]], [self._inlet_blade_angle[0], 0,self._inlet_blade_angle[-1]]])
        b = np.array([[0, np.sqrt(1/2.), 1], [0, np.sqrt(1/2.), 1]])
        self.interpolate_angle(angle= a, weight= b, eps=1e-12)
        self._inlet_blade_angle[1] = a[0, 1]

    def compute_blade_length(self):
        """Computes blade length of an impeller and stores it in the corresponding class attribute ``_blade_length``

            Updates ``_blade_length`` :math:`L_{\\mathrm{b}}` with the blade length computed using :cite:t:`1970:Jansen`:\n
                \t:math:`\\mathrm{cos}\\left(\\beta_{\\mathrm{average}}\\right) = \\frac{1}{2}\\left(\\mathrm{cos}\\left(\\beta_{1\\ \\mathrm{hub}}\\right) + \\mathrm{cos}\\left(\\beta_{1\\ \\mathrm{tip}}\\right)\\right)`
                \t:math:`L_{\\mathrm{b}} = \\frac{\\pi}{8} \\left( D_2 - \\frac{D_{\\mathrm{tip}} + D_{\\mathrm{hub}}}{2} - b_2 + 2 L_z \\right) \\frac{2}{\\mathrm{cos}\\left(\\beta_{\\mathrm{average}}\\right) + \\mathrm{cos}\\left(\\beta_2\\right)}`
        """
        cosbetaaverage = 0.5*(np.cos(self._inlet_blade_angle[0]) + np.cos(self._inlet_blade_angle[-1]))
        self._blade_length = np.pi/8* \
            (self._outlet_diameter - (self._inlet_diameters[0] + self._inlet_diameters[-1])/2 - self._outlet_blade_height + 2*self._axial_extension)* \
            2/(cosbetaaverage + np.cos(self._outlet_blade_angle))
        
    def compute_number_blade(self):
        """Computes the different number of blades and stores them in the corresponding class attribute.

            Updates ``_full_number_blade_w_splitters`` :math:`N_{\\mathrm{b}}` as :math:`N_{\\mathrm{b}} = 2N_{\\mathrm{b\\ full}}` if ``_splitter_blade_length_fraction``:math:`\\in \\left[10^{-5} ; 1\\right[` else :math:`N_{\\mathrm{b}} = N_{\\mathrm{b\\ full}}` where :math:`N_{\\mathrm{b\\ full}}` is the number of full blades (ie. no splitter).

            Updates ``_effective_number_blade`` :math:`N_{\\mathrm{b\\ effective}}` as :math:`N_{\\mathrm{b\\ effective}} = N_{\\mathrm{b\\ full}}( 1 +\ ` ``_splitter_blade_length_fraction`` :math:`)` if ``_splitter_blade_length_fraction``:math:`\\in \\left[10^{-5} ; 1\\right[` else :math:`N_{\\mathrm{b\\ effective}} = N_{\\mathrm{b\\ full}}`  where :math:`N_{\\mathrm{b\\ full}}` is the number of full blades (ie. no splitter).
            
        """
        self._effective_number_blade = self._number_blade_full*(1 + (1 if (self._splitter_blade_length_fraction > 1e-5 and self._splitter_blade_length_fraction < 1 ) else 0))
        self._full_number_blade_w_splitters = int(self._number_blade_full*(1 + (1 if (self._splitter_blade_length_fraction > 1e-5 and self._splitter_blade_length_fraction < 1 ) else 0)))

    def compute_hydraulic_diameter(self):
        """Computes the hydraulic diameter and stores it in the corresponding class attribute ``_hydraulic_diameter``

            Updates ``_hydraulic_diameter`` :math:`D_{\\mathrm{hyd}}` using :cite:t:`1970:Jansen`:\n
                \t:math:`\\mathrm{cos}\\left(\\beta_{\\mathrm{average}}\\right) = \\frac{1}{2}\\left(\\mathrm{cos}\\left(\\beta_{1\\ \\mathrm{hub}}\\right) + \\mathrm{cos}\\left(\\beta_{1\\ \\mathrm{tip}}\\right)\\right)`\n
                \t:math:`\\frac{D_{\\mathrm{hyd}}}{D_2} = \\frac{\\mathrm{cos}\\left(\\beta_2\\right)}{\\frac{N_\\mathrm{b\\ effective}}{\\pi} + D_2\\frac{\\mathrm{cos}\\left(\\beta_2\\right)}{b_2}} + \\frac{\\frac{1}{2} \\left(\\frac{D_{1\\ \\mathrm{tip}}}{D_2} + \\frac{D_{1\\ \\mathrm{hub}}}{D_2} \\right) \\mathrm{cos}\\left(\\beta_{\\mathrm{average}} \\right)}{\\frac{N_{\\mathrm{b\\ effective}}}{\\pi} + \\frac{D_{1\\ \\mathrm{tip}} + D_{1\\ \\mathrm{hub}}}{D_{1\\ \\mathrm{tip}} - D_{1\\ \\mathrm{hub}}} \\mathrm{cos}\\left(\\beta_{\\mathrm{average}}\\right)}`
        """
        cosbetaaverage = 0.5*(np.cos(self._inlet_blade_angle[0]) + np.cos(self._inlet_blade_angle[-1]))
        self._hydraulic_diameter = \
            (np.cos(self._outlet_blade_angle)/ \
            (   self._effective_number_blade/np.pi + \
                self._outlet_diameter*np.cos(self._outlet_blade_angle)/self._outlet_blade_height
            ) + \
            (   0.5*(self._inlet_diameters[-1]/self._outlet_diameter + self._inlet_diameters[0]/self._outlet_diameter)* \
                cosbetaaverage
            )/ \
            (   self._effective_number_blade/np.pi + \
                (self._inlet_diameters[-1] + self._inlet_diameters[0])/(self._inlet_diameters[-1] - self._inlet_diameters[0])*\
                cosbetaaverage
            )
            )*self._outlet_diameter
    
    def compute_inlet_area(self):
        """Computes the inlet area and stores it in the corresponding class attribute ``_inlet_area``

            Updates ``_inlet_area`` with :math:`\\frac{\\pi}{4} \\left( D_{1\\ \\mathrm{tip}}^2 - D_{1\\ \\mathrm{hub}}^2 \\right)`
        """
        self._inlet_area = np.pi/4*(self._inlet_diameters[-1]**2 - self._inlet_diameters[0]**2)
    
    def compute_pitch_blade(self, D, t):
        """Computes and returns the pitch of a blade at the inlet for a line along the blade situated at a diameter ``D`` and a thickness ``t``

            Computes and returns :math:`\\pi D/N_{\\mathrm{b\\ full}} - t_1` where :math:`D` is the diameter where this is computed (hub, mid or tip) and :math:`t_1` is the corresponding blade thickness. 

            Parameters
            ----------
            D : float
                diameter of the line where the computation is done. hub, mid or tip typically.
            t : float
                thickness of the blade at the diameter ``D``

            Returns
            -------
            s : float
                max between 0 and the pitch of the blade at this position as :math:`\\pi D/N_{\\mathrm{b\\ full}} - t_1` where :math:`D` is the diameter where this is computed (hub, mid or tip) and :math:`t_1` is the corresponding blade thickness. 

            Raise
            -----
            RuntimeError
                When the pitch is negative or zero, the fluid can't pass through

            Note
            ----
            The pitch of the blade is the distance between 2 blades at the diameter (ie how far are the 2 points defining the blade at the diameter of interest) and not the smallest distance between 2 blades which defines the throat.
        """
        r = np.pi*D/self._number_blade_full - t
        if r <= 0:
            raise RuntimeError("Throat pitch blade is negative or null ie. passage is blocked. D = {:.2e}  _number_blade_full = {:.2e}  t = {:.2e}".format(D, self._number_blade_full, t))
        
        return max(0, np.pi*D/self._number_blade_full - t)
    
    def compute_throat_area(self):
        """Computes the throat area for hub, mid and tip position and stores them in the corresponding class attribute.

            1. Updates ``_throat_pitch_blade`` using :func:`~geometry.Impeller.compute_pitch_blade`
            2. Updates ``_throat_width`` as :math:`o_i = s_i \\mathrm{cos}\\left( \\beta_{1,\\ i} \\right)` where :math:`o_i` is the throat width of line :math:`i` (hub, mid or tip), :math:`s_i` is the blade pitch of line :math:`i` from ``_throat_pitch_blade`` and :math:`\\beta_{1,\\ i}` is the inlet blade angle for the line :math:`i`
            3. Updates ``_throat_area`` using geometrical considerations and blockage due to diffusion according to :cite:t:`2000:Aungier`
                The geometrical passage area is given by :math:`\\frac{N_{\\mathrm{b\\ full}}}{4} \\left( o_{\\mathrm{hub}}\\left( D_{\\mathrm{mid}} - D_{\\mathrm{hub}} \\right) + o_{\\mathrm{mid}}\\left( D_{\\mathrm{tip}} - D_{\\mathrm{hub}} \\right) + o_{\\mathrm{tip}}\\left( D_{\\mathrm{tip}} - D_{\\mathrm{mid}} \\right) \\right)`\n
                The reduction is given by :math:`AR = \\frac{A_1 \\mathrm{cos}\\left( \\beta_{1\\ \\mathrm{mid}} \\right)}{A_{\\mathrm{throat}}}`\n
                If :math:`AR < 10^{-10}` the throat area is not modified. Else, the throat area :math:`A_{\\mathrm{throat}}` is given by :math:`A_{\\mathrm{throat}} = A_{\\mathrm{throat}} \\mathrm{min}\\left( \\sqrt{AR} \\ ;\\ 1 - (AR - 1)^2 \\right)`  
        """
        self._throat_pitch_blade = [self.compute_pitch_blade(D= self._inlet_diameters[i], t= self._inlet_blade_thickness[i]) for i in range(0, 3)]

        self._throat_width = [self._throat_pitch_blade[i]*np.cos(self._inlet_blade_angle[i]) for i in range(0, 3)]

        self._throat_area = self._number_blade_full/4*(
            self._throat_width[0]*(self._inlet_diameters[1] - self._inlet_diameters[0]) + \
            self._throat_width[1]*(self._inlet_diameters[-1] - self._inlet_diameters[0]) + \
            self._throat_width[-1]*(self._inlet_diameters[-1] - self._inlet_diameters[1])
        )

        AR = self._inlet_area * np.abs(np.sin(self._inlet_blade_angle[1]))/self._throat_area

        if np.abs(AR) > 1e-10:
            self._throat_area = self._throat_area*np.min([np.sqrt(AR), 1 - (AR - 1)**2])
         
    def compute_outlet_area(self):
        """Computes the outlet area and stores it in the corresponding class attribute ``_outlet_area``.

            Updates ``_outlet_area`` with :math:`\\left( \\pi D_2 - N_{\\mathrm{b}} t_2 \\right) b_2`
        """
        self._outlet_area = (np.pi*self._outlet_diameter - self._full_number_blade_w_splitters*self._outlet_average_blade_thickness)*self._outlet_blade_height

    def compute_optimal_angle(self):
        """Computes the optimal inlet angles for hub, tip and mid and stores them in the corresponding class attribute ``_inlet_optimal_angle``.

        Updates ``_inlet_optimal_angle`` with :math:`\\beta_{1\\ \\mathrm{opti}\\ i} = \\mathrm{arctan}\\left( \\frac{\\pi D_{1\\ i}}{\\pi D_{1\\ i} - N_{\\mathrm{b\\ full}} t_{1\\ i}} \\mathrm{tan}\\left( \\beta_{1\\ i} \\right)\\right)` with :math:`i` standing for the line reference (hub, mid or tip)

        Note
        ----
        Verification TODO according to S. Parisi, p19 where :math:`\\mathrm{tan}\\left(\\beta_{1\\ \\mathrm{opt}\\ i}\\right)/D_{1\\ i} = (\\omega/(2V_{1m})) = \\mathrm{const.}`
        """
        self._inlet_optimal_angle = [
            np.arctan(np.pi*self._inlet_diameters[i]/(np.pi*self._inlet_diameters[i] - self._number_blade_full*self._inlet_blade_thickness[i])*\
            np.tan(self._inlet_blade_angle[i])) for i in range(0, 3)
        ]

    def set_optimal_angle_overblade(self):
        """Sets ``_inlet_blade_angle`` using an optimal angle distribution derived from the rotation speed and meridional absolute velocity

            Sets the different blades angles with :math:`\\mathrm{tan}\\left(\\beta_{1\\ \\mathrm{opt}\\ i}\\right)/D_{1\\ i} = (\\omega/(2V_{1m})) = \\mathrm{const.}` where :math:`i` stands for the line considered (hub, mid)
            
            Note
            ----
            Inlet tip blade angle ``_inlet_blade_angle[2]`` is set using value from ``_inlet_optimal_angle``. The latter should be computed with :func:`~geometry.Impeller.compute_optimal_angle` by the user or another method before.
        """
        self._inlet_blade_angle[-1] = self._inlet_optimal_angle[-1]
        self._inlet_blade_angle[0] = np.arctan(np.tan(self._inlet_blade_angle[-1]) * self._inlet_diameters[0]/self._inlet_diameters[-1])
        self._inlet_blade_angle[1] = np.arctan(np.tan(self._inlet_blade_angle[-1]) * self._inlet_diameters[1]/self._inlet_diameters[-1])

    def compute_fictitious_angle(self):
        """Computes and returns the fictitious angle used for the cubic law to generate the impeller geometry.

        Compute and returns :math:`\\beta_x = (1 - K)\\left( \\beta_{1\\ \\mathrm{hub}} + \\beta_2 \\right)/2` used for the cubic law where :math:`K` is a parameter of the model to adjust blade geometry stored in ``_parameter_angle``
        From :cite:t:`2012:Parisi` equation 2.64.
        
        Returns
        -------
        beta : float
               Fictitious angle used in the cubic law :math:`\\beta_x = (1 - K)\\left( \\beta_{1\\ \\mathrm{hub}} + \\beta_2 \\right)/2` from :cite:t:`2012:Parisi` equation 2.64.
        """
        return (1 - self._parameter_angle)*(self._inlet_blade_angle[0] + self._outlet_blade_angle)/2.

    def compute_angle_parametersABC(self):
        """Computes and returns the polynomial parameters used for the cubic law to generate the impeller geometry.

        The following equations are used, with :math:`\\beta_x` the fictitious angle given by :func:`~geometry.Impeller.compute_fictitious_angle`:\n
        :math:`A = -4 \\beta_2 + 8 \\beta_x - 4 \\beta_{1\\ \mathrm{hub}}`\n
        :math:`B = 11 \\beta_2 - 16 \\beta_x + 5 \\beta_{1\\ \mathrm{hub}}`\n
        :math:`C = -6 \\beta_2 + 8 \\beta_x - 2 \\beta_{1\\ \mathrm{hub}}`

        From :cite:t:`2012:Parisi` equation 2.63.

        Returns
        -------
        A, B, C : tuple of floats
                  polynomial parameters used in the cubic law
        """
        betax = self.compute_fictitious_angle()
        A = -4*self._outlet_blade_angle + 8*betax - 4*self._inlet_blade_angle[0]
        B = 11*self._outlet_blade_angle - 16*betax + 5*self._inlet_blade_angle[0]
        C = -6*self._outlet_blade_angle + 8*betax - 2*self._inlet_blade_angle[0]
        return A, B, C

    def compute_tx(self, x):
        """ Computes and returns the normalized blade thickness for a point of the blade at normalized radial distance ``x``.

        Depending on ``_tapertype`` returns:\n
        :math:`t(x) = 1 + (t_{\\mathrm{hub}}/t_{\\mathrm{tip}} - 1 ) x^2` if ``_tapertype`` is ``parabolic`` where :math:`t_{\\mathrm{hub}}/t_{\\mathrm{tip}}` is stored in ``_taperratio``\n
        :math:`t(x) = 1 + (t_{\\mathrm{hub}}/t_{\\mathrm{tip}} - 1 ) x` if ``_tapertype`` is ``linear`` where :math:`t_{\\mathrm{hub}}/t_{\\mathrm{tip}}` is stored in ``_taperratio``\n
        raises a ``NotImplementedError`` else.

        Parameters
        ----------
        x : float
            Normalized radial distance between 0 (tip) and 1 (hub) where the normalized blade thickness is computed

        Returns
        -------
        t(x) : float
               Normalized blade thickness at point x depending on ``_tapertype`` value

        Raises
        ------
        NotImplementedError
            If ``tapertype`` is not ``parabolic`` or ``linear``.

        Note
        ----
        The ``x`` parameter goes from the tip (x = 0) to the hub (x = 1).
        """
        if 'parabolic' == self._tapertype:
            return 1 + (self._taperratio - 1)*x**2
        elif 'linear' == self._tapertype:
            return 1 + (self._taperratio - 1)*x
        else:
            raise NotImplementedError("Taper type should be parabolic or linear")

    def set_thicknesstaper(self):
        """Computes and stores the blade thicknesses in the corresponding class attributes

        1. Updates the hub blade thickness in ``_inlet_blade_thickness[0]`` using the tip blade thickness stored in ``_inlet_blade_thickness[2]`` and :func:`~geometry.Impeller.compute_tx`.
        2. Updates the mid line (root mean square value) blade thickness ``_inlet_blade_thickness[1]`` using the tip blade thickness stored in ``_inlet_blade_thickness[2]`` and :func:`~geometry.Impeller.compute_tx`.
        3. Updates the outlet average blade thickness ``_outlet_average_blade_thickness`` :math:`\\left< t_2 \\right>` from the outlet tip blade thickness stored in ``_outlet_blade_thickness`` :math:`t_2` and the ``_taperratio`` :math:`t_{\\mathrm{hub}}/t_{\\mathrm{tip}}` with\n
            :math:`\\left< t_2 \\right> = \\left( 1 + \\frac{1}{2}\\left( t_{\\mathrm{hub}}/t_{\\mathrm{tip}} - 1 \\right) \\right) t_2` if ``_tapertype`` is ``parabolic``\n
            :math:`\\left< t_2 \\right> = \\left( 1 + \\frac{1}{3}\\left( t_{\\mathrm{hub}}/t_{\\mathrm{tip}} - 1 \\right) \\right) t_2` if ``_tapertype`` is ``linear``\n
            else raises ``NotImplementedError``
        
        Raises
        ------
        NotImplementedError
            If ``tapertype`` is not ``parabolic`` or ``linear``.
        """
        self._inlet_blade_thickness[0] = self._taperratio*self._inlet_blade_thickness[-1]
        self._inlet_blade_thickness[1] = self.compute_tx(1 - np.sqrt(0.5))*self._inlet_blade_thickness[-1]

        if 'parabolic' == self._tapertype:
            self._outlet_average_blade_thickness = (1 + 0.5*(self._taperratio - 1))*self._outlet_blade_thickness
        elif 'linear' == self._tapertype:
            self._outlet_average_blade_thickness = (1 + 1/3*(self._taperratio - 1))*self._outlet_blade_thickness
        else:
            raise NotImplementedError("Taper type should be parabolic or linear")

    def compute_radial_coordinate(self, xt, theta):
        """Computes and returns the radial coordinate in the meridional plane of the points to plot a blade from ``_Xt`` and ``_Theta``

            Radial coordinate is given by :math:`2r = D_2 - \\left( D_2 - D_{1\\ \\mathrm{hub}} - x\\left( D_{1\\ \\mathrm{tip}} - D_{1\\ \\mathrm{hub}} \\right) \\mathrm{cos} \\left( \\theta \\right) \\right)`
            with :math:`x` from argument ``xt`` the position along the blade from hub (0) to tip (1) and :math:`\\theta` from argument theta the angle description of the blade from inlet (0) to outlet (:math:`\\pi/2`)

            Parameters
            ----------
            xt : numpy array of floats
                x position along the blade from hub (0) to tip (1)
            theta : numpy array of floats
                    angle description of the blade from inlet (0) to outlet (:math:`\\pi/2`) in radians

            Returns
            -------
            r : numpy array of floats
                radial coordinates of the points of the blade
        """
        return 0.5*(self._outlet_diameter - (self._outlet_diameter - self._inlet_diameters[0] - xt*(self._inlet_diameters[2] - self._inlet_diameters[0]))*np.cos(theta))
    
    def compute_axial_coordinate(self, xt, theta):
        """Computes and returns the axial coordinate of for an impeller using its position along the blade.

        Axial coordinate is given by :math:`z = (-L_z + (x - 0.5)b_2)\\mathrm{sin}\\left(\\theta\\right)` 
        with :math:`L_z` the axial extension of the blade stored in ``_axial_extension``, :math:`x` from argument xt the position along the blade from hub (0) to tip (1) and :math:`\\theta` from argument theta the angle description of the blade from inlet (0) to outlet (:math:`\\pi/2`)

        Parameters
        ----------
        xt : numpy array of floats
            x position along the blade from hub (0) to tip (1)
        theta : numpy array of floats
            angle description of the blade from inlet (0) to outlet (:math:`\\pi/2`) in radians

        Returns
        -------
        z : numpy array of floats
            axial coordinates of the points of the blade
        """
        return (-self._axial_extension + (xt - 0.5)*self._outlet_blade_height)*np.sin(theta)
    
    def compute_curvilinear_abscissa(self, xt=None, theta=None, r=None, z=None):
        """Computes and returns the curvilinear abscissa along a blade similar to ``_mt`` but do not update it.

        The curvilinear abscissa is given by :math:`m = \\int \\sqrt{\\left( dr \\right)^2 + \\left( dz \\right)^2}`
        with :math:`dr` an infinitesimal element of the radial coordinate and :math:`dz` an infinitesimal element of the axial coordinate

        Parameters
        ----------
        xt : numpy array of floats, defaults to ``None``
             x position along the blade from hub (0) to tip (1)
        theta : numpy array of floats, defaults to ``None``
                angle description of the blade from inlet (0) to outlet (:math:`\\pi/2`) in radians
        r : numpy array of floats, defaults to ``None``
            radial coordinates of the points of the blade given by :func:`~geometry.Impeller.compute_radial_coordinate`
            if None, ``xt`` and ``theta`` are used to compute ``r``
        z : numpy array of floats, defaults to ``None``
            axial coordinates of the points of the blade given by :func:`~geometry.Impeller.compute_axial_coordinate`
            if None, ``xt`` and ``theta`` are used to compute ``z``

        Returns
        -------
        mt, z, r : tuple of numpy arrays
                   arrays giving the curvilinear abscissa along a blade, axial and radial coordinates given by :func:`~geometry.Impeller.compute_radial_coordinate` and :func:`~geometry.Impeller.compute_axial_coordinate` respectively
        """
        if (r is None):
            r = self.compute_radial_coordinate(xt, theta)
        if (z is None):
            z = self.compute_axial_coordinate(xt, theta)
        mt = np.zeros(xt.shape)
        for i in range(1, mt.shape[0]):
            mt[i] = mt[i-1] + np.sqrt((r[i] - r[i-1])**2 + (z[i] - z[i-1])**2)
        return mt, z, r
    
    def compute_curvilinear_abscissa_adim(self, m):
        """Scales and returns the normalized curvilinear abscissa along a blade so that it's between 0 and 1, similar to ``_mt_adim`` but do not update it.

        The normalized curvilinear abscissa is given by :math:`\\tilde{m} = m/m(\\theta = \\pi/2)` for each line of the blade (x position).

        Parameters
        ----------
        m : numpy array of floats
            curvilinear abscissa along a blade given by the :func:`~geometry.Impeller.compute_curvilinear_abscissa`

        Returns
        -------
        mt_adim : numpy array of floats
                  normalized curvilinear abscissa
        """
        return m/m[-1,:]
    
    def compute_angle_blade(self, xt=None, theta=None, r=None, z=None, mt=None):
        """Computes and returns the blade angles of the impeller blade, similar to ``_beta`` but do not update it

        Using :func:`~geometry.Impeller.compute_angle_parametersABC` to get the A, B and C polynomial parameters, the blade angle are given by:\n
        :math:`\\beta_{\\mathrm{hub}} = \\beta_{1\\ \\mathrm{hub}} + A\\tilde{m}(x = 0) + B\\left(\\tilde{m}(x = 0)\\right)^2 + C\\left(\\tilde{m}(x = 0)\\right)^3`\n
        :math:`\\beta_{\\mathrm{tip}} = \\beta_{1\\ \\mathrm{tip}} + \\left( \\beta_2 - \\beta_{1\\ \\mathrm{tip}} \\right)\\left( 3\\left(\\tilde{m}(x = 1)\\right)^2 - 2\\left(\\tilde{m}(x = 1)\\right)^3\\right)`\n
        with 1 standing for inlet data, 2 standing for outlet data and :math:`\\tilde{m}` given by :func:`~geometry.Impeller.compute_curvilinear_abscissa_adim` or computed manually with :func:`~geometry.Impeller.compute_curvilinear_abscissa`.

        Parameters
        ----------
        xt : numpy array of floats, defaults to ``None``
            x position along the blade from hub (0) to tip (1)
        theta : numpy array of floats, defaults to ``None``
            angle description of the blade from inlet (0) to outlet (:math:`\\pi/2`) in radians
        r : numpy array of floats, defaults to ``None``
            radial coordinates of the points of the blade given by :func:`~geometry.Impeller.compute_radial_coordinate`
            if ``None``, ``xt`` and ``theta`` are used to compute ``r``
        z : numpy array of floats, defaults to ``None``
            axial coordinates of the points of the blade given by :func:`~geometry.Impeller.compute_axial_coordinate`
            if ``None``, ``xt`` and ``theta`` are used to compute ``z``
        mt : numpy array of floats, defaults to ``None``
            curvilinear abscissa of the blades given by :func:`~geometry.Impeller.compute_curvilinear_abscissa`
            if ``None``, will be computed using :func:`~geometry.Impeller.compute_curvilinear_abscissa`

        Returns
        -------
        beta, mt, z, r : tuple of numpy array of floats
                         beta are the angles along the blade, the others are described in the Parameters section.
        """
        if (mt is None):
            mt, z, r = self.compute_curvilinear_abscissa(xt, theta, r, z)
        A, B, C = self.compute_angle_parametersABC()
        beta = np.zeros(mt.shape)
        beta[:, 0] = self._inlet_blade_angle[0] + A*(mt[:, 0]/mt[-1][0]) + B*(mt[:, 0]/mt[-1][0])**2 + C*(mt[:, 0]/mt[-1][0])**3
        beta[:, -1] = self._inlet_blade_angle[2] + (self._outlet_blade_angle - self._inlet_blade_angle[2])*(3*(mt[:, -1]/mt[-1][-1])**2 - 2*(mt[:, -1]/mt[-1][-1])**3)

        self.interpolate_angle(angle= beta, weight= xt, eps=1e-12)

        return beta, mt, z, r
    
    def interpolate_angle(self, angle, weight, eps=1e-12):
        """Computes and updates angle argument to interpolate between the two angles given by ``angle[:, 0]`` and ``angle[:, -1]``.

            The interpolation of angles is given by: 
            :math:`\\alpha = \\left( 2 \\left( \\alpha_{\\mathrm{max}} - \\alpha_{\\mathrm{min}} \\right) \\% (2\\pi) - \\left( \\alpha_{\\mathrm{max}} - \\alpha_{\\mathrm{min}} \\right) \\right) x + \\alpha_{\\mathrm{min}}`\n
            with :math:`\\alpha_{\\mathrm{max}}` given by ``angle[:, -1]``, :math:`\\alpha_{\\mathrm{min}}` given by ``angle[:, 0]`` and :math:`x` given by ``weight[:, i]`` for the point :math:`i`.

            Parameters
            ----------
            angle: array of floats of shape (n, m>2)
                   angles in radians to interpolate. 
                   The columns (m) ``0`` and ``-1`` are used as the two angles to interpolate from.
                   Results are stored in columns ``1:-1``.
            weight : array of floats of shape (n, m>2)
                     should be the same shape as angle. 
                     Weight between 0 and 1 for the interpolation between ``angle[:, 0]`` (``weight = 0``) and ``angle[:, -1]`` (``weight = 1``).
            eps: float, defaults to ``1e-12``
                 precision to find zeros.
        """
        if not angle.shape[1] > 2:
            return
        d = (np.where(np.abs(angle[:, -1] - angle[:, 0]) > eps, angle[:, -1] - angle[:, 0], 0))%(2*np.pi)
        angle[:, 1:-1] = (2*d%(2*np.pi) - d).reshape(-1, 1)*np.ones(angle[:, 1:-1].shape)*weight[:, 1:-1] \
                       + angle[:, 0].reshape(-1, 1)*np.ones(angle[:, 1:-1].shape)

    def compute_polar_angle(self, xt=None, theta=None, r=None, z=None, mt=None, beta=None, adjustthickness=False):
        """Computes and returns the polar angle of an impeller blade, similar to ``_phi`` but do not update it

        The polar angle is given by:
        :math:`\\varphi(\\tilde m) = \\int \\mathrm{tan}\\left( \\beta \\right) \\frac{dm}{r}` with:\n
        :math:`\\tilde m` the normalized curvilinear abscissa (see :func:`~geometry.Impeller.compute_curvilinear_abscissa_adim`),\n
        :math:`m` the curvilinear abscissa (see :func:`~geometry.Impeller.compute_curvilinear_abscissa`),\n
        :math:`\\beta` the blade angle (see :func:`~geometry.Impeller.compute_angle_blade`) and\n
        :math:`r` the radial coordinate (see :func:`~geometry.Impeller.compute_radial_coordinate`).\n
        The polar angle is computed for the hub and tip line then interpolated for each other ``xt`` positions between hub and tip with :func:`~geometry.Impeller.interpolate_angle`.

        When using the ``adjustthickness`` parameter, the thickness currently do not uses the thickness distribution profile given by ``_tapertype``.
        For the inlet, the thickness is linearly interpolated using ``xt`` and ``_inlet_blade_thickness`` for hub and tip only.
        For the outlet, the thickness is taken constant and equals to ``_outlet_average_blade_thickness``.
        Additionally, only one one value is modified for the inlet and for the outlet insofar as only the values for :math:`\\theta = 0` and :math:`\\theta = \\pi/2` are modified 
        leading to clunky contours when the descretization in theta is good for a smoot plot.

        Parameters
        ----------
        xt : numpy array of floats, defaults to ``None``
             x position along the blade from hub (0) to tip (1)
        theta : numpy array of floats, defaults to ``None``
                angle description of the blade from inlet (0) to outlet (:math:`\\pi/2`) in radians
        r : numpy array of floats, defaults to ``None``
            radial coordinates of the points of the blade given by :func:`~geometry.Impeller.compute_radial_coordinate`
            if ``None``, ``xt`` and ``theta`` are used to compute ``r``
        z : numpy array of floats, defaults to ``None``
            axial coordinates of the points of the blade given by :func:`~geometry.Impeller.compute_axial_coordinate`
            if ``None``, ``xt`` and ``theta`` are used to compute ``z``
        mt : numpy array of floats, defaults to ``None``
            curvilinear abscissa of the blades given by :func:`~geometry.Impeller.compute_curvilinear_abscissa`
            if ``None``, will be computed using :func:`~geometry.Impeller.compute_curvilinear_abscissa`
        beta : numpy array of floats, defaults to ``None``
               angles along the blade given by :func:`~geometry.Impeller.compute_angle_blade method`
               if ``None``, will be computed using :func:`~geometry.Impeller.compute_angle_blade method`
        adjustthickness : bool, defaults to ``False``
                          bool to adjust or not the angles on the pressure side (outlet) and suction side (inlet) with the thicknesses of the blade ``_outlet_average_blade_thickness`` and ``_inlet_blade_thickness`` respectively

        Returns
        -------
        phi, beta, mt, z, r : tuple of numpy arrays of floats
                              phi stores the polar angles of the blades, the others are described in the Parameters section.

        Note
        ----
        TODO Improve the adjustment of the angle with the thickness computation and thickness profile. See the corresponding paragraph in the detailed description of this function.
        """
        if (beta is None):
            beta, mt, z, r = self.compute_angle_blade(xt, theta, r, z, mt)
        phi = np.zeros(beta.shape)
        for i in range(1, phi.shape[0]):
            phi[i] = phi[i-1] + np.tan(beta[i])/r[i] * (mt[i] - mt[i-1])
        self.interpolate_angle(angle= phi, weight= xt, eps=1e-12)
        if adjustthickness:
            #thickness is estimated and should be improved later on
            phi[0, :] = phi[0, :] + \
                ((1 - xt[0, :])*self._inlet_blade_thickness[0] + xt[0, :]*self._inlet_blade_thickness[-1])/2./r[0,:]
            phi[-1, :] = phi[-1, :] - \
                np.ones(r[-1,:].shape)*self._outlet_average_blade_thickness/2./r[-1,:]
        return phi, beta, mt, z, r
    
    def compute_polar_angle_repet(self, phi):
        """Computes and returns the polar angles of an impeller's blade to plot all blades by doing a repetition of outputs from :func:`~geometry.Impeller.compute_polar_angle` using ``_full_number_blade_w_splitters``. 

        Parameters
        ----------
        phi : numpy array of floats
              polar angle of a blade given by :func:`~geometry.Impeller.compute_polar_angle`

        Returns
        -------
        phi_allblades : numpy array of arrays of floats
                        array containing the polar angles for each blade
        """
        offset = 2*np.pi/self._full_number_blade_w_splitters
        phi_allblades = np.zeros((self._full_number_blade_w_splitters, *phi.shape))
        for i in range(0, self._full_number_blade_w_splitters):
            phi_allblades[i] = phi + i*offset
            
        return phi_allblades
    
    def cut_splitters(self, xt, mt, phi_allblades):
        """Compute and updates the polar angles of splitters to 0 when splitters are used and there shouldn't be a value (no blade present due to the splitter's length)

        When splitters are not used (``_splitter_blade_length_fraction >= 1`` or ``self._splitter_blade_length_fraction <= 1e-5``), does nothing.

        Parameters
        ----------
        xt : numpy array of floats, defaults to ``None``
             x position along the blade from hub (0) to tip (1)
        mt : numpy array of floats, defaults to ``None``
             curvilinear abscissa of the blades given by :func:`~geometry.Impeller.compute_curvilinear_abscissa`
             if ``None``, will be computed using :func:`~geometry.Impeller.compute_curvilinear_abscissa`
        phi_allblades : numpy array of floats
                        array containing the polar angles for each blade given by :func:`~geometry.Impeller.compute_polar_angle_repet`

        Note
        ----
        TODO Improves the cutting process so that if only 2 lines are used (hub and tip), use self._blade_length to cut the blade
        """
        if self._splitter_blade_length_fraction >= 1 or self._splitter_blade_length_fraction <= 1e-5:
            return
        
        index = np.absolute(xt[0,:] - 0.5).argmin() #index of meanline
        newlength = (1 - self._splitter_blade_length_fraction)*mt[-1, index]
        for i in range(0, int(self._full_number_blade_w_splitters/2)):
            indexes_to_cut = np.argwhere(mt[:, index] < newlength)
            phi_allblades[2*i + 1, :indexes_to_cut[-1][0]+1] = 0

    def compute_blades(self, nb_additional_lines= 0, nb_theta= 200, adjustthickness= False):
        """Wrapper function to computes all the geometrical data to define the blades and stores them in the corresponding class attributes (``_phi`` etc.).

        Defines a numpy array holding each x position for (hub + tip + ``nb_additional_lines``) lines and another numpy array for each :math:`\\theta` for each point where lines needs to be computed.
        These are used to define a numpy meshgrid.\n
        Successively do:
        1. Updates class attribute ``_r`` with :func:`~geometry.Impeller.compute_radial_coordinate`.
        2. Updates class attribute ``_z`` with :func:`~geometry.Impeller.compute_axial_coordinate`.
        3. Updates class attribute ``_mt`` with :func:`~geometry.Impeller.compute_curvilinear_abscissa`.
        4. Updates class attribute ``_beta`` with :func:`~geometry.Impeller.compute_angle_blade`.
        5. Updates class attribute ``_phi`` with :func:`~geometry.Impeller.compute_polar_angle`.
        6. Updates class attribute ``_mt_adim`` with :func:`~geometry.Impeller.compute_curvilinear_abscissa_adim`.
        8. Updates class attribute ``_phi_allblades`` with :func:`~geometry.Impeller.compute_curvilinear_abscissa`.
        9. Updates class attribute ``_mt`` with :func:`~geometry.Impeller.compute_polar_angle_repet`.
        10. Updates class atribute ``_phi_allblades`` with :func:`~geometry.Impeller.cut_splitters`.

        Parameters
        ----------
        nb_additional_lines : int, defaults to ``0``
                            Number of lines to plot in addition to the hub and tip lines. When equalts to 0, the mid line (corresponding to the root mean square value) is still computed.
        nb_theta : int, defaults to ``200``
                Number points used to discretize a single line in terms of theta
        adjustthickness : bool, defaults to ``False``
                          bool to adjust or not the angles on the pressure side (outlet) and suction side (inlet) with the thicknesses of the blade ``_outlet_average_blade_thickness`` and ``_inlet_blade_thickness`` respectively
        
        Returns
        -------
        _phi_allblades, _mt_adim, _phi, _beta, _mt, _z, _r, _Xt, _Theta, theta, xt : tuple of numpy array
                                                                                     Refers to class attributes documentation for the details.
        """
        number_xt = 2 + nb_additional_lines #nombre de lignes tracees
        xt = np.linspace(start= 0, stop=1, num=number_xt)
        if 2 == number_xt:
            xt = np.concatenate((xt, np.array([np.sqrt(1/2.)])))
            xt = np.sort(xt, kind= "mergesort")
            number_xt = number_xt + 1
        theta = np.linspace(start=0, stop=np.pi/2, num=nb_theta)
        self._Xt, self._Theta = np.meshgrid(xt, theta)
        self._r = self.compute_radial_coordinate(self._Xt, self._Theta)
        self._z = self.compute_axial_coordinate(self._Xt, self._Theta)
        self._mt, self._z, self._r = self.compute_curvilinear_abscissa(xt= self._Xt, theta= self._Theta, r= self._r, z= self._z)
        self._beta, self._mt, self._z, self._r = self.compute_angle_blade(xt= self._Xt, theta= self._Theta, r= self._r, z= self._z, mt= self._mt)
        self._phi, self._beta, self._mt, self._z, self._r = self.compute_polar_angle(xt= self._Xt, theta= self._Theta, r= self._r, z= self._z, mt= self._mt, beta= self._beta, adjustthickness= adjustthickness)
        self._mt_adim = self.compute_curvilinear_abscissa_adim(self._mt)
        self._phi_allblades = self.compute_polar_angle_repet(self._phi)
        self.cut_splitters(self._Xt, self._mt, self._phi_allblades)

        return self._phi_allblades, self._mt_adim, self._phi, self._beta, self._mt, self._z, self._r, self._Xt, self._Theta, theta, xt
    
    def compare_phi_find_param_angle(self, K):
        """Updates ``_parameter_angle`` with ``K[0]`` and calls :func:`~geometry.Impeller.compute_blades` to re-compute the blade geometry

        This function is usually used to define a ``_parameter_angle`` so that the blade at the outlet is perfectly vertical 
        using an optimization function such as ``scipy.optimize.minimize``.

        Parameters
        ----------
        K : array of float of size 1
            array containing the new value of ``_parameter_angle`` in ``K[0]``. Should be between 0 and 1.
        
        Returns
        -------
        delta : float
                absolute value of the difference of polar angles at the outlet for tip and hub lines: :math:`\\left| \\varphi_{2\\ \\mathrm{tip}} - \\varphi_{2\\ \\mathrm{hub}} \\right|`
        """
        self._parameter_angle = K[0]
        self.compute_blades(adjustthickness= False)
        return np.abs(self._phi[-1, -1] - self._phi[-1, 0])

class VanelessDiffuser():
    """Class defining and holding all geometrical data for the vaneless diffuser of a centrifugal machine

        Attributes
        ----------
        _outlet_diameter : float, defaults to ``0``
                           Outlet diameter of the vaneless diffuser
        _outlet_height : float, defaults to ``0``
                         Outlet height of the vaneless diffuser
        _outlet_area : float, defaults to ``0``
                       Outlet area of the vaneless diffuser
        _outlet_diameter_old : float, defaults to ``0``
                               Previous outlet diameter of the vaneless diffuser when updated with :func:`~geometry.VanelessDiffuser.update_outlet_diameter`
    """

    def __init__(self):
        """Constructor of VanelessDiffuser class

            Sets ``_outlet_diameter``, ``_outlet_height``, ``_outlet_area`` and ``_outlet_diameter_old`` to ``0``.
        """
        self._outlet_diameter = 0 #D_3
        self._outlet_height = 0 #b3

        self._outlet_area = 0 #A_3
        self._outlet_diameter_old = 0

    def compute_area(self):
        """Computes and updates the outlet area of the vaneless diffuser, stores it in ``_outlet_area``        
        """
        self._outlet_area = np.pi*self._outlet_diameter*self._outlet_height

    def update_outlet_diameter(self, delta):
        """Computes and updates the outlet diameter of the vaneless diffuser, stores it in ``_outlet_diameter``

        The current value of ``_outlet_diameter`` is stored in ``_outlet_diameter_old``.\n
        The new value of ``_outlet_diameter`` is ``_outlet_diameter + delta``.

        Parameters
        ----------
        delta : float
                increase in outlet diameter ``_outlet_diameter`` to add.
        """
        self._outlet_diameter_old = self._outlet_diameter
        self._outlet_diameter = self._outlet_diameter + delta

class VanedDiffuser():

    def __init__(self):
        self._outlet_diameter = 0 #D_4
        self._angle = np.zeros((2)) #inlet, outlet
        self._blade_thickness = 0 #t34
        self._number_blade = 0 #N_vane
        self._pivot_position_chort = 0 #x_c,piv
        self._position_angle = 0 #alpha_vnd
        self._parameters_plot = np.zeros((2))
        self._throat_area = 0
        self._outlet_area = 0

    def compute_parameters_plot(self, D3=0):
        """
        Computes a and b parameters used in the conformal mapping"""    
        self._parameters_plot = np.dot(1./(D3/2 - self._outlet_diameter/2)*np.array([[1, -1], [-self._outlet_diameter/2, D3/2]]), self._angle)

    def compute_psi(self, r=1e-9, c=0):
        return self._parameters_plot[0]*r + self._parameters_plot[1]*np.log(r) + c

    def compute_xy_plot(self, D3=0, nbpoints=2, c=0):
        x = np.array([
            r*np.cos(self.compute_psi(r, c)) 
            for r in np.linspace(start=D3/2., stop=self._outlet_diameter/2, num=nbpoints)])
        y = np.array([
            r*np.sin(self.compute_psi(r, c)) 
            for r in np.linspace(start=D3/2., stop=self._outlet_diameter/2, num=nbpoints)])
        return [x, y]
    
    def compute_pivotpoint(self, xy):
        """
        Rotation center is along the chord at a distance self._pivot_position_chort of the leading edge
        This is computed with a direction vector given by the points
        """
        return np.array([xy[0][0], xy[1][0]]) + self._pivot_position_chort*(np.array([xy[0][-1], xy[1][-1]]) -  np.array([xy[0][0], xy[1][0]]))
    
    def compute_xyblades(self, D3=0, nbpoints=2):
        return np.array([self.compute_xy_plot(D3, nbpoints, 2*np.pi*k/self._number_blade) for k in range(0, self._number_blade)])

    def compute_xypivotpoints(self, xyblades):
        return np.array([self.compute_pivotpoint(xyblades[k]) for k in range(0, self._number_blade)])
    
    def compute_xyrotatedblades(self, xyblades, xypivotpoints, nbpoints):
        rotatedblades = np.array([[ \
            np.dot(
                ([xyblades[k][0][i], xyblades[k][1][i]] - xypivotpoints[k]), 
                np.array([[np.cos(self._position_angle), -np.sin(self._position_angle)], [np.sin(self._position_angle), np.cos(self._position_angle)]])) \
            + xypivotpoints[k] for i in range(0, nbpoints)]
            for k in range(0, self._number_blade) ])
        return np.swapaxes(rotatedblades, 1, 2)
    
    def plotblades(self, D3=0, nbpoints=2, show=True):
        blades = self.compute_xyblades(D3, nbpoints)
        pivotpoints = self.compute_xypivotpoints(blades)
        rotatedblades = self.compute_xyrotatedblades(blades, pivotpoints, nbpoints)
        d, points = self.compute_distancebtwblades(D3)

        fig, ax = plt.subplots()
        for i in range(0, self._number_blade):
            ax.plot(blades[i][0], blades[i][1], color='black')
            ax.plot(rotatedblades[i][0], rotatedblades[i][1], color='magenta')
            ax.plot(pivotpoints[i][0], pivotpoints[i][1], marker='+', color='red')
        ax.plot(points[0], points[1], color='lime')
        plt.gca().set_aspect('equal')
        if show: plt.show()

    def compute_distancebtwblades(self, D3):
        nbpoints = int(round((self._outlet_diameter - D3)/2./0.5e-3)) #1 point per 0.5mm
        blades = self.compute_xyblades(D3=D3, nbpoints=nbpoints)
        pivot = self.compute_xypivotpoints(xyblades=blades)
        rotatedblades = self.compute_xyrotatedblades(xyblades=blades, xypivotpoints=pivot, nbpoints=nbpoints)
        d = sys.float_info.max
        points = [[0, 0],[0, 0]] #points[0] = x of points ; points[1] = y of points
        for i in range(0, nbpoints):
            for j in range(0, nbpoints):
                if i == j:
                    continue
                if d > (rotatedblades[0][0][i] - rotatedblades[1][0][j])**2 + (rotatedblades[0][1][i] - rotatedblades[1][1][j])**2:
                    points[0] = [rotatedblades[0][0][i], rotatedblades[1][0][j]]
                    points[1] = [rotatedblades[0][1][i], rotatedblades[1][1][j]]
                d = min(
                    d, 
                    (rotatedblades[0][0][i] - rotatedblades[1][0][j])**2 + (rotatedblades[0][1][i] - rotatedblades[1][1][j])**2
                )
        return np.sqrt(d)-self._blade_thickness, points

    def compute_areas(self, D3, b3):
        d = self.compute_distancebtwblades(D3)
        self._throat_area = d*b3*self._number_blade
        self._outlet_area = b3*(np.pi*self._outlet_diameter - self._number_blade*self._blade_thickness)

class Volute():
    """Class defining and holding all geometrical data for the volute of a centrifugal machine

        The volute is supposed axi-symmetric with a flat bottom for half of the circle diameter

        Attributes
        ----------
        _D : float, defaults to ``0``
             diameter of the circle defining the volute area
        _area : float, defaults to ``0``
                area of a cross-section of the volute
    """

    def __init__(self):
        """Constructor of VanelessDiffuser class

            Sets ``_D``, and ``_area`` to ``0``.
        """
        self._D = 0
        self._area = 0

    def compute_area_circularsegment(self, h):
        """Computes and returns the area of a circular segment (in the circle) for a line intersecting a circle at a height of ``h`` from the bottom of the circle.

            Maths explained in `Mathworld <https://mathworld.wolfram.com/CircularSegment.html>`_

            Parameters
            ----------
            h : float
                height of the line intersecting the circle
            
            Returns
            -------
            A : float
                area of the circle between the circular segment and the line at height ``h``.
        """
        R = self._D/2
        if self._D - 1e-10 <= h: return R*h
        return R**2*np.arccos((R - h)/R) - (R - h)*np.sqrt(2*R*h - h**2)
    
    def compute_outlet_area(self, diffuser_outlet_height):
        """Computes, updates and returns the area of the volute ``_area`` including the small part that comes from the diffuser intersecting the circle

        Parameters
        ----------
        diffuser_outlet_height : float
                                 diffuser's outlet height ie. height of the line intersecting the circle defining the volute.
        """
        self._area = np.pi*self._D**2/4 + self._D/2*diffuser_outlet_height - 0.5*self.compute_area_circularsegment(diffuser_outlet_height)
        return self._area

    def solve_diameter(self, diffuser_area, diffuser_outlet_height):
        """Computes and updates the volute diameter ``_D`` and area ``_area``.
        
            Using scipy.optimize.brentq on :func:`~geometry.Volute.function_to_optimize`, the diameter ``_D`` is set.
            The area ``_area`` is then updated with the new diameter ``_D`` found.

            Parameters
            ----------
            diffuser_area : float
                            area of the diffuser studied. 
                            Usually ``_area`` but the function could be called with another one because the functions do not specifically depend on this instance.
            diffuser_outlet_height : float
                                     height of the diffuser studied.
                                     Used as the lower bound for the brentq function. ``1000 diffuser_outlet_height`` used as the higher bound of the brentq function.
        """
        self._D = brentq(self.function_to_optimize, 
                         a= diffuser_outlet_height, 
                         b= diffuser_outlet_height*1000,
                         args=(diffuser_outlet_height, diffuser_area))
        self.compute_outlet_area(diffuser_outlet_height)

    def function_to_optimize(self, x, diffuser_outlet_height, obj):
        """Sets the volute diameter ``_D`` with input, updates the area ``_area`` and returns the difference of the new area with input ``obj``

        Parameters
        ----------
        x : float
            new value of volute diameter ``_D``.
        diffuser_outlet_height : float
                                 height of the diffuser studied.
        obj : float
              Usually the previous volute area value.

        Returns
        -------
        diff : float
               arithmetic difference of the new value of the area ``_area`` with the old value given as a parameter by ``obj``.
        """
        self._D = x
        diff = self.compute_outlet_area(diffuser_outlet_height) - obj
        return diff

    def find_offset_diffuser(self, h):
        """Computes and returns the radial position of the intersection of the diffuser with the volute where the diffuser is of height ``h`` given as the parameter.

        The intersection of the diffuser with the volute is defined as a height. 
        Therefore, when doing the intersection with a circle, the radial position of such intersection may be different than initially used in the code.
        This function computes the radial position of such intersection to update it in the code according to the geometric definition of the volute.
        This is then in turns used in :func:`~geometry.VanelessDiffuser.update_outlet_diameter`.

        Parameters
        ----------
        h : float
            Height of the diffuser when it meets the volute

        Returns
        -------
        delta : float
                returns the difference of the radial position of the point where the diffuser intersects the volute to be used in :func:`~geometry.VanelessDiffuser.update_outlet_diameter`
        """
        theta = np.arcsin((h - self._D/2)/(self._D/2))
        return self._D/2*(1 - np.cos(theta))

class Validation():

    def compare_geom_parisi(self, show=False, save=False):
        impeller = Impeller()
        impeller._outlet_diameter = 2*26
        impeller._inlet_diameters = [6*2, 0, 18.5*2]
        impeller._outlet_blade_height = 3.8
        impeller._axial_extension = 17
        impeller._inlet_blade_angle = [np.deg2rad(-30), 0, np.deg2rad(-62)]
        impeller.compute_inlet_average_blade_angle()
        impeller._outlet_blade_angle = np.deg2rad(-35)
        impeller._parameter_angle = 0.33 #1/3.
        impeller._inlet_blade_thickness = [1, 0, 1]
        impeller._outlet_average_blade_thickness = 1
        impeller._number_blade_full = 6
        impeller._splitter_blade_length_fraction = 0.6
        impeller.update_geometry()

        number_xt = 2 + 10 #nombre de lignes tracees
        number_theta = 200
        xt = np.linspace(start= 0, stop=1, num=number_xt)
        if 2 == number_xt:
            xt = np.concatenate((xt, np.array([(impeller._inlet_diameters[1] - impeller._inlet_diameters[0])/(impeller._inlet_diameters[-1] - impeller._inlet_diameters[0])])))
            xt = np.sort(xt, kind= "mergesort")
            number_xt = number_xt + 1
        #print(xt)
        theta = np.linspace(start=0, stop=np.pi/2, num=number_theta)
        Xt, Theta = np.meshgrid(xt, theta)
        #print("Xt = {}".format(Xt))
        #print("Theta = {}".format(Theta))
        #print("Xt.shape = {}".format(Xt.shape))
        r = impeller.compute_radial_coordinate(Xt, Theta)
        z = impeller.compute_axial_coordinate(Xt, Theta)
        #print("r = {}".format(r))
        #print("z = {}".format(z))
        mt, z, r = impeller.compute_curvilinear_abscissa(xt=Xt, theta=Theta, r=r, z=z)
        #print("mt = {}".format(mt))
        beta, mt, z, r = impeller.compute_angle_blade(xt=Xt, theta=Theta, r=r, z=z, mt=mt)
        #print("beta = {}".format(beta))
        phi, beta, mt, z, r = impeller.compute_polar_angle(xt=Xt, theta=Theta, r=r, z=z, mt=mt, beta=beta, adjustthickness=True)
        #print("phi = {}".format(phi))
        mt_adim = impeller.compute_curvilinear_abscissa_adim(mt)
        phi_allblades = impeller.compute_polar_angle_repet(phi)
        # print("phi_allblades = {}".format(phi_allblades))
        impeller.cut_splitters(Xt, mt, phi_allblades)
        #print("phi_allblades = {}".format(phi_allblades))

        #check data from Simone Parisi
        hub_r = [6.026490066, 6.125827815, 6.456953642, 7.052980132, 7.913907285, 9.072847682, 10.56291391, 12.58278146, 15.36423841, 19.86754967, 25.99337748]
        hub_z = [-0.033057851, -2.016528926, -4, -5.983471074, -8.033057851, -10.04958678, -12.03305785, -14.01652893, -16, -17.98347107, -18.90909091]
        mid_r = [13.77483444, 13.8410596, 14.13907285, 14.60264901, 15.29801325, 16.29139073, 17.58278146, 19.47019868, 22.91390728, 26.02649007]
        mid_z = [-8.88E-16, -2.016528926, -4, -5.983471074, -8.033057851, -10.01652893, -12.03305785, -14.01652893, -16, -16.56198347]
        tip_r = [18.54304636, 18.64238411, 18.80794702, 19.13907285, 19.70198675, 20.39735099, 21.49006623, 23.21192053, 26.02649007]
        tip_z = [-0.033057851, -2.016528926, -4, -6.016528926, -8.033057851, -10.01652893, -12.03305785, -14.01652893, -15.10743802]
        hub_m = [0, 0.108669109, 0.2002442, 0.289377289, 0.37973138, 0.449328449, 0.507936508, 0.561660562, 0.623931624, 0.686202686, 0.750915751, 0.807081807, 0.858363858, 0.923076923, 0.998778999]
        hub_beta = [-0.505255023, -0.379134467, -0.324729521, -0.307418856, -0.319783617, -0.345749614, -0.375425039, -0.405100464, -0.445904173, -0.485471406, -0.523802164, -0.554714065, -0.579443586, -0.601700155, -0.611591963]
        tip_m = [0 ,0.098901099 ,0.2002442 ,0.296703297 ,0.376068376 ,0.43956044 ,0.500610501 ,0.555555556 ,0.617826618 ,0.664224664 ,0.717948718 ,0.774114774 ,0.825396825 ,0.876678877 ,0.94017094 ,1]
        tip_beta = [-1.098763524 ,-1.086398764 ,-1.048068006 ,-0.996136012 ,-0.942967543 ,-0.898454405 ,-0.855177743 ,-0.81561051 ,-0.769860896 ,-0.737712519 ,-0.705564142 ,-0.674652241 ,-0.651159196 ,-0.63137558 ,-0.615301391 ,-0.610355487]

        fig2, ax20 = plt.subplots()
        gs = gridspec.GridSpec(3,2, hspace=0.5)
        ax20.set_position(gs[0:2, 0].get_position(fig2))
        ax20.plot(r, z, linewidth=1, marker='+')
        ax20.plot(hub_r, hub_z, linewidth=0, marker='D', color="blue", markersize=5, fillstyle="none")
        ax20.plot(mid_r, mid_z, linewidth=0, marker='D', color="black", markersize=5, fillstyle="none")
        ax20.plot(tip_r, tip_z, linewidth=0, marker='D', color="orange", markersize=5, fillstyle="none")
        ax20.set_xlabel("r [mm]") ; ax20.set_ylabel('z [mm]')
        ax20.set_xlim(4, 26)
        ax20.set_ylim(-20, 0)
        #ax20.legend()
        ax20.grid(axis='both')

        ax25 = fig2.add_subplot()
        ax25.set_position(gs[:, -1].get_position(fig2))
        for i in range(0, number_xt):
            if 0 == i or (number_xt - 1) == i:
                labels = ["hub", "tip"]
                j = i if i == 0 else -1
                ax25.plot(mt_adim[:, i], beta[:, i], linewidth=1, label=labels[j], marker='+')
            else:        
                ax25.plot(mt_adim[:, i], beta[:, i], linewidth=1)
        ax25.plot(hub_m, hub_beta, linewidth=0, marker='D', color="blue", markersize=5, fillstyle="none")
        ax25.plot(tip_m, tip_beta, linewidth=0, marker='D', color="orange", markersize=5, fillstyle="none")
        ax25.set(xlim=(0, 1), ylim=(-1.1, 0))
        ax25.legend()
        ax25.set_xlabel("m (adim curvilinear abscisse)") ; ax25.set_ylabel(r'$\beta$ [deg]')
        ax25.grid(axis='both')
        my_col = cm.viridis(mt_adim)
        """X, Y = r*np.cos(phi), r*np.sin(phi)
        ax30 = fig2.add_subplot(projection='3d')
        ax30.set_position(gs[0:2, 1].get_position(fig2))
        my_col = cm.viridis(mt_adim)
        ax30.plot_surface(X, Y, z, facecolors=my_col, alpha=0.6)
        ax30.plot(X[:, 0], Y[:, 0], z[:, 0], linewidth=3, color="cyan", label="hub K = {}".format(impeller._parameter_angle))
        ax30.plot(X[:, -1], Y[:, -1], z[:, -1], linewidth=3, color="orange", label="tip K = {}".format(impeller._parameter_angle))
        ax30.legend()
        ax30.set_title("3D view") ; ax30.set_xlabel("x [mm]") ; ax30.set_ylabel('y [mm]') ; ax30.set_zlabel('z [mm]')
        ax30.set(xlim=(-impeller._outlet_diameter/2, impeller._outlet_diameter/2), ylim=(-impeller._outlet_diameter/2, impeller._outlet_diameter/2))"""

        """------------------------------------------------------------------------------"""
        fig, ax = plt.subplots()
        ax = fig.add_subplot(projection='3d')
        for i, val in enumerate(phi_allblades):
            X, Y = r*np.cos(val), r*np.sin(val)
            indexes = np.argwhere(np.abs(val) > 1e-15)
            ax.plot_surface(X[indexes[0, 0]:indexes[-1, 0] + 1, :], Y[indexes[0, 0]:indexes[-1, 0] + 1, :], z[indexes[0, 0]:indexes[-1, 0] + 1, :], facecolors=my_col, alpha=1)

        ax.set(xlim=(-impeller._outlet_diameter/2, impeller._outlet_diameter/2), ylim=(-impeller._outlet_diameter/2, impeller._outlet_diameter/2))
        ax.set_aspect('equal')
        ax.set_title("Impeller 3D view") ; ax.set_xlabel("x [mm]") ; ax.set_ylabel('y [mm]') ; ax.set_zlabel('z [mm]')

        if save:
            filename = str(time.time()) + ".txt"
            with open(filename, 'a') as file:
                file.write('X = \n')
                np.savetxt(file, X, delimiter=' ', newline='\n')
                file.write('Y = \n')
                np.savetxt(file, Y, delimiter=' ', newline='\n')
                file.write('Z = \n')
                np.savetxt(file, z, delimiter=' ', newline='\n')

        if show:
            plt.show()


if __name__ == "__main__":
    #validation = Validation()
    #validation.compare_geom_parisi(show=True, save=False)
    
    impeller = Impeller()
    impeller._outlet_diameter = 400
    impeller._inlet_diameters = [90, 0, 280]
    impeller._outlet_blade_height = 26
    impeller._axial_extension = 130
    impeller._inlet_blade_angle = [np.deg2rad(-32), 0, np.deg2rad(-63)]
    impeller.compute_inlet_average_blade_angle()
    impeller._outlet_blade_angle = np.deg2rad(0)
    impeller._parameter_angle = 0.55 #1/3.
    impeller._inlet_blade_thickness = [1, 0, 1]
    impeller._outlet_average_blade_thickness = 1
    impeller._number_blade_full = 20
    impeller._splitter_blade_length_fraction = 0
    impeller.update_geometry()

    number_xt = 2 + 0 #nombre de lignes tracees
    number_theta = 200
    xt = np.linspace(start= 0, stop=1, num=number_xt)
    if 2 == number_xt:
        xt = np.concatenate((xt, np.array([np.sqrt(1/2.)])))
        xt = np.sort(xt, kind= "mergesort")
        number_xt = number_xt + 1
    #print(xt)
    theta = np.linspace(start=0, stop=np.pi/2, num=number_theta)
    Xt, Theta = np.meshgrid(xt, theta)
    #print("Xt = {}".format(Xt))
    #print("Theta = {}".format(Theta))
    #print("Xt.shape = {}".format(Xt.shape))
    r = impeller.compute_radial_coordinate(Xt, Theta)
    z = impeller.compute_axial_coordinate(Xt, Theta)
    #print("r = {}".format(r))
    #print("z = {}".format(z))
    mt, z, r = impeller.compute_curvilinear_abscissa(xt=Xt, theta=Theta, r=r, z=z)
    #print("mt = {}".format(mt))
    beta, mt, z, r = impeller.compute_angle_blade(xt=Xt, theta=Theta, r=r, z=z, mt=mt)
    #print("beta = {}".format(beta))
    phi, beta, mt, z, r = impeller.compute_polar_angle(xt=Xt, theta=Theta, r=r, z=z, mt=mt, beta=beta, adjustthickness=False)
    #print("phi = {}".format(phi))
    phi_allblades = impeller.compute_polar_angle_repet(phi)
    # print("phi_allblades = {}".format(phi_allblades))
    impeller.cut_splitters(Xt, mt, phi_allblades)
    #print("phi_allblades = {}".format(phi_allblades))
    mt_adim = impeller.compute_curvilinear_abscissa_adim(mt)

    fig2, ax20 = plt.subplots()
    gs = gridspec.GridSpec(3,2, hspace=0.5)
    ax20.set_position(gs[0:1, 0].get_position(fig2))
    ax20.plot(r, z, linewidth=1, marker='+')
    ax20.set_xlabel("r [mm]") ; ax20.set_ylabel('z [mm]')
    #ax20.set_xlim(4, 26)
    #ax20.set_ylim(-20, 0)
    #ax20.legend()
    ax20.grid(axis='both')


    ax30 = fig2.add_subplot()
    ax30.set_position(gs[1:, 0].get_position(fig2))
    ax30.grid(axis='both')
    ax30.set_xlabel("mt_adim [mm]") ; ax30.set_ylabel('Phi [°]')
    for i in range(0, r.shape[-1]):
        ax30.plot(mt_adim[:, i], phi[:, i], linewidth=1, marker='+')




    ax25 = fig2.add_subplot()
    ax25.set_position(gs[:, -1].get_position(fig2))
    for i in range(0, number_xt):
        if 0 == i or (number_xt - 1) == i:
            labels = ["hub", "tip"]
            j = i if i == 0 else -1
            ax25.plot(mt_adim[:, i], beta[:, i], linewidth=1, label=labels[j], marker='+')
        else:        
            ax25.plot(mt_adim[:, i], beta[:, i], linewidth=1)
    ax25.set(xlim=(0, 1), ylim=(-1.1, 0))
    ax25.legend()
    ax25.set_xlabel("m (adim curvilinear abscisse)") ; ax25.set_ylabel(r'$\beta$ [deg]')
    ax25.grid(axis='both')
    my_col = cm.viridis(mt_adim)

    fig, ax = plt.subplots()
    ax = fig.add_subplot(projection='3d')
    for i, val in enumerate(phi_allblades):
        X, Y = r*np.cos(val), r*np.sin(val)
        indexes = np.argwhere(np.abs(val) > 1e-15)
        ax.plot_surface(X[indexes[0, 0]:indexes[-1, 0] + 1, :], Y[indexes[0, 0]:indexes[-1, 0] + 1, :], z[indexes[0, 0]:indexes[-1, 0] + 1, :], facecolors=my_col, alpha=1)

    ax.set(xlim=(-impeller._outlet_diameter/2, impeller._outlet_diameter/2), ylim=(-impeller._outlet_diameter/2, impeller._outlet_diameter/2))
    ax.set_aspect('equal')
    ax.set_title("Impeller 3D view") ; ax.set_xlabel("x [mm]") ; ax.set_ylabel('y [mm]') ; ax.set_zlabel('z [mm]')

    plt.show()