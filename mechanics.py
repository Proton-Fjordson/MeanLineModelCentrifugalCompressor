import numpy as np

class DiscStress():
    """
        Computes disc stress

        Attributes
        ----------
        _ksigma : float
                experimental coefficient increasing the stress\n
                usually between 0.75 and 1. > 1 to be conservative.\n
                Default is 1.2
    """

    _ksigma = 1.2

    def compute_stress(self, density, impeller_speed, poisson_ratio):
        """
            Computes the disc stress

            Osbourne disc stress

            Parameters
            ----------
            density : float
                Impeller's materials' density
            impeller_speed : float
                Impeller's peripheral tangential velocity U2
            poisson_ratio : float
                Impeller's materials' poisson ratio

            Returns
            -------
            float
                Maximal disc stress according to Osborn
        """
        return self._ksigma*density*impeller_speed**2*(3+poisson_ratio)/8

class BladeStress():
    """
    Computes blade stress according to Osborne et al.
    """

    def compute_f(self, x, taper_ratio, type_taper="parabolic"):
        """
            Compute the normalized stress factor

            C. Osborne, P. Runstadler Jr, and W. D. Stacy, “Aerodynamic and mechanical design of an 8: 1 pressure ratio centrifugal compressor,” Creare Incorporated, 1974.

            x : float
                Normalized distance along the blade from tip to root (0 to 1).
            taper_ratio : float
                Blade thickness at hub divided by blade thickness at tip t_hub/t_tip.
            type_taper : string
                Taper interpolation type.\n
                linear or parabolic

            Returns
            -------
            float
                Normalized stress factor

            Raises
            ------
            NotImplementedError
                if type is not linear or parabolic, raises the implementation error
        """

        k = taper_ratio - 1

        if "linear" == type_taper:
            t = 1 + k*x
            return (1/t)**2*(x - 0.5*x**2 + 1/6*k*x**3)
        elif "parabolic" == type_taper:
            t = 1 + k*x**2
            return (1/t)**2*(x - 0.5*x**2 + 1/12*k*x**4)
        else:
            raise NotImplementedError("The type_taper should be linear or parabolic")

    def compute_stress(self, blade_height, density, outlet_diameter, rot_speed, outlet_blade_angle, ttip, taper_ratio, type_taper="parabolic"):
        """
            Compute the stress

            C. Osborne, P. Runstadler Jr, and W. D. Stacy, “Aerodynamic and mechanical design of an 8: 1 pressure ratio centrifugal compressor,” Creare Incorporated, 1974.

            blade_height : float
                Impeller's outlet blade height.
            density : float
                Impeller's materials' density.
            outlet_diameter : float
                Impeller's outlet_diameter.
            rot_speed : float
                Impeller's rotation speed in rad/s.
            outlet_blade_angle : float
                Impeller's blades' outlet angle in radians.
            ttip : float
                Impeller's blades' thickness at the tip (opposed to the hub)
            taper_ratio : float
                Blade thickness at hub divided by blade thickness at tip t_hub/t_tip.
            type_taper : string
                Taper interpolation type.\n
                linear or parabolic

            Returns
            -------
            float
                Stress at the root of the blades
        """
        return np.abs(self.compute_f(1, taper_ratio, type_taper) * blade_height**2 * 6 * density * outlet_diameter/2 * rot_speed**2*np.sin(outlet_blade_angle)/ttip)

class Materials():
    """
    """
    def __init__(self, name= '316L', Youngmodulus= 200e6, poisson= 0.33, yieldstress= 179e6, density= 7916):
        self._name = name
        self._youngmodulus = Youngmodulus
        self._poisson = poisson
        self._yieldstress = yieldstress
        self._density = density

    def get_MOS(self, stress=1):
        """
            If we are at the yieldstress = 0
            If we are lower than the yieldstress > 0
            If we are bigger than the yieldstress < 0
        """
        return self._yieldstress/stress - 1


if __name__ == "__main__":
    #Data from source document Osborne et al.
    r = 7.976e-2 #m
    N = 75000 #rpm
    beta = -30 #°
    h = 3.302e-3 #m
    ttip = 1.27e-3 #m
    k = 1

    density = 4590 #kg/m^3 

    st = BladeStress()
    sigma = st.compute_stress(
        blade_height= h,
        density= density,
        outlet_diameter= 2*r,
        rot_speed= N*2*np.pi/60,
        outlet_blade_angle= np.deg2rad(beta),
        ttip= ttip,
        taper_ratio= k + 1,
        type_taper= "parabolic"
        )
    print(sigma)

    #-----------

    print("-------------")
    discstress = DiscStress()
    discstress._ksigma = 1.5

    speedmax = 250000*2*np.pi/60 #rad/s
    outlet_diameter = 3.44e-2 #m
    u2 = outlet_diameter/2*speedmax

    densities = [2700, 4410, 7900] #al, ti, ss
    poissons = [0.33, 0.31, 0.33] #al, ti, ss
    yieldstress = [440, 1220, 380] #al, ti, ss

    stress = np.zeros(len(densities))

    for i, el in enumerate(densities):
        stress[i] = discstress.compute_stress(densities[i], u2, poissons[i])*1e-6

    print("ksigma = {}".format(discstress._ksigma))
    print("Stresses = {}".format(stress))
    print("MOS = {}".format(np.array(yieldstress)/stress - 1))
    

    r = outlet_diameter/2 #m
    N = 250000 #rpm
    beta = -30 #°
    h = 2e-3 #m
    ttip = 1e-3 #m
    k = 1

    density = 2700 #kg/m^3 

    st = BladeStress()
    sigma = st.compute_stress(
        blade_height= h,
        density= density,
        outlet_diameter= 2*r,
        rot_speed= N*2*np.pi/60,
        outlet_blade_angle= np.deg2rad(beta),
        ttip= ttip,
        taper_ratio= k + 1,
        type_taper= "parabolic"
        )
    print(sigma*1e-6)