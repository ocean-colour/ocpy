import numpy as np 

from ocpy.water import absorption
from ocpy.hydrolight import loisel23

from IPython import embed

def betasw_ZHH2009(lambda_,Tc,theta,S,delta=0.039):
    """
    % Xiaodong Zhang, Lianbo Hu, and Ming-Xia He (2009), Scatteirng by pure
    % seawater: Effect of salinity, Optics Express, Vol. 17, No. 7, 5698-5710
    %
    % Xiaodong Zhang, March 10, 2009

    Args:
        lambda_ (np.ndarray): wavelength (nm)
        Tc (float): temperauter in degree Celsius, must be a scalar
        theta (np.ndarray): angle in degrees, must be a vector
        S (float): salinity, must be scalar
        delta (float, optional): depolarization ratio, if not provided, default = 0.039 will be

    Raises:
        ValueError: _description_

    Returns:
        tuple: (betasw, beta90sw, bw)
            % betasw: volume scattering at angles defined by theta. Its size is [x y],
            % where x is the number of angles (x = length(theta)) and y is the number
            % of wavelengths in lambda (y = length(lambda))
            % beta90sw: volume scattering at 90 degree. Its size is [1 y]
            % bw: total scattering coefficient. Its size is [y]
            % for backscattering coefficients, divide total scattering by 2
            %
        
    """
    raise ValueError("THIS IS NOT SUCCESFULLY CONVERTED YET")

    # values of the constants
    Na = 6.0221417930e23  # Avogadros constant
    Kbz = 1.3806503e-22  # Boltzmann constant
    Tk = Tc + 273.15  # Absolute tempearture
    M0 = 18e-3  # Molecular weigth of water in kg/mol

    if not np.isscalar(Tc) or not np.isscalar(S):
        raise ValueError('Both Tc and S need to be scalar variables')

    lambda_ = np.array(lambda_)  # a column variable
    rad = np.deg2rad(theta)  # angle in radian as a column variable

    # nsw: absolute refractive index of seawater
    # dnds: partial derivative of seawater refractive index w.r.t. salinity
    nsw, dnds = RInw(lambda_, Tc, S)

    # isothermal compressibility is from Lepple & Millero (1971,Deep Sea-Research), pages 10-11
    # The error ~ +/-0.004e-6 bar^-1
    IsoComp = BetaT(Tc, S)

    # density of water and seawater,unit is Kg/m^3, from UNESCO,38,1981
    density_sw = rhou_sw(Tc, S)

    # water activity data of seawater is from Millero and Leung (1976,American Journal of Science,276,1035-1077).
    # Table 19 was reproduced using Eq.(14,22,23,88,107) then were fitted to polynominal equation.
    # dlnawds is partial derivative of natural logarithm of water activity w.r.t.salinity
    dlnawds = dlnasw_ds(Tc, S)

    # density derivative of refractive index from PMH model
    DFRI = PMH(nsw)  # PMH model

    # volume scattering at 90 degree due to the density fluctuation
    beta_df = np.pi ** 2 / 2 * ((lambda_ * 1e-9) ** (-4)) * Kbz * Tk * IsoComp * DFRI ** 2 * (6 + 6 * delta) / (
                6 - 7 * delta)

    # volume scattering at 90 degree due to the concentration fluctuation
    flu_con = S * M0 * dnds ** 2 / density_sw / (-dlnawds) / Na
    beta_cf = 2 * np.pi ** 2 * ((lambda_ * 1e-9) ** (-4)) * nsw ** 2 * (flu_con) * (6 + 6 * delta) / (
                6 - 7 * delta)

    # total volume scattering at 90 degree
    beta90sw = beta_df + beta_cf
    bsw = 8 * np.pi / 3 * beta90sw * (2 + delta) / (1 + delta)

    betasw = np.zeros((len(theta), len(lambda_)))
    for i in range(len(lambda_)):
        betasw[:, i] = beta90sw[i] * (1 + ((np.cos(rad)) ** 2) * (1 - delta) / (1 + delta))

    return betasw, beta90sw, bsw

def RInw(lambda_, Tc, S):
    """ refractive index of seawater

    Args:
        lambda_ (np.ndarray): wavelength (nm)
        Tc (float): temperauter in degree Celsius, must be a scalar
        S (float): salinity, must be scalar

    Returns:
        tuple: (nsw, dnswds)
            nsw: absolute refractive index of seawater
            dnswds: partial derivative of seawater refractive index w.r.t. salinity
    """
    # refractive index of air is from Ciddor (1996,Applied Optics)
    n_air = 1.0 + (5792105.0 / (238.0185 - 1 / (lambda_ / 1e3) ** 2) + 167917.0 / (57.362 - 1 / (lambda_ / 1e3) ** 2)) / 1e8

    # refractive index of seawater is from Quan and Fry (1994, Applied Optics)
    n0 = 1.31405
    n1 = 1.779e-4
    n2 = -1.05e-6
    n3 = 1.6e-8
    n4 = -2.02e-6
    n5 = 15.868
    n6 = 0.01155
    n7 = -0.00423
    n8 = -4382
    n9 = 1.1455e6

    nsw = n0 + (n1 + n2 * Tc + n3 * Tc ** 2) * S + n4 * Tc ** 2 + (n5 + n6 * S + n7 * Tc) / lambda_ + n8 / lambda_ ** 2 + n9 / lambda_ ** 3  # pure seawater
    nsw = nsw * n_air
    dnswds = (n1 + n2 * Tc + n3 * Tc ** 2 + n6 / lambda_) * n_air

    return nsw, dnswds

def BetaT(Tc, S):
    """ isothermal compressibility of seawater

    Args:
        Tc (float): temperauter in degree Celsius, must be a scalar
        S (float): salinity, must be scalar

    Returns:
        float: isothermal compressibility of seawater
    """
    # pure water secant bulk Millero (1980, Deep-sea Research)
    kw = 19652.21 + 148.4206 * Tc - 2.327105 * Tc ** 2 + 1.360477e-2 * Tc ** 3 - 5.155288e-5 * Tc ** 4
    Btw_cal = 1. / kw

    # seawater secant bulk
    a0 = 54.6746 - 0.603459 * Tc + 1.09987e-2 * Tc ** 2 - 6.167e-5 * Tc ** 3
    b0 = 7.944e-2 + 1.6483e-2 * Tc - 5.3009e-4 * Tc ** 2
    Ks = kw + a0 * S + b0 * S ** 1.5

    # calculate seawater isothermal compressibility from the secant bulk
    IsoComp = 1. / Ks * 1e-5  # unit is pa

    return IsoComp

def rhou_sw(Tc, S):
    """ density of seawater

    Args:
        Tc (float): temperauter in degree Celsius, must be a scalar
        S (float): salinity, must be scalar

    Returns:
        float: density of seawater
    """
    # density of water and seawater,unit is Kg/m^3, from UNESCO,38,1981
    a0 = 8.24493e-1
    a1 = -4.0899e-3
    a2 = 7.6438e-5
    a3 = -8.2467e-7
    a4 = 5.3875e-9
    a5 = -5.72466e-3
    a6 = 1.0227e-4
    a7 = -1.6546e-6
    a8 = 4.8314e-4
    b0 = 999.842594
    b1 = 6.793952e-2
    b2 = -9.09529e-3
    b3 = 1.001685e-4
    b4 = -1.120083e-6
    b5 = 6.536332e-9

    # density for pure water
    density_w = b0 + b1 * Tc + b2 * Tc ** 2 + b3 * Tc ** 3 + b4 * Tc ** 4 + b5 * Tc ** 5

    # density for pure seawater
    density_sw = density_w + (
                (a0 + a1 * Tc + a2 * Tc ** 2 + a3 * Tc ** 3 + a4 * Tc ** 4) * S + (a5 + a6 * Tc + a7 * Tc ** 2) * S ** 1.5 + a8 * S ** 2)

    return density_sw

def dlnasw_ds(Tc, S):
    """ partial derivative of natural logarithm of water activity w.r.t.salinity

    Args:
        Tc (float): temperauter in degree Celsius, must be a scalar
        S (float): salinity, must be scalar

    Returns:
        float: dlnasw_ds is partial derivative of natural logarithm of water activity w.r.t.salinity
    """
    # water activity data of seawater is from Millero and Leung (1976,American Journal of Science,276,1035-1077).
    # Table 19 was reproduced using Eqs.(14,22,23,88,107) then were fitted to polynominal equation.
    # dlnawds is partial derivative of natural logarithm of water activity w.r.t.salinity
    dlnawds = (-5.58651e-4 + 2.40452e-7 * Tc - 3.12165e-9 * Tc ** 2 + 2.40808e11 * Tc ** 3) + \
              1.5 * (1.79613e-5 - 9.9422e-8 * Tc + 2.08919e-9 * Tc ** 2 - 1.39872e11 * Tc ** 3) * S ** 0.5 + \
              2 * (-2.31065e-6 - 1.37674e-9 * Tc - 1.93316e-11 * Tc ** 2) * S

    return dlnawds

def PMH(n_wat):
    """ density derivative of refractive index from PMH model

    Args:
        n_wat (float): refractive index of water

    Returns:
        float: density derivative of refractive index from PMH model
    """
    n_wat2 = n_wat ** 2
    n_density_derivative = (n_wat2 - 1) * (1 + 2 / 3 * (
        n_wat2 + 2) * (n_wat / 3 - 1 / 3 / n_wat) ** 2)

    return n_density_derivative

def bbw_from_l23(wave):

    # TODO -- FIX THIS!
    # THIS IS A HACK UNTIL I CAN RESOLVE bbw
    ds = loisel23.load_ds(4,0)
    l23_wave = ds.Lambda.data
    idx = 170 # Random choie
    l23_bb = ds.bb.data[idx] 
    l23_bbnw = ds.bbnw.data[idx] 
    l23_bbw = l23_bb - l23_bbnw
    # Interpolate
    bb_w = np.interp(wave, l23_wave, l23_bbw)

    return bb_w

if __name__ == '__main__':
    # Example usage
    lambda_ = [412, 440, 488, 510, 532, 555, 650, 676, 715]
    Tc = 20
    theta = [0, 30, 60, 90]
    S = 35

    betasw, beta90sw, bsw = betasw_ZHH2009(lambda_, Tc, theta, S)
    print(betasw)
    print(beta90sw)
    print(bsw)

