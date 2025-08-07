#!/usr/bin/env python
"""
THIS PROGRAM IS EXPERIMENTAL AND IS PROVIDED "AS IS" WITHOUT
REPRESENTATION OF WARRANTY OF ANY KIND, EITHER EXPRESS OR
IMPLIED. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF
THE PROGRAM IS WITH THE USER.
COPYRIGHT (C) 2010 BY GKSS/KOF

2010-03-22 schoenfeld

2010/11: bug fixes in read_purewater_abs, -999 schoenfeld
2013/05: update for second version of pure water absorption data (<420 nm) with data of Morel et al. 2007
2016/11: update for third version of pure water absorption (<510 nm) with data of Mason et al. 2016
"""
import os
import numpy as np  #import library
import sys
from importlib.resources import files

#functions to compute the IOPs
#########################################################
def read_refri_std(fn,  aw,  ew):
#read the standard spectrum of refraction index
#from file 'fn' 
#from wavelength aw to wavelength ew
#return vector of wavelength, vector of refractive index
    fp=open(fn)
    z=fp.readlines()
    fp.close()
    wl=[]
    refri=[]
    for zz in z:
        if zz.startswith('%'): continue
        if zz:
            zs=zz.split()
            if float(zs[0])<aw: continue
            if float(zs[0])>ew: break
            wl.append(float(zs[0]))
            refri.append(float(zs[1]))
    if ew>700:        
        idx=wl.index(700) # index of 700nm  
    else:
        idx=wl.index(ew)
    refri_err=np.zeros(len(wl))
    refri_err[:idx]=0.00003 #Austin&Halikas, 1976
    refri_err[idx:]=0.0005
#    print 'wl', wl,  refri
    return np.array(wl),  np.array(refri),  refri_err
    
def read_purewater_abs(fn,  aw,  ew):
#read the standard absorptionspectrum of pure water
#from file 'fn' 
#from wavelength aw to wavelength ew
#return vectors of wavelength, absorption,T_coeff, S_coeff , absorption_err, T_err, S_err  
    fp=open(fn)
    z=fp.readlines()
    fp.close()
   
    wl=np.zeros(len(z))
    a0=np.zeros(len(z))
    T_coeff=np.zeros(len(z))
    S_coeff=np.zeros(len(z))
    a0_err=np.zeros(len(z))
    T_err=np.zeros(len(z))
    S_err=np.zeros(len(z))
    i=0
    for zz in z:
        if zz.startswith('%'): continue
        zs=zz.split()
        wl[i]=float(zs[0])
        if wl[i]<aw: continue
        if wl[i]>ew: break
        a0[i]=float(zs[1])
        S_coeff[i]=float(zs[2])
        T_coeff[i]=float(zs[3])
        a0_err[i]=float(zs[4])
        S_err[i]=float(zs[5])
        T_err[i]=float(zs[6])
        i+=1
    return wl[:i],  a0[:i], S_coeff[:i], T_coeff[:i] , a0_err[:i], S_err[:i], T_err[:i ]  
    
def absorption(aw, ew,  S,  Tc,  fn:str=None):
    # from wavelength aw to wavelength ew in nm
    # S - salinity (PSU)
    # T - temperature (Celsius)
    # fn - 
    # 

    #
    # absorption 300 - 4000 nm
    # Temp coeff 300 - 4000 nm
    # Sal coeff 300 - 4000 nm
    # Returns
    # wl - wavelength 
    # abso - absorption
    # a_err -- Error 

    # Reference values
    T_ref=20.
    S_ref=0.

    # File
    if fn is None:
        fn = os.path.join(files('ocpy'), 'water', 'WOPP', 'purewater_abs_coefficients_v3.dat')

    # read the standard spectrum of refractive index
    wl, abso_ref, S_coeff, T_coeff, a0_err, S_err, T_err=read_purewater_abs(fn, aw, ew)
    # calculate 
    abso=abso_ref+(Tc-T_ref)*T_coeff  +(S-S_ref)*S_coeff
    a_err=a0_err+(Tc-T_ref)*T_err  +(S-S_ref)*S_err

    # Return
    return wl,  abso,  a_err
    
def scattering(wl, S, Tc,  theta,  std_RI):
    # Xiaodong Zhang, Lianbo Hu, and Ming-Xia He (2009), Scattering by pure
    # seawater: Effect of salinity, Optics Express, Vol. 17, No. 7, 5698-5710 
    #
    # wl (nm): wavelength
    # Tc: temperauter in degree Celsius, must be a scalar
    # S: salinity, must be scalar
    # delta: depolarization ratio, if not provided, default = 0.039 will be
    # used.
    # betasw: volume scattering at angles defined by theta. Its size is [x y],
    # where x is the number of angles (x = length(theta)) and y is the number
    # of wavelengths in wl (y = length(wl))
    # beta90sw: volume scattering at 90 degree. Its size is [1 y]
    # bw: total scattering coefficient. Its size is [1 y]
    # for backscattering coefficients, divide total scattering by 2
    #
    # Xiaodong Zhang, March 10, 2009
    delta=0.039
    # values of the constants
    Na = 6.0221417930e23     #  Avogadro's constant
    Kbz = 1.3806503e-23      #  Boltzmann constant
    Tk = Tc+273.15           #  Absolute tempearture
    M0 = 18e-3               #  Molecular weigth of water in kg/mol
 
    pi=np.pi
    rad = theta*pi/180  # angle in radian as a colum variable
    
    # nsw: absolute refractive index of seawater
    # dnds: partial derivative of seawater refractive index w.r.t. salinity
    ##nsw, dnds = RInw(wl,Tc,S,  alt_RI) 
    nsw, dnds = refractive_index(wl, S, Tc,  std_RI)
    
#    print 'nsw',  nsw, wl
    # isothermal compressibility is from Lepple & Millero (1971,Deep
    # Sea-Research), pages 10-11
    # The error ~ +/-0.004e-6 bar**-1
    IsoComp = BetaT(Tc,S) 
    
    # density of water and seawater,unit is Kg/m**3, from UNESCO,38,1981
    density_sw = rhou_sw(Tc, S) 
    ##print 'density_sw', density_sw
    # water activity data of seawater is from Millero and Leung (1976,American
    # Journal of Science,276,1035-1077). Table 19 was reproduced using
    # Eq.(14,22,23,88,107) then were fitted to polynominal equation.
    # dlnawds is partial derivative of natural logarithm of water activity
    # w.r.t.salinity
    dlnawds = dlnasw_ds(Tc, S) 
    ##print 'dlnawds', dlnawds
    # density derivative of refractive index from PMH model
    DFRI = PMH(nsw)   ## PMH model
    #print 'DFRI', DFRI.shape, wl.shape
    # volume scattering at 90 degree due to the density fluctuation
    beta_df = pi*pi/2*((wl*1e-9)**(-4))*Kbz*Tk*IsoComp*DFRI**2*(6+6*delta)/(6-7*delta) 
    ##print 'beta_df',  beta_df
    # volume scattering at 90 degree due to the concentration fluctuation
    flu_con = S*M0*dnds**2/density_sw/(-dlnawds)/Na 
    ##print 'flu_con',  flu_con
    beta_cf = 2*pi*pi*((wl*1e-9)**(-4))*nsw**2*(flu_con)*(6+6*delta)/(6-7*delta) 
    ##print 'beta_cf',  beta_cf
    # total volume scattering at 90 degree
    beta90sw = beta_df+beta_cf 
    ##print 'beta90sw',  beta90sw
    bsw=8*pi/3*beta90sw*(2+delta)/(1+delta) 
    ##print 'bsw', bsw
    betasw=np.zeros((len(rad), len(wl)))
    for i in range(len(wl)):
        betasw[:, i]=beta90sw[i]*(1+((np.cos(rad))**2)*(1-delta)/(1+delta))
    betasw=betasw.transpose()    
    err_betasw=betasw*0.02
    err_beta90sw=beta90sw*0.02
    err_bsw=bsw*0.02
    return betasw, beta90sw, bsw,  err_betasw, err_beta90sw,  err_bsw
    
def refractive_index(wl, S, Tc,  std_RI):
    # real part of R_efractive In_dex of Sea_w_ater
    # Refractive Index of air is from Ciddor (1996,Applied Optics)
    n_air = 1.0+(5792105.0/(238.0185-1/(wl/1e3)**2)+167917.0/(57.362-1/(wl/1e3)**2))/1e8 
#    print 'n_air',  n_air 
    # refractive index of seawater is from Quan and Fry (1994, Applied Optics)
    n0 = 1.31405  
    n1 = 1.779e-4   
    n2 = -1.05e-6   
    n3 = 1.6e-8   
    n4 = -2.02e-6  
    n5 = 15.868  
    n6 = 0.01155   
    n7 = -0.00423   
    n8 = -4382.   
    n9 = 1.1455e6 
#    print Tc,  S,  wl
    nsw = n0+(n1+n2*Tc+n3*Tc**2)*S+n4*Tc**2+(n5+n6*S+n7*Tc)/wl+n8/wl**2+n9/wl**3  # pure seawater
    nsw = nsw*n_air 
#    print 'nsw',  nsw
    dnswds = (n1+n2*Tc+n3*Tc**2+n6/wl)*n_air 
    try:
        idx=wl.tolist().index(800) # index of 800nm
        offs=nsw[idx]-std_RI[idx] 
        nsw[idx:]=std_RI[idx:]+offs
    except:
#        idx=len(wl)-1 # index of 800nm
#        print 'idx', idx, len(wl), len(nsw), len(std_RI)
        pass
    return nsw,  dnswds
    
def BetaT(Tc, S):
    # pure water secant bulk Millero (1980, Deep-sea Research)
    kw = 19652.21+148.4206*Tc-2.327105*Tc**2+1.360477e-2*Tc**3-5.155288e-5*Tc**4 
    Btw_cal = 1/kw 
    # isothermal compressibility from Kell sound measurement in pure water
    # Btw = (50.88630+0.717582*Tc+0.7819867e-3*Tc**2+31.62214e-6*Tc**3-0.1323594e-6*Tc**4+0.634575e-9*Tc**5)/(1+21.65928e-3*Tc)*1e-6 
    # seawater secant bulk
    a0 = 54.6746-0.603459*Tc+1.09987e-2*Tc**2-6.167e-5*Tc**3 
    b0 = 7.944e-2+1.6483e-2*Tc-5.3009e-4*Tc**2 
    Ks =kw + a0*S + b0*S**1.5 
    # calculate seawater isothermal compressibility from the secant bulk
    IsoComp = 1/Ks*1e-5  # unit is pa
    return IsoComp    
    
def rhou_sw(Tc, S):
    # density of water and seawater,unit is Kg/m**3, from UNESCO,38,1981
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
    density_w = b0+b1*Tc+b2*Tc**2+b3*Tc**3+b4*Tc**4+b5*Tc**5 
    # density for pure seawater
    density_sw = density_w +((a0+a1*Tc+a2*Tc**2+a3*Tc**3+a4*Tc**4)*S+(a5+a6*Tc+a7*Tc**2)*S**1.5+a8*S**2) 
    return density_sw   
    
def dlnasw_ds(Tc, S):
    # water activity data of seawater is from Millero and Leung (1976,American
    # Journal of Science,276,1035-1077). Table 19 was reproduced using
    # Eqs.(14,22,23,88,107) then were fitted to polynominal equation.
    # dlnawds is partial derivative of natural logarithm of water activity
    # w.r.t.salinity
    # lnaw = (-1.64555e-6-1.34779e-7*Tc+1.85392e-9*Tc**2-1.40702e-11*Tc**3)+......
    #            (-5.58651e-4+2.40452e-7*Tc-3.12165e-9*Tc**2+2.40808e-11*Tc**3)*S+......
    #            (1.79613e-5-9.9422e-8*Tc+2.08919e-9*Tc**2-1.39872e-11*Tc**3)*S**1.5+......
    #            (-2.31065e-6-1.37674e-9*Tc-1.93316e-11*Tc**2)*S**2 
    
    dlnawds = (-5.58651e-4+2.40452e-7*Tc-3.12165e-9*Tc**2+2.40808e-11*Tc**3
               )+1.5*(1.79613e-5-9.9422e-8*Tc+2.08919e-9*Tc**2-1.39872e-11*Tc**3
               )*S**0.5+2*(-2.31065e-6-1.37674e-9*Tc-1.93316e-11*Tc**2)*S
    return dlnawds
    
    # density derivative of refractive index from PMH model
def PMH(n_wat):
    n_wat2 = n_wat**2 
    ##print 'n_wat2', n_wat2
    n_density_derivative=(n_wat2-1)*(1+2./3.*(n_wat2+2.)*(n_wat/3.-1./3./n_wat)**2) 
    ##print 'n_density_derivative',  n_density_derivative
    return n_density_derivative
        

### main program ####################################################
if __name__=="__main__":
    #if len(sys.argv)<2:
    #    print('Enter filename for purewater absoption coefficents:\n(return for "purewater_abs_coefficients_v1.dat")')
    #    fn=raw_input('>')
    #    if not fn:
    #        fn='purewater_abs_coefficients_v1.dat'
            
    #enter here the desired temperature
    Tc=20.0
    #enter here the desired salinity
    S=0.0
    # from wavelength
    aw=300
    #to wavelength
    ew=4000
    fn = None
  #absorption ##############################################################
#    fn='purewater_abs_coefficients.dat'
    wl, abso,  a_err=absorption(aw, ew,  S,  Tc,  fn)
    #optional: plot the result
    import pylab
    #standard spektrum
    wl, abso,  a_err=absorption(300, 4000,  30,  20,  fn)
    
    Tc=21
    wl, abso_60,  a_err_60=absorption(aw, ew,  S,  Tc,  fn)
    pylab.subplot(211)
    Y=(abso_60/abso-1)
    ymax=pylab.ma
    pylab.semilogy(wl, abso)
    pylab.subplot(212)
    pylab.semilogy(wl, a_err)
#    pylab.semilogy(wl, abso,  wl, abso+a_err, ':',  wl, abso-a_err, ':')
#    Tc=60
#    wl, abso,  a_err=absorption(aw, ew,  S,  Tc,  fn)
#    pylab.semilogy(wl, abso,  wl, abso+a_err, ':',  wl, abso-a_err, ':')
    pylab.show()
    
    print('done')
