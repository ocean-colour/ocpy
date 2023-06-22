import numpy as np
import warnings

def LS2_main(sza,lambda_,Rrs,Kd,aw,bw,bp,LS2_LUT,Flag_Raman):
    """Implements the LS2 inversion model to calculate a, anw, bb, and bbp from 
    Rrs at specified input light wavelength
    
    Required Function Inputs: sza, lambda_, Rrs, Kd, aw, bw, bp, LS2_LUT,
    Flag_Raman
        sza [float]: Solar zenith angle [deg]

        lambda_ [float]: Input light wavelength [nm] 
        Rrs [float]: Spectral remote-sensing reflectance [sr^-1] at light wavelength lambda_  

        Kd [float]: Spectral attenuation coefficient of downwelling planar irradiance [m^-1] at lambda_, averaged between the sea surface and first attenuation depth    

        aw [float]: Spectral pure seawater absorption coefficient [m^-1] at lambda_   

        bw [float]: Spectral pure seawater scattering coefficient [m^-1] at lambda_

        bp [float]: Spectral particulate scattering coefficient [m^-1] at lambda_     

        LS2_LUT [dict]: Structure containing five required look-up tables; 

        Flag_Raman[int]: Flag to apply or omit a Raman scattering correction to Rrs. If input value = 1, a Raman scattering correction is applied to Rrs and output is recalculated via a single iteration. If input value is not equal to 1, no Raman scattering correction is applied to Rrs and initial model output is returned
    
    Outputs: a, anw, bb, bbp, kappa
        a [float]: Spectral absorption coefficient [m^-1] at lambda_

        anw [float]: Spectral nonwater absorption coefficient [m^-1] at lambda_  

        bb [float]: Spectral backscattering coefficient [m^-1] at lambda_  

        bbp [float]: Spectral particulate backscattering coefficient [m^-1] at lambda_  

        kappa [float]: Value of the Raman scattering correction factor,kappa (dim), applied to input Rrs
    %Version History: 
    %2018-04-04: Original implementation in C written by David Dessailly
    %2020-03-23: Original Matlab version, D. Jorge 
    %2022-09-01: Revised Matlab version, M. Kehrli
    %2022-11-03: Final Revised Matab version, M. Kehrli, R. A. Reynolds and D. Stramski
    %2023-06-22: Converted by Python by JXP and Claude+
    """
# %% Check function arguments and existence of LUTs
    nw = 1.34  # Refractive index of seawater

# Step 1
# %% Calculation of muw, the cosine of the angle of refraction of the solar beam just beneath the sea surface
    muw = np.cos(np.arcsin(np.sin(sza * np.pi/180)/nw))  # [dim]

#%% Step 2: Calculation of Kd, the average spectral attenuation coefficient of downwelling planar irradiance between the sea surface and first attenuation depth
#    %In this version of the code, Kd is assumed to be known and provided as
#    %input in units of [m^-1]. In Loisel et al. (2018), it is obtained from
#    %a separate neural network algorithm that estimates Kd from
#    %remote-sensing reflectance. Note: See the separate Kd_NN_Distribution
#    %repository where we provide a version of neural network algorithm to
#    %estimate Kd

#%% Step 3: Calculation of b, the total scattering coefficient in units of [m^-1]
#    %In this version of the code, bp and bw are assumed to be known and
#    %provided as input in units of [m^-1]. In Loisel et al. (2018), bp is
#    %estimated from chlorophyll-a concentration (Chla) where Chla is
#    %calculated from spectral remote-sensing reflectance using the ocean
#    %color algorithm OC4v4
    b = bp + bw  # [m^-1] 

# Step 4
# %% Calculation of eta [dim], the ratio of the pure seawater (molecular) scattering coefficient to the total scattering coefficient
    eta = bw/b

# %% Steps 5 & 7: Calculation of a and bb from Eqs. 9 and 8
# %% Calculation of a and bb from Eqs. 9 and 8
    if not np.isnan(eta) and not np.isnan(muw):
        #find leftmost index of eta and mu values in the LUTs for interpolation
        idx_eta = seek_pos(eta,LS2_LUT['eta'],'eta')
        idx_muw = seek_pos(muw,LS2_LUT['muw'],'muw')
        
        #if eta or mu is outside of the bounds of LUTs return nan outputs
        if np.isnan(idx_eta) or np.isnan(idx_muw):
            a = np.nan
            anw = np.nan
            bb = np.nan
            bbp = np.nan
            kappa = np.nan
            return
        
        #calculation of a from Eq. 9
        a00 = Kd/(LS2_LUT['a'][idx_eta,idx_muw,0] + LS2_LUT['a'][idx_eta,idx_muw,1]*Rrs + LS2_LUT['a'][idx_eta,idx_muw,2]*Rrs**2 + LS2_LUT['a'][idx_eta,idx_muw,3]*Rrs**3) 
        a01 = Kd/(LS2_LUT['a'][idx_eta,idx_muw+1,0] + LS2_LUT['a'][idx_eta,idx_muw+1,1]*Rrs + LS2_LUT['a'][idx_eta,idx_muw+1,2]*Rrs**2 + LS2_LUT['a'][idx_eta,idx_muw+1,3]*Rrs**3)
        a10 = Kd/(LS2_LUT['a'][idx_eta+1,idx_muw,0] + LS2_LUT['a'][idx_eta+1,idx_muw,1]*Rrs + LS2_LUT['a'][idx_eta+1,idx_muw,2]*Rrs**2 + LS2_LUT['a'][idx_eta+1,idx_muw,3]*Rrs**3)
        a11 = Kd/(LS2_LUT['a'][idx_eta+1,idx_muw+1,0] + LS2_LUT['a'][idx_eta+1,idx_muw+1,1]*Rrs + LS2_LUT['a'][idx_eta+1,idx_muw+1,2]*Rrs**2 + LS2_LUT['a'][idx_eta+1,idx_muw+1,3]*Rrs**3)
        
        #calculate a using 2-D linear interpolation determined from bracketed values of eta and muw
        a = interpolate.interp2d(LS2_LUT['eta'][idx_eta:idx_eta+2],LS2_LUT['muw'][idx_muw:idx_muw+2],
                                 np.array([[a00,a01],[a10,a11]]),eta,muw) 
        
        #calculation of bb from Eq. 8
        #%bb is first calculated for four combinations of LUT values
        #%bracketing the lower and upper limits of eta and muw
        bb00 = Kd*(LS2_LUT['bb'][idx_eta,idx_muw,0]*Rrs + LS2_LUT['bb'][idx_eta,idx_muw,1]*Rrs**2 + LS2_LUT['bb'][idx_eta,idx_muw,2]*Rrs**3)
        bb01 = Kd*(LS2_LUT['bb'][idx_eta,idx_muw+1,0]*Rrs + LS2_LUT['bb'][idx_eta,idx_muw+1,1]*Rrs**2 + LS2_LUT['bb'][idx_eta,idx_muw+1,2]*Rrs**3)
        bb10 = Kd*(LS2_LUT['bb'][idx_eta+1,idx_muw,0]*Rrs + LS2_LUT['bb'][idx_eta+1,idx_muw,1]*Rrs**2 + LS2_LUT['bb'][idx_eta+1,idx_muw,2]*Rrs**3)
        bb11 = Kd*(LS2_LUT['bb'][idx_eta+1,idx_muw+1,0]*Rrs + LS2_LUT['bb'][idx_eta+1,idx_muw+1,1]*Rrs**2 + LS2_LUT['bb'][idx_eta+1,idx_muw+1,2]*Rrs**3)
        
        #%calculate bb using 2-D linear interpolation determined from
        #%bracketed values of eta and muw
        bb = interpolate.interp2d(LS2_LUT['eta'][idx_eta:idx_eta+2],LS2_LUT['muw'][idx_muw:idx_muw+2],
                                 np.array([[bb00,bb01],[bb10,bb11]]),eta,muw)
        
    else:
        a = np.nan
        anw = np.nan
        bb = np.nan
        bbp = np.nan
        kappa = np.nan
        return
