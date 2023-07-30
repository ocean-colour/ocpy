import numpy as np
import warnings
from scipy import interpolate

from IPython import embed

def LS2_main(sza:float,lambda_:float,Rrs:float,Kd:float,aw:float,
             bw:float,bp:float,LS2_LUT:dict,Flag_Raman:bool):
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

        Flag_Raman[bool]: Flag to apply or omit a Raman scattering correction to Rrs. If input value = 1, a Raman scattering correction is applied to Rrs and output is recalculated via a single iteration. If input value is not equal to 1, no Raman scattering correction is applied to Rrs and initial model output is returned
    
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
        idx_eta = LS2_seek_pos(eta,LS2_LUT['eta'],'eta')
        idx_muw = LS2_seek_pos(muw,LS2_LUT['muw'],'muw')
        
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
        f = interpolate.RegularGridInterpolator(
            (LS2_LUT['eta'][idx_eta:idx_eta+2].flatten(), 
             LS2_LUT['muw'][idx_muw:idx_muw+2].flatten()), 
            np.array([[a00,a01],[a10,a11]]))
        a = float(f([eta,muw]))
        #a = interpolate.interp2d(LS2_LUT['eta'][idx_eta:idx_eta+2],LS2_LUT['muw'][idx_muw:idx_muw+2],
        #                         np.array([[a00,a01],[a10,a11]]),eta,muw) 
        
        #calculation of bb from Eq. 8
        #%bb is first calculated for four combinations of LUT values
        #%bracketing the lower and upper limits of eta and muw
        bb00 = Kd*(LS2_LUT['bb'][idx_eta,idx_muw,0]*Rrs + LS2_LUT['bb'][idx_eta,idx_muw,1]*Rrs**2 + LS2_LUT['bb'][idx_eta,idx_muw,2]*Rrs**3)
        bb01 = Kd*(LS2_LUT['bb'][idx_eta,idx_muw+1,0]*Rrs + LS2_LUT['bb'][idx_eta,idx_muw+1,1]*Rrs**2 + LS2_LUT['bb'][idx_eta,idx_muw+1,2]*Rrs**3)
        bb10 = Kd*(LS2_LUT['bb'][idx_eta+1,idx_muw,0]*Rrs + LS2_LUT['bb'][idx_eta+1,idx_muw,1]*Rrs**2 + LS2_LUT['bb'][idx_eta+1,idx_muw,2]*Rrs**3)
        bb11 = Kd*(LS2_LUT['bb'][idx_eta+1,idx_muw+1,0]*Rrs + LS2_LUT['bb'][idx_eta+1,idx_muw+1,1]*Rrs**2 + LS2_LUT['bb'][idx_eta+1,idx_muw+1,2]*Rrs**3)
        
        #%calculate bb using 2-D linear interpolation determined from
        #%bracketed values of eta and muw
        #embed(header='LS2_main: 115')
        f = interpolate.RegularGridInterpolator(
            (LS2_LUT['eta'][idx_eta:idx_eta+2].flatten(),
            LS2_LUT['muw'][idx_muw:idx_muw+2].flatten()), 
            np.array([[bb00,bb01],[bb10,bb11]]))
        bb = float(f([eta,muw]))
        #bb = interpolate.interp2d(LS2_LUT['eta'][idx_eta:idx_eta+2],LS2_LUT['muw'][idx_muw:idx_muw+2],
        #                         np.array([[bb00,bb01],[bb10,bb11]]),eta,muw)
        
    else:
        a = np.nan
        anw = np.nan
        bb = np.nan
        bbp = np.nan
        kappa = np.nan
        return

    # Step 9: Application of the Raman scattering correction if selected  
    #If Flag_Raman is set to 1 (true), apply Raman correction to input Rrs  
    #and recalculate a and bb the same as above. Otherwise no correction is  
    #applied and original values are returned with kappa value of 1  
    if Flag_Raman:      
        #call subfunction LS2_calc_kappa 
        kappa = LS2_calc_kappa(bb/a,lambda_,LS2_LUT['kappa'])

        #apply Raman scattering correction to Rrs and recalculate a & bb  
        if not np.isnan(kappa):        
            Rrs = Rrs*kappa    
            
            #calculation of a from Eq. 9  
            
            #a is first calculated for four combinations of LUT values  
            #bracketing the lower and upper limits of eta and muw  
            a00 = Kd/(LS2_LUT['a'][idx_eta,idx_muw,0] +                
                LS2_LUT['a'][idx_eta,idx_muw,1]*Rrs +                
                LS2_LUT['a'][idx_eta,idx_muw,2]*Rrs**2 +                
                LS2_LUT['a'][idx_eta,idx_muw,3]*Rrs**3)  
        
            #... (similar calculations for a01, a10, a11)
            
            #calculate a using 2-D linear interpolation determined from  
            #bracketed values of eta and muw  
            f = interpolate.RegularGridInterpolator(
                (LS2_LUT['eta'][idx_eta:idx_eta+2].flatten(), 
                LS2_LUT['muw'][idx_muw:idx_muw+2].flatten()), 
                np.array([[a00,a01],[a10,a11]]))
            a = float(f([eta,muw]))
                    
            #calculation of bb from Eq. 8  
            
            #bb is first calculated for four combinations of LUT values  
            #bracketing the lower and upper limits of eta and muw  
            bb00 = Kd*(LS2_LUT['bb'][idx_eta,idx_muw,0]*Rrs +            
                    LS2_LUT['bb'][idx_eta,idx_muw,1]*Rrs**2 +             
                    LS2_LUT['bb'][idx_eta,idx_muw,2]*Rrs**3)  

            bb01 = Kd*(LS2_LUT['bb'][idx_eta,idx_muw+1,0]*Rrs +            
                    LS2_LUT['bb'][idx_eta,idx_muw+1,1]*Rrs**2 +             
                    LS2_LUT['bb'][idx_eta,idx_muw+1,2]*Rrs**3)  

            bb10 = Kd*(LS2_LUT['bb'][idx_eta+1,idx_muw,0]*Rrs +            
                    LS2_LUT['bb'][idx_eta+1,idx_muw,1]*Rrs**2 +             
                    LS2_LUT['bb'][idx_eta+1,idx_muw,2]*Rrs**3)  
            
            bb11 = Kd*(LS2_LUT['bb'][idx_eta+1,idx_muw+1,0]*Rrs +            
                    LS2_LUT['bb'][idx_eta+1,idx_muw+1,1]*Rrs**2 +             
                    LS2_LUT['bb'][idx_eta+1,idx_muw+1,2]*Rrs**3)  
            
            #calculate bb using 2-D linear interpolation determined from  
            #bracketed values of eta and muw  
            f = interpolate.RegularGridInterpolator(
                (LS2_LUT['eta'][idx_eta:idx_eta+2].flatten(),
                LS2_LUT['muw'][idx_muw:idx_muw+2].flatten()), 
                np.array([[bb00,bb01],[bb10,bb11]]))
            bb = float(f([eta,muw]))
        #if Flag is not 1, do nothing and return original values of a, anw, bb,  
        #bbp with kappa returned as 1  
    else:        
        kappa = 1

    # Steps 6 & 8: Calculation of anw and bbp  
    anw = a - aw  # spectral nonwater absorption coefficient [m^-1]  
    bbw = bw/2    #spectral pure seawater backscattering coefficient [m^-1]  
    bbp = bb - bbw #spectral particulate backscattering coefficient [m^-1]  

    # If output coefficients are negative, replace with NaN  
    if a < 0 :        
        warnings.warn('Solution for a is negative. Output a set to nan.')  
        a = np.nan   
    if anw < 0 :        
        warnings.warn('Solution for anw is negative. Output anw set to nan.')  
        anw = np.nan      
    if bb < 0 :        
        warnings.warn('Solution for bb is negative. Output bb set to nan.')  
        bb = np.nan  
    if bbp < 0 :        
        warnings.warn('Solution for bbp is negative. Output bbp set to nan.')  
        bbp = np.nan

    return a, anw, bb, bbp, kappa

def LS2_seek_pos(param:float, LUT:np.ndarray, itype:str):
    '''
    %MK remake of LS2 seek_pos subroutine. The subroutine finds the leftmost
    %position of the input parameter in relation to its input LUT
    %
    %Inputs: param, LUT, type
    %   param [1x1 Double]: Input muw or eta value
    %
    %   LUT [nx1 Double]: Look-up table of muw or eta values used to determine
    %   coefficients in Loisel et al. 2018. If the input is associated with muw
    %   the LUT must be 8x1 and sorted in descending order, and if the input is
    %   associated with eta the LUT must be 21x1 and sorted in ascending order
    %
    %   type [String]: Characterize param input. Valid values are 'muw' or
    %   'eta'. Other inputs will produce an error
    %
    %Output: idx
    %   idx [1x1 Double]: Leftmost position/index of input param in relation to
    %   its LUT
    %
    %Created: September 8, 2021
    %Completed: September 8, 2021
    %Updates: October 15, 2022 - Added warning messages and changed logic for
    %output
    %
    %Matthew Kehrli
    %Ocean Optics Research Laboratory, Scripps Institution of Oceanography
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    
    #check function input arguments 
    if(itype == 'muw') and (len(LUT) != 8 or not sorted(LUT, reverse=True)): 
        raise ValueError('Look-up table for mu_w must be a 8x1 array sorted in descending order')
    elif (itype == 'eta') and (len(LUT) != 21 or not sorted(LUT)): 
        raise ValueError('Look-up table for eta must be a 21x1 array sorted in ascending order') 
    
    if param < min(LUT): 
        warnings.warn('{} is outside the lower bound of look-up table. Solutions of a and bb are output as nan.'.format(itype))
        return np.nan 
    if param > max(LUT): 
        warnings.warn('{} is outside the upper bound of look-up table. Solutions of a and bb are output nan.'.format(itype))
        return np.nan       
    
    for i in range(len(LUT)-1):
        if ((itype == 'muw' and param <= LUT[i] and param > LUT[i+1]) or  
            (itype == 'eta' and param >= LUT[i] and param < LUT[i+1])): 
            idx = i
            
    return idx

def LS2_calc_kappa(bb_a, lam, rLUT): 
    '''
    %The subroutine determines kappa using a linear
    %interpolation/extrapolation from the Raman scattering look-up tables
    %
    %Inputs: bb_a, lambda, rLUT
    %   bb_a [1x1 Double]: Backscattering to absorption coefficient ratio
    %   output from LS2 model
    %
    %   lam [1x1 Double]: Wavelength of bb/a ratio
    %
    %   rLUT [101x7 Double]: Look-up table for kappa
    %
    %Output: kappa
    %   kappa [1x1 Double]: Value of Raman scattering correction at a
    %   particular bb/a ratio and wavelength
    %
    %Created: September 14, 2021
    %Completed: September 14, 2021
    %Updates:
    %
    %Matthew Kehrli
    %Ocean Optics Research Laboratory, Scripps Institution of Oceanography
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    # interpolate minimum and maximum allowable bb/a values that calculate kappa to the input wavelength.
    mins = np.interp(lam, rLUT[:,0], rLUT[:,5])  
    maxs = np.interp(lam, rLUT[:,0], rLUT[:,6])  
    
    #check if bb/a ratio falls within min/max range
    if bb_a >= mins and bb_a <= maxs:    
        #calculate all kappas for the input bb/a ratio
        kappas = rLUT[:,1]*bb_a**3 + rLUT[:,2]*bb_a**2 + rLUT[:,3]*bb_a + rLUT[:,4] 
        #calculate output kappa using a 1-D linear interpolation to the input wavelength
        kappa = np.interp(lam, rLUT[:,0], kappas)   
    else:
        kappa = np.nan      
        #send warning message to user that no Raman correction is applied to input
        warnings.warn('No Raman Correction since bb/a value is outside of the acceptable range. Kappa set to nan and no correction is applied. See Raman Correction LUT.')  
        
    return kappa