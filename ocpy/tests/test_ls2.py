""" Tests for the ls2 module """

#clear command window and workspace; close figures  
import numpy as np
import pathlib
import datetime
import pytest

import pandas

from oceancolor.ls2.io import load_LUT
from oceancolor.ls2.ls2_main import LS2_main

from IPython import embed

def data_path(filename):
    data_dir = pathlib.Path(__file__).parent.absolute().joinpath('files')
    # TODO: This really should have the `.resolve()`, but it crashes the
    #       Windows/python3.9 CI test (only that one).  When PypeIt advances
    #       to python>=3.10, reinstate the last part of the following line:
    return str(data_dir.joinpath(filename))#.resolve())


#define input parameters:  

def test_ls2_run():
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Test script for the LS2 code. The LS2 code is run for ten specified inputs
    %and the resulting output from the test script is saved to
    %LS2_test_run_YYYYMMDD.xls file for comparison with provided output
    %file LS2_test_run.xls
    %
    %Reference: 
    %
    %Loisel, H., D. Stramski, D. Dessaily, C. Jamet, L. Li, and R.
    %A. Reynolds. 2018. An inverse model for estimating the optical absorption
    %and backscattering coefficients of seawater from remote-sensing
    %reflectance over a broad range of oceanic and coastal marine environments.
    %Journal of Geophysical Research: Oceans, 123, 2141–2171. doi:
    %10.1002/2017JC013632 
    %
    %Created: October 12, 2022
    %Completed: October 14, 2022
    %Updates: N/A
    %
    %M. Kehrli, R. A. Reynolds, and D. Stramski 
    %Ocean Optics Research Laboratory, Scripps Institution of Oceanography
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''


    #input solar zenith angle [deg]  
    sza = [58.3534804715254, 57.8478623967075, 55.7074826164700, 56.3604406592725,  
        56.4697386744023,56.8356539563103,46.7609493705841,42.9575051280531,  
        39.1258070531710,36.7202617271538]  

    #input light wavelengths [nm]  
    lambda_ = [412,443,490,510,555,670]  

    #input Rrs [sr^-1]  
    Rrs = np.array([[0.00233524115013000,0.00294175586230000,0.00393113794469000,  
                    0.00340969897475000,0.00229711245356000,0.000126252474968051],  
                    [0.00389536285161000,0.00380553119300000,0.00370193558239000,  
                    0.00287218595770000,0.00161623731374000, 6.63224220817391e-05], 
        [0.00445894839227000,0.00429316064858000,
        0.00415100665860000,0.00310168437705000,0.00160982272779000,
        7.69489021946615e-05], [0.00369218043611000,0.00338770153633000,
        0.00335975955251000,0.00277502852091000,0.00163977701600000,
        9.94781359675408e-05], [0.00439946721488000,0.00410361567739000,
        0.00384560916516000,0.00281071322991000,0.00145740677722000,
        7.57133335455754e-05], [0.00407536933494000,0.00393833413124000,
        0.00377045558470000,0.00294465987184000,0.00155830890220000,
        5.98787860868771e-05], [0.00839651952650000,0.00703053123029000,
        0.00515382893062000,0.00328301126479000,0.00143446649046000,
        7.37164311815052e-05], [0.0108879293669300,0.00844627966419000,
        0.00598548662295000,0.00380069346088000,0.00162622388794000,
        3.81004114285704e-05], [0.00996484275824000,0.00805847039179000,
        0.00569828863615000,0.00339472028869000,0.00146868678397000,
        3.54625174513673e-05], [0.00654238127443000,0.00606473089160000,
        0.00511788480497000,0.00358608311014000,0.00161083692320000,
        3.81378558561512e-05]])

    #input Kd [m^-1]  
    Kd = np.array([[0.187632830961948,0.138869886516291,0.0903870213746760,0.0887979802802584,  
                    0.105697824445020,0.562373761609887],  
                    [0.0959924017590809,0.0776637266248738,0.0621434486477123,0.0679142033853613,  
                    0.0929498240015621,0.541582660606669],   
        [0.0838916051527420,0.0676778528875712,0.0548092373676583,
        0.0606911584475843,0.0867712558516902,0.532867854871956],
        [0.107015673766332,0.0859385855948466,0.0669716298720907,
        0.0720650641466051,0.0953025020320744,0.535712492953156],
        [0.0819636283043127,0.0664145944821315,0.0545267015934212,
        0.0608791413489761,0.0875697338410355,0.536433237168806],
        [0.0870883103721925,0.0704351866270744,0.0568638096692833,
        0.0627223237995291,0.0883024973409085,0.531148295108121],
        [0.0397738921806664,0.0355556278806364,0.0371061601830686,
        0.0462176494000053,0.0758945231309703,0.500096222235249],
        [0.0358540683547611,0.0323961377543049,0.0353061269500453,
        0.0446719301032437,0.0738646909883413,0.484404799921858],
        [0.0352724958202497,0.0316083929271783,0.0342524209435186,
        0.0435171225162597,0.0732685592707990,0.487660368105567],
        [0.0497417842712074,0.0416885187358688,0.0380380139571626,
        0.0449932910491427,0.0717349865479659,0.474223095982504],
                    ])

    #input aw [m^-1]  
    aw = [0.004673,0.00721,0.015,0.0325,0.0592,0.439]  

    #input bw [m^-1]  
    bw = [0.00658572,0.004777,0.003098,0.002598,0.00184881,0.0008]  

    #input bp [m^-1]  
    bp = np.array([[0.370479484460264,0.344554283516092,0.311505199178834,0.299289309014959,  
                    0.275022608284016,0.227817235220342],
                    [0.222636201160542,0.207056692727185,0.187196152812537,0.179855127212045,  
                        0.165272279059717,0.136904649071855],    
        [0.193719314154925,0.180163335060562,0.162882362105773,
        0.156494818493782,0.143806049426719,0.119122921540043],
        [0.260143039084721,0.241938898652156,0.218732514495725,
        0.210154768829226,0.193115192978207,0.159968555377470],
        [0.179080613026910,0.166549012566787,0.150573903198136,
        0.144669044249190,0.132939121742499,0.110121212786697],
        [0.204161647168565,0.189874940481825,0.171662446190711,
        0.164930585555782,0.151557835375583,0.125544177064849],
        [0.0934854453083219,0.0869435744176718,0.0786040887082217,
        0.0755215754255463,0.0693982044450966,0.0574865723388487],
        [0.0886492843355160,0.0824458355445431,0.0745377656045563,
        0.0716147159730051,0.0658081173805993,0.0545126942481083],
        [0.0800160719773182,0.0744167531707790,0.0672788197033778,
        0.0646404346169708,0.0593993182966759,0.0492039129173957],
        [0.132749681531226,0.123460200430847,0.111618099573194,
        0.107240919197775,0.0985457095330905,0.0816311474490525],
                    ])

    #input LS2 LUTs  
    LS2_LUT = load_LUT()

    #input Raman Flag  
    input_Flag_Raman = 1  

    #preallocate output variables  
    output_a = np.full(Rrs.shape, np.nan)  
    output_anw = np.full(Rrs.shape, np.nan)  
    output_bb = np.full(Rrs.shape, np.nan)  
    output_bbp = np.full(Rrs.shape, np.nan)      
    output_kappa = np.full(Rrs.shape, np.nan)  

    #loop to run LS2 for all samples at every wavelength and calculate outputs
    for i in range(Rrs.shape[0]):
        for j in range(Rrs.shape[1]):
            output_a[i,j], output_anw[i,j], output_bb[i,j], output_bbp[i,j], output_kappa[i,j] = LS2_main(
                sza[i], lambda_[j], Rrs[i,j], Kd[i,j], aw[j], bw[j], bp[i,j], LS2_LUT, input_Flag_Raman)

    #save inputs and outputs into an excel file  
    date_str = datetime.datetime.today().strftime('%Y%m%d') 
    outfile = f'LS2_test_run_{date_str}.xlsx'
    save_dfs = {}
    with pandas.ExcelWriter(outfile) as writer:  
        for i in range(Rrs.shape[1]):
            df = pandas.DataFrame()
            df['Input wavelength [nm]'] = lambda_[i]
            df['Input sza [deg]'] = sza
            df['Input Rrs [1/sr]'] = Rrs[:,i]
            df['Input Kd [1/m]'] = Kd[:,i]
            df['Input aw [1/m]'] = aw[i]
            df['Input bw [1/m]'] = bw[i]
            df['Input bp [1/m]'] = bp[:,i]
            df['Ouput a [1/m]'] = output_a[:,i]
            df['Output anw [1/m]'] = output_anw[:,i]
            df['Output bb [1/m]'] = output_bb[:,i]
            df['Output bbp [1/m]'] = output_bbp[:,i]
            df['Output kappa [dim]'] = output_kappa[:,i]
            
            #df.columns = ['Input wavelength [nm]','Input sza [deg]','Input Rrs [1/sr]',  
            #                'Input Kd [1/m]','Input aw [1/m]','Input bw [1/m]','Input bp [1/m]',  
            #                'Ouput a [1/m]','Output anw [1/m]','Output bb [1/m]',  
            #                'Output bbp [1/m]','Output kappa [dim]']  
            df.to_excel(writer, sheet_name=f'{lambda_[i]} nm', index=False)

            # Save
            save_dfs[f'{lambda_[i]} nm'] = df

    print(f'Output saved to {outfile}')

    # Compare to the LS2 test run output
    ls2_test_run = pandas.read_excel(data_path('LS2_test_run.xls'), sheet_name=None)

    # Test
    ls2_412 = ls2_test_run['412 nm']
    df = save_dfs['412 nm']

    assert np.allclose(ls2_412['Output bb [1/m]'].values, 
                       df['Output bb [1/m]'].values, 
                       rtol=1e-3)