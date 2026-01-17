% Hypera_RhoFromNDSpot computes a new sphere reflectivity from a
% measurement of clean water and the ND spot
%
% Inputs:
%   cal:          Calibration .mat file for the specific Hyper-a instrument
%   pureWater:    Hyper-a measurement with pure water in the sphere and
%                 Fluorilon white plug installed. (file path or struct)
%   T_pureWater:  Water temperature in Celsius of the pure water
%                 measurement
%   spot:         Hyper-a measurement with pure water in the sphere and
%                 ND black spot installed. (file path or struct)
%   T_spot:       Water temperature in Celsius of the spot measurement             
%
% Outputs:
%   rho:          New sphere reflectivity based on the supplied spot
%                 measurement and the known spot absorption.
%
%   a_spot:       Optional output of spot absorption (1/m) using factory rho.
%                 Water absorption has been removed.
%
% Sequoia Scientific, Inc.
% v2.0 
% 11/03/2025


function [rho, a_spot] = Hypera_RhoFromNDSpot(cal, pureWater, T_pureWater, spot, T_spot)

%% Process Spot measurement
[a_spot, T_AB] = Hypera_Process(cal, pureWater, T_pureWater, 0, spot, T_spot, 0, ChlFluorCorr=false); % salinity assumed to be 0 PSU

%% Compute Pure Water Absorption 
PureWater_aW_modeled = Hypera_lib.Get_IOCCG_aW(cal.wl, T_pureWater, 0); 
Cal_spot_aW_modeled  = Hypera_lib.Get_IOCCG_aW(cal.wl, T_spot, 0);       

% add water absorption to known spot absorption
Cal_spot_a_total = cal.spotAbsorp + Cal_spot_aW_modeled;

%% Calculate new rho (sphere reflectivity) using known pure water and spot absorption values
rho = Hypera_lib.ComputeRho(cal, ...
    Cal_spot_a_total, ...
    PureWater_aW_modeled, ...
    median(T_AB,1));

rho = smoothdata(rho,"sgolay",30);

end