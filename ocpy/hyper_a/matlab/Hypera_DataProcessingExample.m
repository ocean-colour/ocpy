%% EXAMPLE HYPER-A PROCESSING SCRIPT
%
% This script will process raw Hyper-a data to absorption coefficient
% (1/m). Data can be supplied from Hyper-a .bin file or .mat files.
%
% INPUTS
% Edit the first 'Load Data' section to point to the data files (sample and
% pure water reference). Both temperature (Celsius) and salinity (PSU) must
% be measured and entered below. Lastly, you must specify the
% instrument specific calibration file (.mat file).
%
% OUTPUT
% See 'hypera_proc' table for final absorption results. The absorption
% coefficient (1/m) in the table has absorption by water removed.
% Wavelengths and instrument configuration information are stored in the 
% table properties (hypera_proc.Properties.CustomProperties).
%
% Sequoia Scientific, Inc.
% v2.0 
% 11/03/2025

%% Load Data

% path to calibration file supplied with instrument
load('.\Calibrations\CAL_20240801.mat');

 sample = '.\ExampleData\Sample.mat'; % data collected in .mat file in real time using 'Hypera_RealTimeExample.m'
% OR, data could be loaded from a .bin file offloaded from the instrument, example:
% sample = 'P:\Hyper-a\Instrument Files\SN-8021 - Sequoia Engineering\HA_2024-07-31_14-43-06_Spot2.bin';

 purewater = '.\ExampleData\PureWater.mat'; % data collected in .mat file in real time using 'Hypera_RealTimeExample.m'
% OR, data could be loaded from a .bin file offloaded from the instrument, example:
% purewater = 'P:\Hyper-a\Instrument Files\SN-8021 - Sequoia Engineering\HA_2024-07-31_14-35-23_PureWater2.bin';

% Temperature and salinity values for the sample and pure water measurements.
% Typically the pure water measurements are discrete with only one temperature and salinity associated with the measurement.
% If you have an array of water temperatures and salinities for the sample measurements, see note in 'Process Data to Absorption' section.
% Otherwise, enter one T and S sample value for the whole dataset.

T_PureWater = 22; % Celsius
S_PureWater = 0;  % PSU
T_Sample = 22;
S_Sample = 0;

%% Use ND spot measurement to correct for drift in cavity reflectivity (optional) 

% PathToSpotMeasurementFile = '';
% PathToPureWaterFile = '';
% 
% T_whitePlug = 22;
% T_spot = 22;
% 
% [NewRho, a_spot] = Hypera_RhoFromNDSpot(cal, PathToPureWaterFile, T_whitePlug, PathToSpotMeasurementFile, T_spot);
% 
% figure
% plot(cal.wl, cal.rho)
% hold on
% plot(cal.wl, NewRho);
% xlabel('Wavelength (nm)')
% ylabel('rho')
% grid on
% legend({'Factory Rho', 'New Rho From ND Spot'});
% 
% % load new rho into calibration structure to pass into hypera_Process
% cal.rho = NewRho;

%% Process Data to Absorption

% This is the simple case where there is one sample temperature and salinity for
% the whole data set
hypera_proc = Hypera_Process(cal, ...
                            purewater, ...
                            T_PureWater, ...
                            S_PureWater, ...
                            sample, ...
                            T_Sample, ...
                            S_Sample);

% If you have an array of sample water temperatures and salinities, modify this
% processing as needed: 

% hypera_proc = Hypera_Process(cal, ...
%                             purewater, ...
%                             T_PureWater, ...
%                             S_PureWater, ...
%                             sample, ...
%                             0, ...                 
%                             0, ...                 
%                             RemoveWaterAbsorption=false); % <----  turn off pure water subtraction in the main processing
% 
% % Interpolate your temperature and salinity data onto the timestamps of the Hyper-a data
% YourTempData = ;
% YourSalData = ;
% YourTSTimestamps = ;
% 
% T_Sample_interp = interp1(YourTSTimestamps, YourTempData, hypera_proc.date);
% S_Sample_interp = interp1(YourTSTimestamps, YourSalData,  hypera_proc.date);
% 
% % Calculate the water absorption for each sample using your specific
% % temperature and salinity values
% aW_sample = Hypera_lib.Get_IOCCG_aW(cal.wl,T_Sample_interp,S_Sample_interp);
% 
% % subtract water absorption from Hyper-a result
% hypera_proc.absorption = hypera_proc.absorption - aW_sample;

%% Plot results

figure
subplot(2,2,1)
plot(hypera_proc.Properties.CustomProperties.wavelengths,hypera_proc.absorption,'.-')
xlabel('Wavelength (nm)')
ylabel('Absorption (1/m)')
grid on

subplot(2,2,2)
plot(hypera_proc.date,hypera_proc.depth,'.-k', 'markersize', 10, 'linewidth', 1)
xlabel('Date')
ylabel('Depth (m)')
grid on

subplot(2,2,3)
plot(hypera_proc.date,hypera_proc.waterTemp,'.-k', 'markersize', 10, 'linewidth', 1)
xlabel('Date')
ylabel('Water Temperature (C)')
grid on

subplot(2,2,4)
plot(hypera_proc.date,hypera_proc.inputVoltage,'.-k', 'markersize', 10, 'linewidth', 1)
xlabel('Date')
ylabel('Input Voltage (V)')
grid on

