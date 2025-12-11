% Hypera_Process processes raw Hyper-a data to absorption coefficient
% (1/m). Data can be imported from Hyper-a .bin file or .mat files. This
% script calls the core Hyper-a processing functions in 'Hypera_lib.m'.
%
% Inputs:
%   cal:                   Calibration .mat file for the specific Hyper-a instrument
%   pureWater:             Hyper-a measurement with pure water in the sphere (file path or struct)
%   T_pureWater:           Water temperature in Celsius of the pure water measurement
%   S_pureWater:           Salinity (PSU) of the pure water measurement (typically 0)
%   sample:                Hyper-a measurement with unknown water sample in the sphere (file path or struct)
%   T_Sample:              Water temperature in Celsius of the sample measurement  
%   S_Sample:              Salinity (PSU) of the sample measurement
%   RemoveWaterAbsorption: Optional argument to turn off subtraction of pure water absorption
%   ChlFluorCorr:          Optional argument to turn off chlorophyll fluorescence correction
%
% Outputs:
%   hypera_proc:  Table of calibrated hyper-a absorption values (1/m) as
%                 well as metadata. The absorption coefficient (1/m) in 
%                 the table has absorption by water removed. Wavelengths 
%                 and instrument configuration information are stored in 
%                 the table properties (hypera_proc.Properties.CustomProperties).
%
%  T_AB:          Optional output of transmission (intermediate processing step)
%
% Sequoia Scientific, Inc.
% v2.0 
% 11/03/2025

function [hypera_proc, T_AB] = Hypera_Process(cal, purewater, T_PureWater, S_PureWater, sample, T_Sample, S_Sample, options)

    arguments
        cal
        purewater
        T_PureWater
        S_PureWater
        sample
        T_Sample
        S_Sample
        options.RemoveWaterAbsorption logical = true;
        options.ChlFluorCorr logical = true  
    end

%% import data

sample    = Hypera_lib.ImportHyperaData(sample);
purewater = Hypera_lib.ImportHyperaData(purewater);

%% linearly correct spectrometer pixels

sample    = Hypera_lib.linearityCorrectPixels(sample);
purewater = Hypera_lib.linearityCorrectPixels(purewater);

%% median bin dark and chl short pass filter measurements

sample    = Hypera_lib.GetMedianOfFilterRuns(sample,    [Hypera_lib.DarkRecordID Hypera_lib.ChlaFilterRecordIDs]);
purewater = Hypera_lib.GetMedianOfFilterRuns(purewater, [Hypera_lib.DarkRecordID Hypera_lib.ChlaFilterRecordIDs]);

%% subtract dark

sample    = Hypera_lib.DarkCorrectSpectrum(sample);
purewater = Hypera_lib.DarkCorrectSpectrum(purewater);

%% interpolate onto cal wavelengths

sample    = Hypera_lib.InterpolatePixelsToCalWls(cal, sample);
purewater = Hypera_lib.InterpolatePixelsToCalWls(cal, purewater);

%% compute chl fluoresence correction

if options.ChlFluorCorr
    f_fluor = Hypera_lib.ComputeChlFluorescenceCorrection(cal, sample, purewater);
else
    f_fluor = [];
end

%% calculate transmission

T_AB = Hypera_lib.ComputeTransmission(sample, purewater, f_fluor);

%% compute absorption, with pure water as reference

aW_ref = Hypera_lib.Get_IOCCG_aW(cal.wl,T_PureWater,S_PureWater);
a_hypera = Hypera_lib.ComputeAbsorption(cal, aW_ref, T_AB);

%% remove absorption by water

if options.RemoveWaterAbsorption
    aW_sample = Hypera_lib.Get_IOCCG_aW(cal.wl,T_Sample,S_Sample);
    a_hypera = a_hypera - repmat(aW_sample,height(a_hypera),1);
end

%% merge results

% only keep no filter data
noFilterIdx = sample.recordID == Hypera_lib.NoFilterRecordID;
T_AB    (~noFilterIdx,:) = [];
a_hypera(~noFilterIdx,:) = [];
sample  (~noFilterIdx,:) = [];

% merge with original meta data
hypera_proc = addvars(sample(:, 2:7), a_hypera, 'NewVariableNames', 'absorption');

% save wavelengths and cal in the table properties
hypera_proc = addprop(hypera_proc, {'wavelengths', 'cal'}, {'table', 'table'});
hypera_proc.Properties.CustomProperties.wavelengths = cal.wl;
hypera_proc.Properties.CustomProperties.cal = cal;

end

