%% Hyper-a Library of Functions
% Core functions used for processing Hyper-a data. Computation of
% absorption follows procedure described in:
%
% Röttgers, R., W. Schönfeld, P.-R. Kipp, and R. Doerffer, 2005: Practical test of a point-source integrating
% cavity absorption meter: the performance of different collector assemblies. Appl. Opt., 44: 5549–5560.
%
% Sequoia Scientific, Inc.
% v2.0
% 11/03/2025

classdef Hypera_lib

    properties (Constant)
        NoFilterRecordID = 10;
        DarkRecordID = 999;
        ChlaFilterRecordIDs = [601];
        ChlaSPFWavelengthRange = [305 595];
    end

    methods(Static)

        %% Absorption Calculations

        % Compute transmission I/I0
        function T_AB = ComputeTransmission(sample, purewater, f_fluor)

            sample_sig = sample.sigPix;

            % apply chl fluorescence correction, if provided
            if ~isempty(f_fluor)
                NoFilterIdx = sample.recordID == Hypera_lib.NoFilterRecordID;
                sample_sig(NoFilterIdx,:) = sample.sigPix(NoFilterIdx,:) - f_fluor;
            end

            sample_refCorr    =       sample_sig ./ sample.refPix;
            pureWater_refCorr = purewater.sigPix ./ purewater.refPix;

            IDs = unique(sample.recordID);
            T_AB = nan(height(sample),length(sample.sigPix));
            for id = IDs(:)'

                idx_sampleID = sample.recordID == id;
                idx_purewaterID = purewater.recordID == id;

                % Compute median purewater for this ID
                pureWater_refCorr_median = median(pureWater_refCorr(idx_purewaterID, :), 1);

                T_AB(idx_sampleID,:) = sample_refCorr(idx_sampleID, :) ./ pureWater_refCorr_median;

            end
        end

        % Computes Absorption From Hyper-a Data Given a Known Reference, Rho, and Sample
        function a_tot_hypera = ComputeAbsorption(cal, aW_ref, T_AB)

            r = cal.r;
            r_0 = cal.r_0;
            rho = cal.rho;

            T_AB = smoothdata(T_AB,2,'sgolay',15);

            a_tot_hypera = nan(size(T_AB));
            for i_wl = 1:length(aW_ref)

                T_AB_guess = @(a_guess) ...
                    exp(-r_0 .* (a_guess - aW_ref(i_wl))) ...
                    .* ((1 - rho(i_wl) .* Hypera_lib.Ps(aW_ref(i_wl),r)) ./ (1 - rho(i_wl) .* Hypera_lib.Ps(a_guess,r)) ...
                    .* Hypera_lib.Ps(a_guess,r) ./ Hypera_lib.Ps(aW_ref(i_wl),r));

                for i_meas = 1:height(T_AB)
                    % minimize absolute percent error
                    ToMinimize = @(a_guess) abs(T_AB_guess(a_guess) - T_AB(i_meas,i_wl))./T_AB(i_meas,i_wl);
                    a_tot_hypera(i_meas,i_wl) = fminsearch(ToMinimize, aW_ref(i_wl));

                end
            end
        end

        % Probablity Function
        function Ps_x = Ps(a,r)

            % Röttgers et al. 2005 - Equation 5
            Ps_x = (1 - exp(-2. * a .* r) .* (2 .* a .* r + 1)) ...
                ./ (2 .* a.^2 .* r.^2);
        end

        % Computes Theoretical Absorption For Pure Water
        function aW_TS = Get_IOCCG_aW(wls,T,S)

            aW_IOCCG.wl = [250;255;260;265;270;275;280;285;290;295;300;305;310;315;320;325;330;335;340;345;350;355;360;365;370;375;380;385;390;395;400;405;410;415;420;425;430;435;440;445;450;455;460;465;470;475;480;485;490;495;500;505;510;515;520;525;530;535;540;545;550;555;560;565;570;575;580;585;590;595;600;605;610;615;620;625;630;635;640;645;650;655;660;665;670;675;680;685;690;695;700;705;710;715;720;725;730;735;740;745;750;755;760;765;770;775;780;785;790;795;800;805;810;815;820;825;830;835;840;845;850;855;860;865;870;875;880;885;890;895;900];
            aW_IOCCG.aW = [0.0450000000000000;0.0392000000000000;0.0344000000000000;0.0303000000000000;0.0269000000000000;0.0240000000000000;0.0216000000000000;0.0194000000000000;0.0176000000000000;0.0160000000000000;0.0147000000000000;0.0134000000000000;0.0124000000000000;0.0114000000000000;0.0106000000000000;0.00980000000000000;0.00920000000000000;0.00850000000000000;0.00800000000000000;0.00750000000000000;0.00710000000000000;0.00680000000000000;0.00660000000000000;0.00630000000000000;0.00600000000000000;0.00560000000000000;0.00520000000000000;0.00500000000000000;0.00480000000000000;0.00470000000000000;0.00460000000000000;0.00460000000000000;0.00460000000000000;0.00460000000000000;0.00454000000000000;0.00478000000000000;0.00495000000000000;0.00530000000000000;0.00635000000000000;0.00751000000000000;0.00922000000000000;0.00962000000000000;0.00979000000000000;0.0101100000000000;0.0106000000000000;0.0114000000000000;0.0127000000000000;0.0136000000000000;0.0150000000000000;0.0173000000000000;0.0204000000000000;0.0256000000000000;0.0325000000000000;0.0396000000000000;0.0409000000000000;0.0417000000000000;0.0434000000000000;0.0452000000000000;0.0474000000000000;0.0511000000000000;0.0565000000000000;0.0596000000000000;0.0619000000000000;0.0642000000000000;0.0695000000000000;0.0772000000000000;0.0896000000000000;0.110000000000000;0.135100000000000;0.167200000000000;0.222400000000000;0.257700000000000;0.264400000000000;0.267800000000000;0.275500000000000;0.283400000000000;0.291600000000000;0.301200000000000;0.310800000000000;0.325000000000000;0.340000000000000;0.371000000000000;0.410000000000000;0.429000000000000;0.439000000000000;0.448000000000000;0.465000000000000;0.486000000000000;0.516000000000000;0.559000000000000;0.624000000000000;0.704000000000000;0.827000000000000;1.00700000000000;1.23100000000000;1.48900000000000;1.97000000000000;2.51000000000000;2.78000000000000;2.83000000000000;2.85000000000000;2.88000000000000;2.86000000000000;2.86000000000000;2.82000000000000;2.76000000000000;2.69000000000000;2.59000000000000;2.47000000000000;2.36000000000000;2.25000000000000;2.20000000000000;2.19000000000000;2.23000000000000;2.34000000000000;2.61000000000000;3.22000000000000;3.72000000000000;3.94000000000000;4.09000000000000;4.20000000000000;4.32000000000000;4.60000000000000;4.60000000000000;4.77000000000000;5.01000000000000;5.28000000000000;5.57000000000000;5.85000000000000;6.13000000000000;6.40000000000000];
            aW_IOCCG.psiT = 10.^-4 .* [3;3;2;2;2;2;2;1;1;1;1;1;1;1;1;0;0;0;1;0;-1;-1;0;0;-0.700000000000000;-1;0;0;0;-0.200000000000000;0.100000000000000;0.100000000000000;0;0.200000000000000;0;-0.100000000000000;-0.100000000000000;0;0;0.100000000000000;0.200000000000000;0.100000000000000;0.100000000000000;0;-0.100000000000000;-0.100000000000000;0;-0.100000000000000;-0.100000000000000;0;0.100000000000000;0.300000000000000;0.800000000000000;1.20000000000000;1.10000000000000;0.700000000000000;0.400000000000000;0.100000000000000;0;-0.100000000000000;0;-0.200000000000000;-0.400000000000000;-0.600000000000000;-0.700000000000000;-0.600000000000000;0;1.20000000000000;2.50000000000000;4.50000000000000;7.90000000000000;10.3000000000000;9.50000000000000;7.20000000000000;5.40000000000000;3;0.900000000000000;-0.500000000000000;-2;-3;-3;-2.20000000000000;0.800000000000000;0.900000000000000;-0.400000000000000;-2.10000000000000;-4;-4;-4.30000000000000;-4.10000000000000;-2;1.70000000000000;11;27.3000000000000;46.3000000000000;65.8000000000000;98.3000000000000;148.500000000000;161;137.200000000000;105;74.4000000000000;44.7000000000000;18.6000000000000;-4.40000000000000;-24.5000000000000;-40.4000000000000;-52.1000000000000;-59.4000000000000;-62;-60;-52;-38.4000000000000;-20;0;33;101;153;145;115;83;49;27;-0.800000000000000;-46;-63;-78;-87;-90;-85;-70];
            aW_IOCCG.psiS = 10.^-4 .* [0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.430000000000000;0.370000000000000;0.360000000000000;0.340000000000000;0.320000000000000;0.280000000000000;0.260000000000000;0.250000000000000;0.220000000000000;0.190000000000000;0.170000000000000;0.160000000000000;0.140000000000000;0.130000000000000;0.110000000000000;0.0900000000000000;0.0800000000000000;0.0600000000000000;0.0600000000000000;0.0500000000000000;0.0500000000000000;0.0400000000000000;0.0200000000000000;0.0800000000000000;0.130000000000000;0.140000000000000;0.150000000000000;0.150000000000000;0.150000000000000;0.130000000000000;0.130000000000000;0.160000000000000;0.160000000000000;0.150000000000000;0.130000000000000;0.100000000000000;0.0400000000000000;0.0100000000000000;0.0300000000000000;-0.0200000000000000;-0.160000000000000;0.430000000000000;0.750000000000000;0.830000000000000;0.840000000000000;0.800000000000000;0.770000000000000;0.730000000000000;0.700000000000000;0.660000000000000;0.600000000000000;0.340000000000000;0.410000000000000;0.630000000000000;0.620000000000000;0.460000000000000;0.250000000000000;-0.0200000000000000;-0.340000000000000;-0.700000000000000;-1.16000000000000;-1.40000000000000;-1.80000000000000;-1.90000000000000;-1.80000000000000;-2.10000000000000;-4.40000000000000;-3.70000000000000;1.80000000000000;4.70000000000000;6.50000000000000;6.80000000000000;6.60000000000000;6;5.30000000000000;4.50000000000000;3.60000000000000;2.30000000000000;1.10000000000000;-0.100000000000000;-1.30000000000000;-2.80000000000000;-4.40000000000000;-5.20000000000000;-6.20000000000000;-9;-13;-8;-2;0;0;0;-2;-6;-8;-12;-16;-20;-22;-23;-25];

            if isempty(wls)
                wls =  aW_IOCCG.wl;
            end

            T = T(:);
            S = S(:);

            aW = interp1(aW_IOCCG.wl, aW_IOCCG.aW, wls);
            psiT = interp1(aW_IOCCG.wl, aW_IOCCG.psiT, wls);
            psiS = interp1(aW_IOCCG.wl, aW_IOCCG.psiS, wls);
            aW_TS = aW + (T - 22).*psiT + (S - 0).*psiS; % for  temperature, 22 is the reference; for salinity, 0 is the reference

        end

        %% Sphere Reflectivity (rho) Calculation

        % Computes Reflectivity of the Sphere Given Two Solutions with Known Absorption
        function rho = ComputeRho(cal, a_A_known, a_B_known, T_AB)

            r = cal.r;
            r_0 = cal.r_0;

            a_A = a_A_known;
            a_B = a_B_known;
            T_AB = smoothdata(T_AB,2,'sgolay',15);

            % Röttgers et al. 2005 - Equation 10
            rho = (T_AB .* exp(-a_B .* r_0) .* Hypera_lib.Ps(a_B,r) - exp(-a_A .* r_0) .* Hypera_lib.Ps(a_A,r)) ...
                ./ (T_AB .* exp(-a_B .* r_0) .* Hypera_lib.Ps(a_A,r) .* Hypera_lib.Ps(a_B,r) - exp(-a_A .* r_0) .* Hypera_lib.Ps(a_B,r) .* Hypera_lib.Ps(a_A,r));

        end

        %% Spectrometer Corrections

        % Apply spectrometer linearity correction
        function linearityCorrected = linearityCorrectPixels(dataTable)

            config = dataTable.Properties.CustomProperties.config;

            if isfield(config,'sigSpecLinCoeff')
                % apply linearity correction
                dataTable.sigPix = dataTable.sigPix ./ polyval(flipud(config.sigSpecLinCoeff),dataTable.sigPix);
                dataTable.refPix = dataTable.refPix ./ polyval(flipud(config.refSpecLinCoeff),dataTable.refPix);
            else
                warning('No linearity coeffiencts found in configuration, linearity correction not applied.')
            end

            linearityCorrected = dataTable;
        end


        % Removes dark measurements from Hyper-a spectrometer data
        function DarkCorrected = DarkCorrectSpectrum(dataTable)

            darkRecordIdx = dataTable.recordID == Hypera_lib.DarkRecordID;
            darkData = dataTable(darkRecordIdx,:);
            dataTable(darkRecordIdx,:) = [];

            for n = 1:height(dataTable)

                % find closest dark measurement in time
                [~,closestIndex] = min(abs(dataTable.date(n) - darkData.date));

                % subtract dark
                dataTable.sigPix(n,:) = dataTable.sigPix(n,:) - darkData.sigPix(closestIndex,:);
                dataTable.refPix(n,:) = dataTable.refPix(n,:) - darkData.refPix(closestIndex,:);
            end

            DarkCorrected = dataTable;
        end

        % Interpolate onto calibration wavelenghts
        function interpData = InterpolatePixelsToCalWls(cal, dataTable)

            config = dataTable.Properties.CustomProperties.config;

            % Interplate to cal wavelengths. Cal wls are typically the sig wavelengths
            dataTable.sigPix = interp1(config.sigWls(:)', dataTable.sigPix', cal.wl(:)')';
            dataTable.refPix = interp1(config.refWls(:)', dataTable.refPix', cal.wl(:)')';

            interpData = dataTable;
        end

        %% Fluorescence Corrections

        % Calculates chlorophyll fluorescence correction (IOCCG 2018)
        function f_fluor = ComputeChlFluorescenceCorrection(cal, sample, purewater)

            purewater_NoFilter_Idx = purewater.recordID == Hypera_lib.NoFilterRecordID;
            purewater_SPF_Idx =  any(purewater.recordID == Hypera_lib.ChlaFilterRecordIDs, 2);

            purewater_NoFilter_sig = median(purewater.sigPix(purewater_NoFilter_Idx,:), 1);
            purewater_NoFilter_ref = median(purewater.refPix(purewater_NoFilter_Idx,:), 1);
            purewater_SPF_sig      = median(purewater.sigPix(purewater_SPF_Idx, :), 1);
            purewater_SPF_ref      = median(purewater.refPix(purewater_SPF_Idx, :), 1);

            sample_NoFilter_Idx = sample.recordID==Hypera_lib.NoFilterRecordID;
            sample_NoFilter_sig = sample.sigPix(sample_NoFilter_Idx,:);
            sample_NoFilter_ref = sample.refPix(sample_NoFilter_Idx,:);

            % extract SPF chl measurements
            sample_SPF = sample(any(sample.recordID == Hypera_lib.ChlaFilterRecordIDs , 2), :);

            % Mask for wavelenths outside the range of the chl SPF
            ChlaExcitationWls = cal.wl > Hypera_lib.ChlaSPFWavelengthRange(1) & ...
                                cal.wl < Hypera_lib.ChlaSPFWavelengthRange(2);

            f_fluor = nan(height(sample_NoFilter_sig),length(cal.wl));
            for n = 1:height(sample_NoFilter_sig)

                % find closest SPF measurement in time
                [~,closestIndex] = min(abs(sample.date(n) - sample_SPF.date));

                sample_SPF_sig = sample_SPF.sigPix(closestIndex,:);
                sample_SPF_ref = sample_SPF.refPix(closestIndex,:);

                % calculate scaling factor for to account for changes in
                % lamp output
                S =     sample_NoFilter_ref(n,:) ./ purewater_NoFilter_ref;
                S_SPF = sample_SPF_ref           ./ purewater_SPF_ref;
                S_SPF(:,~ChlaExcitationWls) = 0;

                % compute total absorbed light for No Filter and SPF measurements
                a_NoFilter = sum(S                          .* purewater_NoFilter_sig               - sample_NoFilter_sig(n,:)           , 2);
                a_SPF =      sum(S_SPF(:,ChlaExcitationWls) .* purewater_SPF_sig(ChlaExcitationWls) - sample_SPF_sig(:,ChlaExcitationWls), 2);

                % R_F: Scaling factor to account for change in illumination conditions between No Filter an SPF measurements.
                % 0.1 determined empirically for PSICAM
                R_f = (a_NoFilter ./ a_SPF) + 0.1; 

                % ref corrected sample / purewater
                T_AB = (sample_NoFilter_sig(n,:) ./ sample_NoFilter_ref(n,:)) ./ median(purewater.sigPix(purewater_NoFilter_Idx,:) ./ purewater.refPix(purewater_NoFilter_Idx,:));

                % Scale SPF measured fluorescence to No Filter illumination conditions 
                f_fluor(n,:) = R_f .* (sample_SPF_sig - purewater_SPF_sig .* T_AB);

            end

             f_fluor(:,cal.wl<660) = 0;

        end

        %% Data Handling Functions

        % Import from file or variable, configure dataTable for processing
        function dataTable = ImportHyperaData(pathOrVariable)

            % Load data from file or use provided struct
            if isstring(pathOrVariable) || ischar(pathOrVariable)
                if endsWith(pathOrVariable, '.bin')
                    data = Hypera_ReadBin(pathOrVariable);
                elseif endsWith(pathOrVariable, '.mat')
                    data = load(pathOrVariable);
                else
                    error('File must be .bin or .mat');
                end
            else
                data = pathOrVariable;
            end

            assert(isfield(data, 'config') && isfield(data, 'dataTable'), ...
                'Data must contain config and dataTable fields');

            dataTable = data.dataTable;
            config = data.config;

            % Convert cell arrays
            if iscell(dataTable)
                dataTable = vertcat(dataTable{:});
            end
            if iscell(config)
                config = config{1};
            end

            % Attach config to table properties
            dataTable = addprop(dataTable, {'config'}, {'table'});
            dataTable.Properties.CustomProperties.config = config;
        end


        % Calculate median for specified recordIDs, keep original data for others
        function medianData = GetMedianOfFilterRuns(dataTable, recordIDsToMedian)

            % Identify consecutive runs
            runID = cumsum([1; diff(dataTable.recordID) ~= 0]);

            % Process each run
            uniqueRuns = unique(runID);
            results = cell(length(uniqueRuns), 1);

            for i = 1:length(uniqueRuns)
                runMask = (runID == uniqueRuns(i));
                runData = dataTable(runMask, :);

                % Check if this recordID should be medianed
                if ismember(runData.recordID(1), recordIDsToMedian)
                    % Calculate median for this run - take median of each column
                    medianRow = varfun(@median, runData);

                    % Fix column names
                    medianRow.Properties.VariableNames = dataTable.Properties.VariableNames;

                    results{i} = medianRow;
                else
                    % Keep original data
                    results{i} = runData;
                end
            end

            % Combine all results
            medianData = vertcat(results{:});
        end

    end
end