% Hypera_ReadBin reads the binary (.bin) file produced by the Hyper-a
% instrument
%
% Outputs:
%   dataTable:  A table where each row corresponds to one measurement.
%               Columns in the table are named, for easy identification
%   config:     A structure containing information about how the instrument
%               was configured.
%
% Sequoia Scientific, Inc.
% v2.0 
% 11/03/2025

function dataConfig = Hypera_ReadBin(filename)

%% Open data file
fid = fopen(filename);
fseek(fid, 0, 'eof');
fileSize_bytes = ftell(fid);
fseek(fid, 0, 'bof');

%% Parse header
config.serialNumber = fread(fid,1,'uint16');
config.firmwareVer  = fread(fid,1,'uint16') / 100;
config.sigSpecSN    = fread(fid,1,'uint32');
config.sefSpecSN    = fread(fid,1,'uint32');
config.sigNumWls    = fread(fid,1,'uint16');
config.refNumWls    = fread(fid,1,'uint16');
config.configByte   = fread(fid,1,'uint8');
config.mainBoardRev = fread(fid,1,'*char');
config.pumpFlushSec = fread(fid,1,'uint16');

if config.firmwareVer >= 1.07 
    % added in v1.07
    config.sequenceInterval_sec = fread(fid,1,'uint16');
	config.burstInterval_min = fread(fid,1,'uint16');
	config.sequencesPerBurst = fread(fid,1,'uint16');
    
    if config.firmwareVer >= 1.09
        % added in v1.09
        config.fileInterval_hours = fread(fid,1,'uint16');
        config.sigSpecLinCoeff = fread(fid,8,'float'); 
        config.refSpecLinCoeff = fread(fid,8,'float'); 
    else
       fread(fid,1,'uint16'); % unassigned, used to preserve memory alignment
    end        
end

for n=1:8
   config.measName(n,:) = deblank(convertCharsToStrings(fread(fid,16,'*char'))); 
end
config.measID       = fread(fid,8,'uint16'); 
config.sigWls       = fread(fid,config.sigNumWls,'uint32') / 1000;
config.refWls       = fread(fid,config.refNumWls,'uint32') / 1000;

config.autoExposure = bitget(config.configByte,1);
config.sigLinCorr   = bitget(config.configByte,2);
config.refLinCorr   = bitget(config.configByte,3);
config.pumpEnabled  = bitget(config.configByte,4);
config.autoStart    = bitget(config.configByte,5);
config.switchStart  = bitget(config.configByte,6);

%% Preallocate table
tableVars = {
    'recordID'    , 'double';
    'date'        , 'datetime';
    'inputVoltage', 'double';
    'waterTemp'   , 'double';
    'depth'       , 'double';
    'boardTemp'   , 'double';
    'boardHumid'  , 'double';
    'lampFreq'    , 'double';
    'sigAvg'      , 'double';
    'refAvg'      , 'double';
    'sigExp'      , 'double';
    'refExp'      , 'double';
    'sigNumFlash' , 'double';
    'refNumFlash' , 'double';
    'sigPix'      , 'double';
    'refPix'      , 'double';
    };

dataBytesInFile = fileSize_bytes - ftell(fid);
dataRecordBytes = 32 + config.sigNumWls * 2 + config.refNumWls * 2;
numDataRecords = dataBytesInFile / dataRecordBytes;

% Number of data records should be an integer
if floor(numDataRecords) ~= numDataRecords
   warning('File contains incomplete data records');
   numDataRecords = floor(numDataRecords);
end

dataTable = table('Size',[numDataRecords height(tableVars)], ... 
                  'VariableNames', tableVars(:,1), ...
                  'VariableTypes', tableVars(:,2));

%% Parse data records
for dataRecord = 1:numDataRecords
        
dataTable.recordID(dataRecord)     = fread(fid,1,'uint16');
dataTable.date(dataRecord)         = datetime(fread(fid,6,'uint8')') + calyears(1900);
dataTable.inputVoltage(dataRecord) = fread(fid,1,'uint16') / 100;
dataTable.waterTemp(dataRecord)    = fread(fid,1,'uint16') / 100 - 10;
dataTable.depth(dataRecord)        = fread(fid,1,'uint16') / 100 - 10;
dataTable.boardTemp(dataRecord)    = fread(fid,1,'uint16') / 100 - 10;
dataTable.boardHumid(dataRecord)   = fread(fid,1,'uint8');
dataTable.lampFreq(dataRecord)     = fread(fid,1,'uint8');
dataTable.sigAvg(dataRecord)       = fread(fid,1,'uint8');
dataTable.refAvg(dataRecord)       = fread(fid,1,'uint8');
dataTable.sigExp(dataRecord)       = fread(fid,1,'uint32');
dataTable.refExp(dataRecord)       = fread(fid,1,'uint32');
dataTable.sigNumFlash(dataRecord)  = fread(fid,1,'uint16');
dataTable.refNumFlash(dataRecord)  = fread(fid,1,'uint16');

dataTable.sigPix(dataRecord,1:config.sigNumWls) = fread(fid,config.sigNumWls,'uint16');
dataTable.refPix(dataRecord,1:config.refNumWls) = fread(fid,config.refNumWls,'uint16');

end

fclose(fid);

dataConfig.config = config;
dataConfig.dataTable = dataTable;
