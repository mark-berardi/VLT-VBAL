% Load stereo audio file
[audioData, fs] = audioread('babble_2.wav');

% Load Reference HRTF
load('ReferenceHRTF.mat', 'hrtfData', 'sourcePosition');
hrtfData = permute(double(hrtfData),[2,3,1]);
sourcePosition = sourcePosition(:,[1,2]);

% Define positions
leftAz = -45;   % left channel position
rightAz = 45;   % right channel position
elevation = 0;

% Interpolate HRTF
leftIR = interpolateHRTF(hrtfData, sourcePosition, [leftAz elevation], ...
    Algorithm="vbap");
rightIR = interpolateHRTF(hrtfData, sourcePosition, [rightAz elevation], ...
    Algorithm="vbap");

% Create spatial filters
leftFilter = dsp.FrequencyDomainFIRFilter(squeeze(leftIR), ...
    SumFilteredOutputs=false);
rightFilter = dsp.FrequencyDomainFIRFilter(squeeze(rightIR), ...
    SumFilteredOutputs=false);

% Apply spatialization and sum appropriately
leftProcessed = leftFilter(audioData(:,1));    % [left_to_left, left_to_right]
rightProcessed = rightFilter(audioData(:,2));  % [right_to_left, right_to_right]

% Combine the contributions
finalLeft = leftProcessed(:,1) + rightProcessed(:,1);   % Sum left-ear contributions
finalRight = leftProcessed(:,2) + rightProcessed(:,2);  % Sum right-ear contributions

% Distance attenuation
distance = 1;
attenuation = 1/distance;
spatialized = [finalLeft finalRight] * attenuation;

% Normalize
spatialized = spatialized / max(abs(spatialized(:)));

% Write output
audiowrite('babble_spatialized_stereo.wav', spatialized, fs);