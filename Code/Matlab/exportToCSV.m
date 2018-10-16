clear all
close all

addpath('/media/tedpap/092C42C166B32A14/SM&PR Project/Code/tftb-0.2/mfiles')

load("EEG_Eye_State.mat");

%% --------------------- Remove Outliers Start ---------------------
x = data(1:14, :);
[dx, dy] = size(data);

idxSize = 0;
for i=1:14
    %Detect Outliers
    M=median(x(i, :)); % The median value from samples
    sd=std(x(i, :)); % Standard Deviation from samples
    sdWeight=1; % Threshold weight
    Thresh=sdWeight*sd; % Threshold
    
    % Values above threshold
    idx=find(x(i, :)>M+Thresh | (x(i, :)<M-Thresh));
    if size(idx) > idxSize
        idxSize = size(idx);
        idxFinal = idx;
    end
end
%Remove columns with outliers from dataset
data(:, idxFinal) = [];
clear M sd sdWeight Thresh idx idxSize idxFinal dx dy;
% --------------------- Remove Outliers End ---------------------

csvwrite("EEG_Eye_State.csv", data');