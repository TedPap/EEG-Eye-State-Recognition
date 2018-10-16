clc; 
clear all; 
close all; 

load("EEG_Eye_State.mat");

x = data(1:14, :);
[dx, dy] = size(data);

figure()
plot(data(1, :));
hold on;
plot(data(2, :));
hold on;
plot(data(3, :));
hold on;
plot(data(4, :));
hold on;
plot(data(5, :));
hold on;
plot(data(6, :));
hold on;
plot(data(7, :));
hold on;
plot(data(8, :));
hold on;
plot(data(9, :));
hold on;
plot(data(10, :));
hold on;
plot(data(11, :));
hold on;
plot(data(12, :));
hold on;
plot(data(13, :));
hold on;
plot(data(14, :));
title('EEG EyeState All signals');

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
x = data(1:14, :);
[dx, dy] = size(data);

figure()
plot(data(1, :));
hold on;
plot(data(2, :));
hold on;
plot(data(3, :));
hold on;
plot(data(4, :));
hold on;
plot(data(5, :));
hold on;
plot(data(6, :));
hold on;
plot(data(7, :));
hold on;
plot(data(8, :));
hold on;
plot(data(9, :));
hold on;
plot(data(10, :));
hold on;
plot(data(11, :));
hold on;
plot(data(12, :));
hold on;
plot(data(13, :));
hold on;
plot(data(14, :));
title('EEG EyeState - Remove outliers');

clear i M sd sdWeight Thresh idx idxFinal idxSize;

%% Variance computation

row_dataVar = var(x, 0, 1);
Variance = zeros(2, dy);
Variance(1, :) = row_dataVar(1, :);
Variance(2, :) = data(15, :);

figure();
plot(Variance(1, :));
title('Variance');

clear dataVar

%% Skewness computation

dataSkew = skewness(x);
Skewness = zeros(2, dy);
Skewness(1, :) = dataSkew(1, :);
Skewness(2, :) = data(15, :);

figure();
plot(Skewness(1, :));
title('Skewness');

clear dataSkew

%% Kurtosis computation

dataKurt = kurtosis(x);
Kurtosis = zeros(2, dy);
Kurtosis(1, :) = dataKurt(1, :);
Kurtosis(2, :) = data(15, :);

figure();
plot(Kurtosis(1, :));
title('Kurtosis');

clear dataKurt

%% Divide electrodes into groups as suggested in the paper
dataLeft = data([1 4 5 6 7 13 14 15], :);
dataRight = data([2 3 8 9 10 11 12 15], :);

xL = dataLeft(1:7, :);
xR = dataRight(1:7, :);

figure()
plot(dataLeft(1, :));
hold on;
plot(dataLeft(2, :));
hold on;
plot(dataLeft(3, :));
hold on;
plot(dataLeft(4, :));
hold on;
plot(dataLeft(5, :));
hold on;
plot(dataLeft(6, :));
hold on;
plot(dataLeft(7, :));
title('EEG EyeState - Electrodes of left group');

figure()
plot(dataRight(1, :));
hold on;
plot(dataRight(2, :));
hold on;
plot(dataRight(3, :));
hold on;
plot(dataRight(4, :));
hold on;
plot(dataRight(5, :));
hold on;
plot(dataRight(6, :));
hold on;
plot(dataRight(7, :));
title('EEG EyeState - Electrodes of right group');

%% Left Variance computation

dataVarL = var(xL, 0, 1);
VarianceL = zeros(2, dy);
VarianceL(1, :) = dataVarL(1, :);
VarianceL(2, :) = data(15, :);

figure();
plot(VarianceL(1, :));
title('Variance Left');

clear dataVarL

%% Left Skewness computation

dataSkewL = skewness(xL);
SkewnessL = zeros(2, dy);
SkewnessL(1, :) = dataSkewL(1, :);
SkewnessL(2, :) = data(15, :);

figure();
plot(SkewnessL(1, :));
title('Skewness Left');

clear dataSkewL

%% Left Kurtosis computation

dataKurtL = kurtosis(xL);
KurtosisL = zeros(2, dy);
KurtosisL(1, :) = dataKurtL(1, :);
KurtosisL(2, :) = data(15, :);

figure();
plot(KurtosisL(1, :));
title('Kurtosis Left');

clear dataKurtL

%% Right Variance computation

dataVarR = var(xR, 0, 1);
VarianceR = zeros(2, dy);
VarianceR(1, :) = dataVarR(1, :);
VarianceR(2, :) = data(15, :);

figure();
plot(VarianceR(1, :));
title('Variance Right');

clear dataVarR

%% Right Skewness computation

dataSkewR = skewness(xR);
SkewnessR = zeros(2, dy);
SkewnessR(1, :) = dataSkewR(1, :);
SkewnessR(2, :) = data(15, :);

figure();
plot(SkewnessR(1, :));
title('Skewness Right');

clear dataSkewR

%% Right Kurtosis computation

dataKurtR = kurtosis(xR);
KurtosisR = zeros(2, dy);
KurtosisR(1, :) = dataKurtR(1, :);
KurtosisR(2, :) = data(15, :);

figure();
plot(KurtosisR(1, :));
title('Kurtosis Right');

clear dataKurtR

%% Row Variance computation

row_dataVar = var(x, 0, 2);
row_Variance = zeros(dx-1, 1);
row_Variance(:, 1) = row_dataVar(:, 1);

figure();
plot(row_Variance(:,1));
title('Row Variance');

clear row_dataVar

mean_row_var = mean(row_Variance);

%% Data with variance, skewness and kurtosis

data_all = data;
data_all (16, :)= Variance(1,:);
data_all (17, :)= Skewness(1,:);
data_all (18, :)= Kurtosis(1,:);
data_all (19, :)= VarianceL(1,:);
data_all (20, :)= SkewnessL(1,:);
data_all (21, :)= KurtosisL(1,:);
data_all (22, :)= VarianceR(1,:);
data_all (23, :)= SkewnessR(1,:);
data_all (24, :)= KurtosisR(1,:);
data_all (25, :)= row_Variance(1,:);