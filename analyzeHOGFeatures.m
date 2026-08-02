clear;
clc;

%% Load extracted features and fixed split

load("initial_HOG_features.mat");

%%

%use only the training data
trainMask = dataSplit == "train";
XTrainHOG = X_HOG(trainMask, :);
YTrain = Y(trainMask);

fprintf("Analyzing %d training images with %d HOG features\n", size(XTrainHOG, 1), size(XTrainHOG, 2));

%remove constant dimensions
featureStandardDeviations = std(XTrainHOG, 0, 1);
keepFeatures = featureStandardDeviations > 0;

fprintf("Constant HOG dimensions removed: %d\n", sum(~keepFeatures));

XTrainHOG = XTrainHOG(:, keepFeatures);

%standardize
XTrainHOGStandardized = zscore(XTrainHOG);

%% PCA

[coefficients, scores, latent, ~, explained] = pca(XTrainHOGStandardized);

% PCA scatter plot
figure;
gscatter( ...
    scores(:, 1), ...
    scores(:, 2), ...
    YTrain);

xlabel(sprintf("PC1 (%.1f%% variance)", explained(1)));
ylabel(sprintf("PC2 (%.1f%% variance)", explained(2)));
title("PCA of training-set HOG features");
grid on;


%% Cumulative variance

cumulativeExplained = cumsum(explained);

figure;
plot( ...
    1:numel(cumulativeExplained), ...
    cumulativeExplained, ...
    "LineWidth", 1.5);

xlabel("Number of principal components");
ylabel("Cumulative variance explained (%)");
title("Cumulative PCA variance for HOG features");
yline(90, "--", "90%");
grid on;
