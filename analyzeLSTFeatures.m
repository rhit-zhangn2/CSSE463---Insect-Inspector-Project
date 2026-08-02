clear;
clc;

%% Load extracted features and fixed split

load("initial_LST_features.mat");

%%

%use only the training data for feature exploration
trainMask = dataSplit == "train";
X = X_LST(trainMask, :);
labels = Y(trainMask);

fprintf("Analyzing %d training images with %d features\n", size(X, 1), size(X, 2));

%check for constant features
featureStd = std(X, 0, 1);
keepFeatures = featureStd > 0;

if any(~keepFeatures)
    fprintf("Removing %d constant features\n", sum(~keepFeatures));
end

X = X(:, keepFeatures);
analysisFeatureNames = featureNamesLST(keepFeatures);
numFeatures = size(X, 2);
classNames = categories(labels);


%% One-way ANOVA and eta-squared for every feature
%
% ANOVA p-value:
%   Tests whether all four class means are equal
%
% Eta-squared:
%   Measures how much of the feature's variation is associated with class
%
% A tiny p-value does not automatically mean a feature is useful, so the
% effect size is used for ranking

pValues = zeros(numFeatures, 1);
etaSquared = zeros(numFeatures, 1);

for featureIndex = 1:numFeatures
    values = X(:, featureIndex);
    pValues(featureIndex) = anova1(values, labels, "off");
    grandMean = mean(values);
    totalSumSquares = sum((values - grandMean).^2);
    betweenClassSumSquares = 0;

    for c = 1:numel(classNames)
        classValues = values(labels == classNames{c});
        betweenClassSumSquares = ...
            betweenClassSumSquares + ...
            numel(classValues) * ...
            (mean(classValues) - grandMean)^2;
    end

    if totalSumSquares > 0
        etaSquared(featureIndex) = betweenClassSumSquares / totalSumSquares;
    else
        etaSquared(featureIndex) = 0;
    end
end

%store and rank results
anovaResults = table( ...
    reshape(analysisFeatureNames, [], 1), ...
    pValues, ...
    etaSquared, ...
    'VariableNames', ...
    {'Feature', 'PValue', 'EtaSquared'});

anovaResults = sortrows( ...
    anovaResults, ...
    "EtaSquared", ...
    "descend");

disp("Top 15 features by eta-squared:");
disp(anovaResults(1:min(15, height(anovaResults)), :));

writetable(anovaResults, "LST_ANOVA_results.csv");


%% Plot the top feature effect sizes

numTopFeatures = min(15, height(anovaResults));
topResults = anovaResults(1:numTopFeatures, :);

figure;

barh(categorical(topResults.Feature), topResults.EtaSquared);

set(gca, "YDir", "reverse");
xlabel("Eta-squared");
ylabel("Feature");
title("Top LST features ranked by class effect size");
grid on;


%% Boxplots for the strongest individual features

numBoxplots = min(4, height(anovaResults));

figure;
tiledlayout(2, 2, ...
    "TileSpacing", "compact", ...
    "Padding", "compact");

for plotIndex = 1:numBoxplots
    featureName = anovaResults.Feature(plotIndex);
    featureColumn = find( ...
        analysisFeatureNames == featureName, ...
        1);

    nexttile;

    boxchart(labels, X(:, featureColumn));

    xlabel("Class");
    ylabel("Feature value");
    title(featureName, "Interpreter", "none");
    grid on;
end

sgtitle("Class distributions of strongest LST features");


%% PCA

%standardize data
XStandardized = zscore(X);
[coefficients, scores, latent, ~, explained] = pca(XStandardized);

% PCA scatter plot
figure;
gscatter( ...
    scores(:, 1), ...
    scores(:, 2), ...
    labels);

xlabel(sprintf("PC1 (%.1f%% variance)", explained(1)));
ylabel(sprintf("PC2 (%.1f%% variance)", explained(2)));
title("PCA of training-set LST features");
grid on;

%% Cumulative PCA variance
cumulativeExplained = cumsum(explained);

figure;
plot( ...
    1:numel(cumulativeExplained), ...
    cumulativeExplained, ...
    "LineWidth", 1.5);

xlabel("Number of principal components");
ylabel("Cumulative variance explained (%)");
title("Cumulative variance explained by LST PCA");
yline(90, "--", "90%");
grid on;

