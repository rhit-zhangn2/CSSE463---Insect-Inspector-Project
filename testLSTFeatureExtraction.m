clear;
clc;

%% Load Data

%Folder containing the four class subfolders
dataDir = fullfile("initial_subset_2000/initial_subset_2000/", "images/");

assert(isfolder(dataDir), "Image folder not found: %s", dataDir);

imds = imageDatastore(dataDir, "IncludeSubfolders",true, "LabelSource","foldernames");

fprintf("Loaded %d image paths\n", numel(imds.Files));
classCounts = countEachLabel(imds);
disp(classCounts);

classes = categories(imds.Labels);
%--------------------------------------------------------------------


%% Configuration

nBlocks = 4;
expectedLSTLength = nBlocks * nBlocks * 6;
numClasses = numel(classes);
exampleFeatures = zeros(numClasses, expectedLSTLength);
numImages = numel(imds.Files);
numFeatures = nBlocks * nBlocks * 6;


%% Test one image from each class

for c = 1:numClasses
    className = classes{c};
    classIndices = find(imds.Labels == className);
    imageIndex = classIndices(1);
    img = readimage(imds, imageIndex);

    [features, featureNames] = ...
        extractLSTGridFeatures(img, nBlocks);

    exampleFeatures(c, :) = features;

    fprintf("\nClass: %s\n", className);
    fprintf("File: %s\n", imds.Files{imageIndex});
    fprintf("Image dimensions: %d x %d x %d\n", ...
        size(img, 1), ...
        size(img, 2), ...
        size(img, 3));
    fprintf("LST feature length: %d\n", numel(features));
    fprintf("NaN values: %d\n", sum(isnan(features)));
    fprintf("Infinite values: %d\n", sum(isinf(features)));
end

%checks
assert(size(exampleFeatures, 2) == expectedLSTLength, "Incorrect feature-vector length");
assert(all(isfinite(exampleFeatures), "all"), "Feature matrix contains NaN or Inf values");
fprintf("\nAll four images produced valid %d-dimensional LST vectors\n", expectedLSTLength);

%inspect first image's first two blocks
previewNames = reshape(featureNames(1:12), [], 1);
previewValues = reshape(exampleFeatures(1, 1:12), [], 1);
previewTable = table( ...
    previewNames, ...
    previewValues, ...
    'VariableNames', {'Feature', 'Value'});

disp(previewTable);


%% Extract from all 2000 images

%allocate feature matrix
X_LST = zeros(numImages, numFeatures);
Y = imds.Labels;
files = string(imds.Files);

featureNamesLST = strings(1, numFeatures);
failed = false(numImages, 1);

%extract features
for i = 1:numImages
    try
        img = readimage(imds, i);

        [features, names] = extractLSTGridFeatures(img, nBlocks);
        X_LST(i, :) = features;
        if i == 1
            featureNamesLST = names;
        end

        %helps check processing progress
        if mod(i, 100) == 0
            fprintf("Processed %d of %d images.\n", i, numImages);
        end

    catch ME
        failed(i) = true;
        fprintf("Failed on image %d: %s\n", i, imds.Files{i});
        fprintf("Reason: %s\n", ME.message);
    end
end

%remove failed rows, if any
X_LST = X_LST(~failed, :);
Y = Y(~failed);
files = files(~failed);

%verify final matrix
fprintf("\nSuccessful images: %d\n", size(X_LST, 1));
fprintf("Failed images: %d\n", sum(failed));
fprintf("Features per image: %d\n", size(X_LST, 2));
assert( ...
    all(isfinite(X_LST), "all"), ...
    "X_LST contains NaN or Inf values.");


%% Split into training, validation, and test;
% using 70% - 15% - 15% split

rng(463);
numImages = numel(Y);

%start every image as unassigned
dataSplit = categorical( ...
    repmat("unassigned", numImages, 1), ...
    ["train", "validation", "test", "unassigned"]);

classes = categories(Y);

for c = 1:numel(classes)
    className = classes{c};
    classIndices = find(Y == className);
    classIndices = classIndices(randperm(numel(classIndices)));

    numClassImages = numel(classIndices);
    numTrain = floor(0.70 * numClassImages);
    numValidation = floor(0.15 * numClassImages);

    trainIndices = classIndices(1:numTrain);
    validationIndices = classIndices(numTrain + 1 : numTrain + numValidation);
    testIndices = classIndices(numTrain + numValidation + 1 : end);

    dataSplit(trainIndices) = "train";
    dataSplit(validationIndices) = "validation";
    dataSplit(testIndices) = "test";
end

splitSummary = groupsummary(table(Y, dataSplit), ["Y", "dataSplit"]);
disp(splitSummary);


%% Save results
save( ...
    "initial_LST_features.mat", ...
    "X_LST", ...
    "Y", ...
    "files", ...
    "featureNamesLST", ...
    "dataSplit", ...
    "nBlocks");




%% --------------------------------------------------------------------
%Extract LST color features from one image
% Inputs:
%   img
%   nBlocks   - number of grid rows and columns, like 7 in Sunset Detector
% Outputs:
%   features
%   featureNames
%   debugInfo - useful for testing and visualization
%
% Each grid block:
%   1. L mean
%   2. L standard deviation
%   3. S mean
%   4. S standard deviation
%   5. T mean
%   6. T standard deviation
function [features, featureNames, debugInfo] = extractLSTGridFeatures(img, nBlocks)
    arguments
        img
        nBlocks (1,1) double {mustBeInteger, mustBePositive}
    end

    %validate image format
    if ndims(img) ~= 3 || size(img, 3) ~= 3
        error("Input image must be an RGB image");
    end

    %convert to double
    img = double(img);
    R = img(:, :, 1);
    G = img(:, :, 2);
    B = img(:, :, 3);

    %calculate LST channels
    L = R + G + B;
    S = R - B;
    T = R - 2 .* G + B;

    imageHeight = size(img, 1);
    imageWidth = size(img, 2);

    %These boundaries divide the entire image into approximately equal
    % relative regions. Every pixel is included, no resizing.
    rowEdges = round(linspace(1, imageHeight + 1, nBlocks + 1));
    colEdges = round(linspace(1, imageWidth + 1, nBlocks + 1));

    %allocate output arrays
    featuresPerBlock = 6; %{L,S,T} x {mean, std}
    numFeatures = nBlocks * nBlocks * featuresPerBlock;
    features = zeros(1, numFeatures);
    featureNames = strings(1, numFeatures);
    featureIndex = 1;

    %extract statistics block by block
    for blockRow = 1:nBlocks

        rowStart = rowEdges(blockRow);
        rowEnd   = rowEdges(blockRow + 1) - 1;

        for blockCol = 1:nBlocks

            colStart = colEdges(blockCol);
            colEnd   = colEdges(blockCol + 1) - 1;

            LBlock = L(rowStart:rowEnd, colStart:colEnd);
            SBlock = S(rowStart:rowEnd, colStart:colEnd);
            TBlock = T(rowStart:rowEnd, colStart:colEnd);

            % Convert each block to a column before calculating std.
            % std(std(block)) would not calculate the overall standard
            % deviation correctly.
            blockFeatures = [
                mean(LBlock(:))
                std(LBlock(:))
                mean(SBlock(:))
                std(SBlock(:))
                mean(TBlock(:))
                std(TBlock(:))
            ];
            features(featureIndex:featureIndex + 5) = blockFeatures;
            prefix = sprintf("r%d_c%d_", blockRow, blockCol);
            featureNames(featureIndex:featureIndex + 5) = [
                prefix + "L_mean"
                prefix + "L_std"
                prefix + "S_mean"
                prefix + "S_std"
                prefix + "T_mean"
                prefix + "T_std"
            ];
            featureIndex = featureIndex + featuresPerBlock;
        end
    end

    %return intermediate values for debugging

    debugInfo.L = L;
    debugInfo.S = S;
    debugInfo.T = T;
    debugInfo.rowEdges = rowEdges;
    debugInfo.colEdges = colEdges;
    debugInfo.imageHeight = imageHeight;
    debugInfo.imageWidth = imageWidth;
end