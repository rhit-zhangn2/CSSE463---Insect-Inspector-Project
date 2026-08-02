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

%% HOG settings

targetSize = [256 256]; %majority of images are 500x375 or 375x500, but the insect is mostly centered
cellSize = [32 32];

numClasses = numel(classes);
numImages = numel(imds.Files);
sampleFeatures = cell(numClasses, 1);
sampleImages = cell(numClasses, 1);
sampleVisualizations = cell(numClasses, 1);


%% Extract HOG from one image per class

for c = 1:numClasses
    className = classes{c};

    classIndices = find(imds.Labels == className);
    imageIndex = classIndices(1);

    img = readimage(imds, imageIndex);

    [features, visualization] = ...
        extractFixedHOGFeatures( ...
        img, targetSize, cellSize);

    sampleFeatures{c} = features;
    sampleImages{c} = img;
    sampleVisualizations{c} = visualization;

    fprintf("\nClass: %s\n", className);
    fprintf("File: %s\n", imds.Files{imageIndex});
    fprintf("Original dimensions: %d x %d x %d\n", size(img, 1), size(img, 2), size(img, 3));
    fprintf("HOG input dimensions: %d x %d\n", targetSize(1), targetSize(2));
    fprintf("HOG feature length: %d\n", numel(features));
    fprintf("NaN values: %d\n", sum(isnan(features)));
    fprintf("Infinite values: %d\n", sum(isinf(features)));

end

%confirm that every class produced the same feature length
featureLengths = cellfun(@numel, sampleFeatures);
assert(isscalar(unique(featureLengths)), "Vectors have inconsistent lengths");
fprintf("\nAll four images produced valid %d-dimensional HOG vectors\n", featureLengths(1));


%show one useful HOG visualization
exampleClass = 1;
preparedImage = prepareImageForHOG(sampleImages{exampleClass}, targetSize);

figure;
tiledlayout(1, 2, ...
    "TileSpacing", "compact", ...
    "Padding", "compact");

nexttile;
imshow(preparedImage);
title(sprintf( ...
    "%s: prepared HOG input", ...
    classes{exampleClass}), ...
    "Interpreter", "none");

nexttile;
imshow(preparedImage);
hold on;
plot(sampleVisualizations{exampleClass});
hold off;

title("HOG gradient visualization");


%% Extract from all 2000 images

%load previously saved results from LST feature extraction
load( ...
    "initial_LST_features.mat", ...
    "Y", ...
    "files", ...
    "dataSplit");

%confirm that this datastore uses the same image order
currentFiles = string(imds.Files);
savedFiles = string(files);

assert( ...
    numel(currentFiles) == numel(savedFiles), ...
    "The current datastore has a different number of images.");
assert( ...
    all(currentFiles == savedFiles), ...
    "The current datastore file order differs from the LST file order.");
assert( ...
    all(imds.Labels == Y), ...
    "The current labels differ from the saved labels.");

%determine HOG vector length using the first image
firstImage = readimage(imds, 1);
firstFeatures = extractFixedHOGFeatures(firstImage, targetSize, cellSize);
numHOGFeatures = numel(firstFeatures);

fprintf("HOG features per image: %d\n", numHOGFeatures);

%allocate feature matrix
X_HOG = zeros(numImages, numHOGFeatures);
failed = false(numImages, 1);

%extract HOG features
for i = 1:numImages
    try
        img = readimage(imds, i);

        X_HOG(i, :) = extractFixedHOGFeatures(img, targetSize, cellSize);

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

%validate results
fprintf("\nSuccessful images: %d\n", sum(~failed));
fprintf("Failed images: %d\n", sum(failed));
fprintf("HOG matrix dimensions: %d x %d\n", size(X_HOG, 1), size(X_HOG, 2));

assert(~any(failed), "One or more HOG extractions failed");
assert(all(isfinite(X_HOG), "all"), "X_HOG contains NaN or Inf values");


%% Save results separately from LST

save( ...
    "initial_HOG_features.mat", ...
    "X_HOG", ...
    "Y", ...
    "files", ...
    "dataSplit", ...
    "targetSize", ...
    "cellSize", ...
    "-v7.3");


%% --------------------------------------------------------------------
% Resize and pad an image for HOG extraction
% Inputs:
%   img
%   targetSize - final [height width], like [256 256]
%
% Output:
%   grayImage  - fixed-size grayscale image
%
% Preserves the original aspect ratio. It resizes the image
% to fit inside the requested dimensions, then pads the remaining area.
function grayImage = prepareImageForHOG(img, targetSize)
arguments
    img
    targetSize (1,2) double {mustBeInteger, mustBePositive} = [256 256]
end

if ndims(img) ~= 3 || size(img, 3) ~= 3
    error("Need to be an RGB image.");
end

originalHeight = size(img, 1);
originalWidth = size(img, 2);

%choose the largest scale that fits the entire image inside the
%requested dimensions without cropping
scale = min( ...
    targetSize(1) / originalHeight, ...
    targetSize(2) / originalWidth);

resizedHeight = max(1, round(originalHeight * scale));
resizedWidth = max(1, round(originalWidth * scale));

resizedImage = imresize(img, [resizedHeight resizedWidth]);

%use gray padding rather than black padding
if isa(img, "uint8")
    paddingValue = uint8(128);
elseif isa(img, "uint16")
    paddingValue = uint16(32768);
else
    paddingValue = cast(0.5, "like", img);
end

paddedImage = repmat( ...
    paddingValue, ...
    targetSize(1), ...
    targetSize(2), ...
    3);

%center the resized image inside the padded image
rowStart = floor((targetSize(1) - resizedHeight) / 2) + 1;
colStart = floor((targetSize(2) - resizedWidth) / 2) + 1;

rowEnd = rowStart + resizedHeight - 1;
colEnd = colStart + resizedWidth - 1;

paddedImage(rowStart:rowEnd, colStart:colEnd, :) = resizedImage;
grayImage = rgb2gray(paddedImage);

end


%% --------------------------------------------------------------------
% Extract fixed-length HOG features.
% Inputs:
%   img
%   targetSize - fixed HOG image dimensions
%   cellSize   - HOG cell dimensions
% Outputs:
%   features
%   visualization
%
% The visualization is useful during testing but is not needed when
% processing the complete dataset.
function [features, visualization] = extractFixedHOGFeatures(img, targetSize, cellSize)
arguments
    img
    targetSize (1,2) double {mustBeInteger, mustBePositive} = [256 256]
    cellSize (1,2) double {mustBeInteger, mustBePositive} = [32 32]
end

grayImage = prepareImageForHOG(img, targetSize);
[features, visualization] = extractHOGFeatures(grayImage, "CellSize", cellSize);

end
