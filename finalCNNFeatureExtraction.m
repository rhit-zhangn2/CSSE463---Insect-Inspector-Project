clear;
clc;

%% Load Data

%Folder containing the four class subfolders
dataDir = fullfile("full_set/");

assert(isfolder(dataDir), "Image folder not found: %s", dataDir);

imds = imageDatastore(dataDir, "IncludeSubfolders",true, "LabelSource","foldernames");

fprintf("Loaded %d image paths\n", numel(imds.Files));
classCounts = countEachLabel(imds);
disp(classCounts);
%--------------------------------------------------------------------


%% Load pretrained CNN

% Candidate layers:
%
% AlexNet:
%   pool5
%   fc7
%
%   pool5:
%       lower-level transferable features
%   fc7:
%       higher-level semantic features
%
% ResNet18:
%   pool5
%
% ResNet50:
%   avg_pool
%
% EfficientNet-B0:
%   efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool
%

networkName = "efficientnetb0"; %for printing convenience
net = efficientnetb0;
featureLayer = "efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool";

inputSize = net.Layers(1).InputSize;

fprintf("\nNetwork: %s\n",networkName);
fprintf("Input size: %d x %d x %d\n",...
    inputSize(1),...
    inputSize(2),...
    inputSize(3));
fprintf("Feature layer: %s\n",featureLayer);


%% Test extraction on one image

img = readimage(imds,1);

imgPrepared = prepareCNNImage(img, inputSize);

activation = activations(net, imgPrepared, featureLayer);

fprintf("\nRaw activation size:\n");
disp(size(activation));

featureVector = reshape(activation,1,[]);

fprintf("Feature vector length:\n");
disp(length(featureVector));


%% Extract all features

numImages = numel(imds.Files);

%determine feature size
featureLength = length(featureVector);

%load("X_CNN_FINAL01.mat", "X_CNN");
X_CNN = zeros(numImages, featureLength);
startIdx = 1;
%edit the batch size for whatever gpu is running this
batchSize = 512; 
i = startIdx;
while i <= numImages % i = startIdx:batchSize:numImages
    
    lastIdx = min(i + batchSize - 1, numImages);
    batchIdx = i:lastIdx;

    XBatch = zeros([inputSize numel(batchIdx)], "single");

    validIdx = zeros(1, numel(batchIdx), "like", batchIdx);
    batchPos = 0;
    
    for j = 1:numel(batchIdx)
        % mostly for CMYK jpegs that MATLAB cant read
        try
            img = readimage(imds, batchIdx(j));
            imgPrepared = prepareCNNImage(img, inputSize);
    
            batchPos = batchPos + 1;
            XBatch(:,:,:,batchPos) = imgPrepared;
            validIdx(batchPos) = batchIdx(j);
    
        catch err
            fprintf("Skipping image %d: %s\n", batchIdx(j), err.message);
        end
    end
    
    if batchPos > 0
        XBatch = XBatch(:,:,:,1:batchPos);
        validIdx = validIdx(1:batchPos);
    
        activation = activations( ...
            net, XBatch, featureLayer, ...
            "ExecutionEnvironment", "gpu", ...
            "OutputAs", "rows", ...
            "MiniBatchSize", batchSize);
 
        %IMPORTANT:
        %convert arbitrary CNN output into one feature row
        %X_CNN(i,:) = reshape(activation, 1, []);

        % pull data to main memory
        X_CNN(validIdx,:) = gather(activation);
    end

    if mod(lastIdx, 1000) < batchSize
        fprintf("Processed %d/%d images\n", lastIdx, numImages);
    end

    i = i + batchSize;
end

fprintf("\nFinished CNN extraction\n");
disp(size(X_CNN));

%% Labels

Y = imds.Labels;
files = string(imds.Files);


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


%% Save
save("final_dataSplit.mat", "dataSplit");
fprintf("Saved: final_dataSplit.mat");

save( ...
    "final_CNN_features.mat",...
    "X_CNN",...
    "Y",...
    "files",...
    "networkName",...
    "featureLayer",...
    "-v7.3");

fprintf("Saved: final_CNN_features.mat\n");


%% Prepare Image Function
%Similar to how images are processed for HOG
function output = prepareCNNImage(img,inputSize)

%convert grayscale images if necessary
if size(img,3)==1
    img = cat(3,img,img,img);
end

% Current strategy:
%   Resize image while preserving aspect ratio
%   Then pad to CNN input size
% This avoids stretching insects.

targetH = inputSize(1);
targetW = inputSize(2);

scale = min(targetH/size(img,1), targetW/size(img,2));
newSize = round([size(img,1),size(img,2)]*scale);
img = imresize(img,newSize);

%create padded image
output = zeros(targetH, targetW, 3, "like", img);

rowStart = floor((targetH-newSize(1))/2)+1;
colStart = floor((targetW-newSize(2))/2)+1;

output(rowStart:rowStart+newSize(1)-1, colStart:colStart+newSize(2)-1, :) = img;

end