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

X_CNN = zeros(numImages, featureLength, "single");

for i = 1:numImages
    
    img = readimage(imds,i);
    imgPrepared = prepareCNNImage(img, inputSize);
    activation = activations(net, imgPrepared, featureLayer);

    %IMPORTANT:
    %convert arbitrary CNN output into one feature row
    X_CNN(i,:) = reshape(activation, 1,...
        []);

    if mod(i,100)==0
        fprintf("Processed %d/%d images\n", i, numImages);
    end
end

fprintf("\nFinished CNN extraction\n");
disp(size(X_CNN));


%% Labels

Y = imds.Labels;
files = string(imds.Files);


%% Reuse existing split

load("initial_LST_features.mat", "dataSplit");


%% Save

save( ...
    "CNN_features_efficientnetb0_globalAvgPool.mat",...
    "X_CNN",...
    "Y",...
    "files",...
    "dataSplit",...
    "networkName",...
    "featureLayer",...
    "-v7.3");

fprintf("Saved: initial_CNN_features.mat\n");


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