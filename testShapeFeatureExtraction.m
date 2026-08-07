clear;
clc; 

% Folder containing the four class subfolders
dataDir = fullfile("initial_subset_2000/initial_subset_2000/", "images/");

assert(isfolder(dataDir), "Image folder not found: %s", dataDir);

imds = imageDatastore(dataDir, "IncludeSubfolders",true, "LabelSource","foldernames");

fprintf("Loaded %d image paths\n", numel(imds.Files));
classCounts = countEachLabel(imds);
disp(classCounts);

classes = categories(imds.Labels);

nFeatures = 10; 
nImages = numel(imds.Files);

X_shape = zeros(nImages, nFeatures); 

% for i = 1:nImages
%     disp(i)
%     [img, fileinfo] = readimage(imds, i); 
%     feature_vector = process_bug(img); 
%     X_shape(i, :) = feature_vector; 
% end

% y = imds.Labels; 

% save("initial_shape_features.mat", "X_shape", "y", "-v7.3"); 

img = imread("initial_subset_2000\initial_subset_2000\images\fulgoromorpha\203142606.jpg");
process_bug(img)

function features = process_bug(img)
    figure(2); 
    
    % Convert the image to grayscale 
    img = rgb2gray(img); 
    subplot(2, 2, 1);
    imshow(img); 
    title("Grayscale Image");

    % Find the edges 
    sobel_edges = edge(img, "sobel"); 

    subplot(2, 2, 2); 
    imshow(sobel_edges); 
    title("Sobel Edge Finder"); 
    
    % Convert to uint8 
    mask = zeros(size(sobel_edges)); 
    mask(sobel_edges) = 255; 

    % Gaussian blur the image, and only select pixels with high enough
    % intensity after the blur
    % This means they have lots of neighbors and are ideally part of the
    % insect 
    gauss = imgaussfilt(mask, 4); 
    mask = zeros(size(mask)); 
    mask(gauss >= 32) = 255; 

    subplot(2, 2, 3); 
    imshow(mask);
    title("Gaussian Blur Thresholding")

    % Connected components 
    cc = bwlabel(mask); 
    nRegions = max(max(cc)); 
    
    % Calculate average area 
    areas = zeros(nRegions); 
    for i = 1:nRegions
        areas(i) = sum(sum(mask(cc == i)));
    end
    avg_area = sum(sum(areas)) / max(max(cc)); 

    % Use average area for size thresholding 
    for i = 1:nRegions
        if(sum(sum(mask(cc == i))) < 0.25 * avg_area)
            mask(cc == i) = 0; 
        end
    end

    % subplot(2, 2, 3); 
    % imshow(mask); 

    % Fill in holes      
    mask = imfill(mask); 

    subplot(2, 2, 4); 
    imshow(mask); 
    title("Size Thresholding & Fill Holes"); 


    % Calculate feature vector
    nFeatures = 10; 

    % we're going to take the average of all the remaining regions
    cc = bwlabel(mask); 
    nRegions = max(max(cc)); 
    features = zeros(nFeatures, 1); 

    if(nRegions ~= 0)  % yes I did have this happen
        features = zeros(nRegions, nFeatures); 
        
        features(:, 1) = table2array(regionprops("table",cc,"Area")); 
        features(:, 2) = table2array(regionprops("table",cc,"Perimeter")); 
        features(:, 3) = table2array(regionprops("table",cc,"MajorAxisLength")); 
        features(:, 4) = table2array(regionprops("table",cc,"MinorAxisLength")); 
        features(:, 5) = table2array(regionprops("table",cc,"Eccentricity")); 
        features(:, 6) = table2array(regionprops("table",cc,"Circularity")); 
        features(:, 7) = table2array(regionprops("table",cc,"Solidity")); 
        features(:, 8) = table2array(regionprops("table",cc,"Extent")); 

        min_feret = regionprops("table", cc, "MinFeretProperties"); 
        features(:, 9) = table2array(min_feret(:, 1)); 
    
        max_feret = regionprops("table", cc, "MaxFeretProperties"); 
        features(:, 10) = table2array(max_feret(:, 1)); 

        if(nRegions ~= 1)
            features = mean(features); 
        end
    end

end
