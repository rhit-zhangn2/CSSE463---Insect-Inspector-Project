clear;
clc;

%Folder containing the four class subfolders
dataDir = fullfile("initial_subset_2000/initial_subset_2000/", "images/");

assert(isfolder(dataDir), "Image folder not found: %s", dataDir);

imds = imageDatastore(dataDir, "IncludeSubfolders",true, "LabelSource","foldernames");

fprintf("Loaded %d image paths\n", numel(imds.Files));
classCounts = countEachLabel(imds);
disp(classCounts);

%--------------------------------------------------------------------
classes = categories(imds.Labels);

figure;
tiledlayout(2, 2);

for i = 1:numel(classes)
    className = classes{i};

    classIndices = find(imds.Labels == className);
    imgIndex = classIndices(1);

    img = readimage(imds, imgIndex);

    nexttile;
    imshow(img);
    title(sprintf("%s\n%d x %d x %d", ...
        className, ...
        size(img, 1), ...
        size(img, 2), ...
        size(img, 3)), ...
        "Interpreter", "none");
end


rng(463);

figure;
tiledlayout(4, 5, ...
    "TileSpacing", "compact", ...
    "Padding", "compact");

for c = 1:numel(classes)
    className = classes{c};
    classIndices = find(imds.Labels == className);

    selected = classIndices(randperm(numel(classIndices), 5));

    for j = 1:5
        img = readimage(imds, selected(j));

        nexttile;
        imshow(img);

        if j == 1
            title(className, "Interpreter", "none");
        end
    end
end

%---------------------------------------------------------------
nImages = numel(imds.Files);

heights = zeros(nImages, 1);
widths = zeros(nImages, 1);
channels = zeros(nImages, 1);
readFailed = false(nImages, 1);

for i = 1:nImages
    try
        info = imfinfo(imds.Files{i});

        heights(i) = info.Height;
        widths(i) = info.Width;
        img = imread(imds.Files{i});
        if ndims(img) == 2
            channels(i) = 1;
        else
            channels(i) = size(img, 3);
        end

    catch ME
        readFailed(i) = true;
        fprintf("Could not inspect %s\n", imds.Files{i});
        fprintf("Reason: %s\n", ME.message);
    end
end

fprintf("Unreadable images: %d\n", sum(readFailed));

valid = ~readFailed;

fprintf("Image height range: %d to %d\n", min(heights(valid)), max(heights(valid)));
fprintf("Image width range: %d to %d\n", min(widths(valid)), max(widths(valid)));
fprintf("Grayscale images: %d\n", sum(channels(valid) == 1));
fprintf("RGB images: %d\n", sum(channels(valid) == 3));
fprintf("Other channel counts: %d\n", sum(~ismember(channels(valid), [1 3])));


%--------------------------------------------------------------------
