# Insect Inspector Preliminary Feature Extraction

## Dataset

This package uses a balanced preliminary subset of 2,000 randomly
chosen iNaturalist images from the shared workspace:

- Cicadomorpha: 500
- Fulgoromorpha: 500
- Heteroptera: 500
- Sternorrhyncha: 500

All images were readable RGB JPEG images. Their dimensions varied, with
the longest image dimensions commonly around 500 pixels.

## Shared split

Random seed: 463

- Training: 70%, 350 images per class
- Validation: 15%, 75 images per class
- Testing: 15%, 75 images per class

The split is stored in `dataSplit` 

## LST features

Each image was divided into a 4-by-4 normalized grid without resizing.

For every block, the following were calculated:

- mean and standard deviation of L
- mean and standard deviation of S
- mean and standard deviation of T

L = R + G + B  
S = R - B  
T = R - 2G + B

This produces 96 features per image.

Initial PCA and feature-distribution plots showed weak class separation,
suggesting that broad spatial color information alone is unlikely to be
sufficient.

## HOG features

Note: Run LST extraction before running HOG

Images were resized to fit within 256-by-256 pixels while preserving
aspect ratio, then centered using gray padding and converted to
grayscale.

HOG was extracted using a 32-by-32 cell size.

The exact HOG dimensionality is stored in `X_HOG`.

## Files

- `initial_subset_2000/initial_subset_2000/`: downloaded images
- `initial_LST_features.mat`: LST features and metadata
- `initial_HOG_features.mat`: HOG features and metadata
- `SavedOutputs/`: plots and ANOVA results

Main Scripts:
- `inspectDataset.m`: first inspection of the dataset
- `testLSTFeatureExtraction.m`: extract using LST, saved results to `initial_LST_features.mat`
- `testHOGFeatureExtraction.m`: extract using HOG, saved results to `initial_HOG_features.mat`
- `analyzeLSTFeatures.m`: used to analyze and plot LST results
- `analyzeHOGFeatures.m`: used to analyze and plot HOG results

