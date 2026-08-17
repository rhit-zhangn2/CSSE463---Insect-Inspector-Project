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

## Feature Extraction

### LST features

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

### HOG features

Note: Run LST extraction before running HOG

Images were resized to fit within 256-by-256 pixels while preserving
aspect ratio, then centered using gray padding and converted to
grayscale.

HOG was extracted using a 32-by-32 cell size.

The exact HOG dimensionality is stored in `X_HOG`.

### Shape features 

Images were converted to grayscale, before being run through a sobel image detector, then
postprocessed to remove noise and convert the insect to one homogenous region. 

Area, Perimeter, Major Axis Length, Minor Axis Length, Eccentricity, Circularity, Solidity, 
Extent, Mininum Feret Diameter, and Maximum Feret Diameter were calculated for each 
region and averaged across all regions 

### CNN Features

Images were resized to 224 x 224, then passed into ElasticNetB0, the outputs being
taken from the Global Average Pool Layer 


## Files

- `initial_subset_2000/initial_subset_2000/`: downloaded images, in subfolders according to clade
- `initial_LST_features.mat`: LST features and metadata
- `initial_HOG_features.mat`: HOG features and metadata
- `SavedOutputs/`: plots and ANOVA results


Main Scripts:
- `inspectDataset.m`: first inspection of the dataset
- `analyzeLSTFeatures.m`: used to analyze and plot LST results
- `analyzeHOGFeatures.m`: used to analyze and plot HOG results
- `analyzeSVM.m`: used to calculate the accuracy and AUC-ROC of SVM models
- `createSVM.m`: initializes an SVM 
- `testLSTFeatureExtraction.m`: extract using LST, saved results to `initial_LST_features.mat`
- `testHOGFeatureExtraction.m`: extract using HOG, saved results to `initial_HOG_features.mat`
- `testShapeFeatureExtraction.m`: extract using Shape Features, saved results to `initial_Shape_features.mat`
- `testLSTBlocks.m`: tried many different block sizes and evaluated their performance
- `KNN.m`: used to train and evaluate k-Nearest Neighbor models
- `logisticRegression.m`: used to train and evaluate Logistic Regression models
- `testNNPerformance`: used to train and evaluate Neural Network models
- `testRandomForestPerformance`: used to train and evaluate Random Forest Models 
- `testSVMPerformance.m`: used to train and evaluate Support Vector Machine Models 
- `testSVMPerformance2.m`: used to generate the graphs in the week 8 report  


Saved Features:
- `CNN_features_alexnet_fc7.mat`
- `CNN_features_alexnet_pool5.mat`
- `CNN_features_efficientnetb0_globalAvgPool.mat'
- `CNN_features_resnet18_pool5.mat`
- `CNN_features_resnet50_avgPool.mat`
- `initial_HOG_features.mat`
- `initial_LST_features.mat`
- `initial_shape_features.mat`

Saved Trained Models: 
- `CNN_RF.mat`
- `CNN_SVM_alexnet_fc7.mat`
- `CNN_SVM_alexnet_pool5.mat`
- `CNN_SVM_efficientnetb0_globalAvgPool.mat`
- `CNN_SVM_resnet18_pool5.mat`
- `CNN_SVM_resnet50_avgPool.mat`
- `HOG_Shape_SVM.mat`
- `HOG_SVM.mat`
- `knn_cnn.mat`
- `knn_lst_hog.mat`
- `knn_lst_shape.mat`
- `knn_lst.mat`
- `logreg_cnn_efficientnet.mat`
- `logreg_lst_hog.mat`
- `logreg_lst_shape.mat`
- `logreg_lst.mat`
- `LST_HOG_RF.mat`
- `LST_HOG_Shape_SVM.mat`
- `LST_HOG_SVM.mat`
- `LST_RF.mat`
- `LST_SHAPE_RF.mat`
- `LST_SVM.mat`
- `nn_cnn_efficientnetb0.mat`
- `nn_cnn_resnet50.mat`
- `nn_cnn.mat`
- `nn_lst_hog.mat`
- `nn_lst_shape.mat`
- `nn_lst.mat`
- `Shape_SVM.mat`









