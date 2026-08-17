% KNN! 
clear; 
clc; 
iter = 200; 
fresh = false; 

% Load in the previously extracted features 
load("initial_HOG_features.mat", "X_HOG");
load("initial_LST_features.mat", "X_LST");
load("initial_shape_features.mat", "X_shape");
load("CNN_features_efficientnetb0_globalAvgPool.mat", "X_CNN");

% Load the labels and split 
load("initial_LST_features.mat", "Y", "dataSplit");
trainMask = (dataSplit == "train");
validMask = (dataSplit == "validation");
y_train = Y(trainMask);
y_valid = Y(validMask);

% Extract the training samples
X_train_lst = X_LST(trainMask, :);
X_train_hog = X_HOG(trainMask, :);
X_train_shape = X_shape(trainMask, :); 
X_train_cnn = X_CNN(trainMask, :); 

% Extract the validation samples 
X_valid_lst = X_LST(validMask, :);
X_valid_hog = X_HOG(validMask, :);
X_valid_shape = X_shape(validMask, :); 
X_valid_cnn = X_CNN(validMask, :); 

% Combine the combined feature training + validation sets 
X_train_lst_hog = [X_train_lst, X_train_hog]; 
X_train_lst_shape = [X_train_lst, X_train_shape]; 

X_valid_lst_hog = [X_valid_lst, X_valid_hog]; 
X_valid_lst_shape = [X_valid_lst, X_valid_shape]; 


% LST 
createKNN_wrapper("knn_lst.mat", X_train_lst, y_train, X_valid_lst, y_valid, "LST", fresh, iter); 

% LST-HOG
createKNN_wrapper("knn_lst_hog.mat", X_train_lst_hog, y_train, X_valid_lst_hog, y_valid, "LST_HOG", fresh, iter); 

% LST-Shape 
createKNN_wrapper("knn_lst_shape.mat", X_train_lst_shape, y_train, X_valid_lst_shape, y_valid, "LST_shape", fresh, iter); 

% CNN
createKNN_wrapper("knn_cnn.mat", X_train_cnn, y_train, X_valid_cnn, y_valid, "CNN", fresh, iter); 


function createKNN_wrapper(filename, X_train, y_train, X_valid, y_valid, type, fresh, iterations)

    if (isfile(filename) & ~fresh)
        knn = load(filename);
        knn = knn.knn;
    else
        fprintf("Creating %s KNN...\n", type)
        knn = createKNN(X_train, y_train, iterations, type);
        save(filename, "knn");
    end

    [accuracy, classAUCs] = analyzeKNN(knn, X_valid, y_valid);
    fprintf("\n%s KNN Results...\nOverall Accuracy: %.4f ", type, accuracy);
    fprintf("\n%s Class AUCs: ", type);
    disp(classAUCs);

end


function mdl = createKNN(x, y, iter, type)
    mkdir("SavedOutputs/KNN/" + type);
    diary("SavedOutputs/KNN/" + type + "/" + iter + "Log");
    hpoOptions = hyperparameterOptimizationOptions(AcquisitionFunctionName="expected-improvement-plus", MaxObjectiveEvaluations=iter);
    mdl = fitcknn(x, y, "Standardize", true, OptimizeHyperparameters={'NumNeighbors', 'Distance', 'DistanceWeight'},HyperparameterOptimizationOptions=hpoOptions);
    diary off
end

function [accuracy, classAUCs] = analyzeKNN(mdl, xValid, yValid)

    [label, score] = predict(mdl, xValid);
    
    accuracy = sum((label == yValid), "all") / numel(yValid);
    
    rocObj = rocmetrics(yValid, score, mdl.ClassNames);
    
    classAUCs = auc(rocObj);

end