clear; 
clc; 

%Number of iterations to run the hyperparameter tuning
iterations = 20;
%Whether or not to recreate SVMs from scratch
fresh = false;

%Create masks
load("initial_LST_features.mat", "Y", "dataSplit");
trainMask = (dataSplit == "train");
validMask = (dataSplit == "validation");

Y_Train = Y(trainMask);
Y_Valid = Y(validMask);

function createSVM_wrapper(filename, X_train, y_train, X_valid, y_valid, type, fresh, iterations)

    if (isfile(filename) & ~fresh)
        load(filename);
    else
        fprintf("Creating %s SVM...", type)
        svm = createSVM(iterations, X_train, y_train, type);
        save(filename, "svm");
    end

    [accuracy, classAUCs] = analyzeSVM(svm, X_valid, y_valid);
    fprintf("\n %s SVM Results...\nOverall Accuracy: %.4f ", type, accuracy);% + "\nLST Class Accuracies: ");
    %celldisp(classAccuracies);
    fprintf("\n %s Class AUCs: ", type);
    disp(classAUCs);

end


load("initial_HOG_features.mat", "X_HOG");
load("initial_LST_features.mat", "X_LST");
load("initial_shape_features.mat", "X_shape");
load("CNN_features_efficientnetb0_globalAvgPool.mat", "X_CNN");

% Define Individual Train Sets
X_HOG_Train = X_HOG(trainMask, :); 
X_LST_Train = X_LST(trainMask, :); 
X_shape_Train = X_shape(trainMask, :); 
X_CNN_Train = X_CNN(trainMask, :);

% Define Pairs of 2 Train Sets
X_COMB_LST_HOG_Train = [X_HOG, X_LST];
X_COMB_LST_HOG_Train = X_COMB_LST_HOG_Train(trainMask, :);
X_COMB_LST_SHAPE_Train = [X_shape, X_LST];
X_COMB_LST_SHAPE_Train = X_COMB_LST_SHAPE_Train(trainMask, :);
X_COMB_HOG_SHAPE_Train = [X_shape, X_HOG];
X_COMB_HOG_SHAPE_Train = X_COMB_HOG_SHAPE_Train(trainMask, :);

% Define Trios of 3 Train Set(s) 
X_COMB_LST_HOG_SHAPE_Train = [X_LST, X_shape, X_HOG];
X_COMB_LST_HOG_SHAPE_Train = X_COMB_LST_HOG_SHAPE_Train(trainMask, :);

% Define Individual Validation Sets
X_HOG_Valid = X_HOG(validMask, :);
X_LST_Valid = X_LST(validMask, :);
X_shape_Valid = X_shape(validMask, :); 
X_CNN_Valid = X_CNN(validMask, :);

% Define Pairs of 2 Train Sets
X_COMB_LST_HOG_Valid = [X_HOG, X_LST];
X_COMB_LST_HOG_Valid = X_COMB_LST_HOG_Valid(validMask, :);
X_COMB_LST_SHAPE_Valid = [X_shape, X_LST];
X_COMB_LST_SHAPE_Valid = X_COMB_LST_SHAPE_Valid(validMask, :);
X_COMB_HOG_SHAPE_Valid = [X_shape, X_HOG];
X_COMB_HOG_SHAPE_Valid = X_COMB_HOG_SHAPE_Valid(validMask, :);


% Define Trios of 3 Train Set(s) 
X_COMB_LST_HOG_SHAPE_Valid = [X_LST, X_shape, X_HOG];
X_COMB_LST_HOG_SHAPE_Valid = X_COMB_LST_HOG_SHAPE_Valid(validMask, :);


% Individual SVM
createSVM_wrapper("HOG_SVM.mat", X_HOG_Train, Y_Train, ...
                                    X_HOG_Valid, Y_Valid, "HOG", fresh, iterations); 
createSVM_wrapper("LST_SVM.mat", X_LST_Train, Y_Train, ...
                                    X_LST_Valid, Y_Valid, "LST", fresh, iterations); 
createSVM_wrapper("Shape_SVM.mat", X_shape_Train, Y_Train, ...
                                    X_shape_Valid, Y_Valid, "Shape", fresh, iterations); 
createSVM_wrapper("CNN_SVM_efficientnetb0_globalAvgPool.mat", X_CNN_Train, Y_Train, ...
                                    X_CNN_Valid, Y_Valid, "CNN_efficientnetb0_globalAvgPool", fresh, iterations);

% Pair SVM
createSVM_wrapper("LST_HOG_SVM.mat", X_COMB_LST_HOG_Train, Y_Train, ...
                                    X_COMB_LST_HOG_Valid, Y_Valid, "Combined LST HOG", fresh, iterations); 
createSVM_wrapper("LST_Shape_SVM.mat", X_COMB_LST_SHAPE_Train, Y_Train, ...
                                    X_COMB_LST_SHAPE_Valid, Y_Valid, "Combined LST Shape", fresh, iterations); 
createSVM_wrapper("HOG_Shape_SVM.mat", X_COMB_HOG_SHAPE_Train, Y_Train, ...
                                    X_COMB_HOG_SHAPE_Valid, Y_Valid, "Combined HOG Shape", fresh, iterations); 

% Trio SVM 
createSVM_wrapper("LST_HOG_Shape_SVM.mat", X_COMB_LST_HOG_SHAPE_Train, Y_Train, ...
                                    X_COMB_LST_HOG_SHAPE_Valid, Y_Valid, "Combined LST HOG Shape", fresh, iterations); 

