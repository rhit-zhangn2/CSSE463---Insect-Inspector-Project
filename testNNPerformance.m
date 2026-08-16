% Neural Network
clear;
clc;
%% -----------------------------------------
% Load

iter = 20;
fresh = false;

load("initial_HOG_features.mat", "X_HOG");
load("initial_LST_features.mat", "X_LST");
load("initial_shape_features.mat", "X_shape");
load("CNN_features_efficientnetb0_globalAvgPool.mat", "X_CNN");

load("initial_LST_features.mat", "Y", "dataSplit");
trainMask = (dataSplit == "train");
validMask = (dataSplit == "validation");
y_train = Y(trainMask);
y_valid = Y(validMask);

X_train_lst = X_LST(trainMask, :);
X_train_hog = X_HOG(trainMask, :);
X_train_shape = X_shape(trainMask, :);
X_train_cnn = X_CNN(trainMask, :);

X_valid_lst = X_LST(validMask, :);
X_valid_hog = X_HOG(validMask, :);
X_valid_shape = X_shape(validMask, :);
X_valid_cnn = X_CNN(validMask, :);

X_train_lst_hog = [X_train_lst, X_train_hog];
X_train_lst_shape = [X_train_lst, X_train_shape];

X_valid_lst_hog = [X_valid_lst, X_valid_hog];
X_valid_lst_shape = [X_valid_lst, X_valid_shape];

%% --------------------------------------------------------------------
% Run

% LST
createNN_wrapper("nn_lst.mat", X_train_lst, y_train, X_valid_lst, y_valid, "LST", fresh, iter);

% LST-HOG
createNN_wrapper("nn_lst_hog.mat", X_train_lst_hog, y_train, X_valid_lst_hog, y_valid, "LST_HOG", fresh, iter);

% LST-Shape
createNN_wrapper("nn_lst_shape.mat", X_train_lst_shape, y_train, X_valid_lst_shape, y_valid, "LST_shape", fresh, iter);

% CNN
createNN_wrapper("nn_cnn_efficientnetb0.mat", X_train_cnn, y_train, X_valid_cnn, y_valid, "CNN_Efficientnetb0", fresh, iter);

%% ------------------------------------------------------------------

% Wrapper function
function createNN_wrapper(filename, X_train, y_train, X_valid, y_valid, type, fresh, iterations)
    if (isfile(filename) && ~fresh)
        nn = load(filename);
        nn = nn.nn;
    else
        fprintf("Creating %s Neural Network...\n", type)
        nn = createNN(X_train, y_train, iterations, type);
        save(filename, "nn");
    end

    [accuracy, classAUCs] = analyzeNN(nn, X_valid, y_valid);
    fprintf("\n%s Neural Network Results...\nOverall Accuracy: %.4f ", type, accuracy);
    fprintf("\n%s Class AUCs: ", type);
    disp(classAUCs);
end

%% ----------------------------------------------------------------------------------

% Create 
function [mdl, optimizationResults, bestObserved] = createNN(x, y, iter, type)
    outputFolder = "SavedOutputs/NN/" + type;
    if ~isfolder(outputFolder)
        mkdir(outputFolder);
    end

    diary(outputFolder + "/" + iter + "Log");

    rng("default");

    hpoOptions = hyperparameterOptimizationOptions( ...
        AcquisitionFunctionName="expected-improvement-plus", ...
        MaxObjectiveEvaluations=iter);

    params = hyperparameters("fitcnet", x, y);
    paramNames = string({params.Name});

    %only optimize selected params
    for i = 1:numel(params)
        params(i).Optimize = false;
    end

    % seems like nn can achieve great performance with only 1 layer,
    % but here we allow exploring 2 layers. More layers don't seem to
    % have much effect
    numLayersParam = find(paramNames == "NumLayers");
    params(numLayersParam).Range = [1 2];
    params(numLayersParam).Optimize = true;
    
    layer1Param = find(paramNames == "Layer_1_Size");
    params(layer1Param).Range = [4 64];
    params(layer1Param).Optimize = true;
    
    layer2Param = find(paramNames == "Layer_2_Size");
    params(layer2Param).Range = [4 32];
    params(layer2Param).Optimize = true;

    % seems relu is the most prefered, but we can keep the rest
    activationParam = find(paramNames == "Activations");
    params(activationParam).Range = {'relu', 'tanh', 'sigmoid', 'none'};
    params(activationParam).Optimize = true;

    lambdaParam = find(paramNames == "Lambda");
    params(lambdaParam).Range = [1e-7 1e-1];
    params(lambdaParam).Optimize = true;

    optimizedModel = fitcnet(x, y, ...
        "Standardize", true, ...
        OptimizeHyperparameters=params, ...
        HyperparameterOptimizationOptions=hpoOptions);

    optimizationResults = optimizedModel.HyperparameterOptimizationResults;
    bestObserved = bestPoint(optimizationResults);

    %get the best params found and rerun
    bestNumLayers = bestObserved.NumLayers;
    if bestNumLayers == 1
        bestLayerSizes = bestObserved.Layer_1_Size;
    else
        bestLayerSizes = [
            bestObserved.Layer_1_Size, bestObserved.Layer_2_Size];
    end

    bestLayerSize1 = bestObserved.Layer_1_Size;
    bestLayerSize2 = bestObserved.Layer_2_Size;
    bestActivation = string(bestObserved.Activations);
    bestLambda = bestObserved.Lambda;

    fprintf("\nRefitting best observed %s NN...\n", type);
    fprintf("Layer1 size: %d | Layer2 size: %d | Activation: %s | Lambda: %.4g\n", ...
        bestLayerSize1, bestLayerSize2, bestActivation, bestLambda);

    rng("default");
    mdl = fitcnet(x, y, ...
        "Standardize", true, ...
        "LayerSizes", bestLayerSizes, ...
        "Activations", bestActivation, ...
        "Lambda", bestLambda, ...
        "IterationLimit", 3000);

    diary off
end

%% ---------------------------------------------------------------------

% Analyze
function [accuracy, classAUCs] = analyzeNN(mdl, xValid, yValid)
    [label, score] = predict(mdl, xValid);
    accuracy = sum((label == yValid), "all") / numel(yValid);
    rocObj = rocmetrics(yValid, score, mdl.ClassNames);
    classAUCs = auc(rocObj);
end
