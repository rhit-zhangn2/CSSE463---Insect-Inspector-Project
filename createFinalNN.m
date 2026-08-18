% Create our final model on the GPU
clear;
clc;
%% -----------------------------------------
% Load

iter = 215;
fresh = false;

load("final_CNN_features.mat", "X_CNN", "Y");
load("final_dataSplit.mat", "dataSplit");

trainMask = (dataSplit == "train");
validMask = (dataSplit == "validation");

X_Train = gpuArray(single(X_CNN(trainMask, :)));
X_Valid = gpuArray(single(X_CNN(validMask, :)));

[Y_numeric, gnames] = grp2idx(Y);

Y_Train = gpuArray(single(Y_numeric(trainMask, :)));
Y_Valid = Y(validMask, :);

%% --------------------------------------------------------------------
% Run

mdl = createNN_wrapper("NN_EfficientnetB0FINAL02.mat", X_Train, Y_Train, X_Valid, Y_Valid, "CNN_Efficientnetb0", fresh, iter, gnames);

%% ------------------------------------------------------------------

% Wrapper function
function nn = createNN_wrapper(filename, X_train, y_train, X_valid, y_valid, type, fresh, iterations, gnames)
    if (isfile(filename) && ~fresh)
        nn = load(filename);
        nn = nn.nn;
    else
        fprintf("Creating %s Neural Network...\n", type)
        nn = createNN(X_train, y_train, iterations, type);
        save(filename, "nn");
    end

    [accuracy, classAUCs] = analyzeNN(nn, X_valid, y_valid, gnames);
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

    % seems relu is the most preferred, but we can keep the rest
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
function [accuracy, classAUCs] = analyzeNN(mdl, xValid, yValid, gnames)
    [label, score] = predict(mdl, xValid);

    catLabels = gnames(label);
    classNames = gnames(mdl.ClassNames);
    arr_score = gather(score);

    rocObj = rocmetrics(yValid, arr_score, classNames);
    accuracy = sum((catLabels == yValid), "all") / numel(yValid);
    classAUCs = auc(rocObj);
end
