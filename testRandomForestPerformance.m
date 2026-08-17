fresh = false;

load("initial_HOG_features.mat", "X_HOG", "Y", "dataSplit");
load("initial_LST_features.mat", "X_LST");
load("initial_shape_features.mat", "X_shape");
load("CNN_features_efficientnetb0_globalAvgPool.mat", "X_CNN");

trainMask = (dataSplit == "train");
validMask = (dataSplit == "validation");

Y_Train = Y(trainMask);
Y_Valid = Y(validMask);

X_COMB_LST_HOG_Train = [X_HOG, X_LST];
X_COMB_LST_HOG_Train = X_COMB_LST_HOG_Train(trainMask, :);

X_COMB_LST_SHAPE_Train = [X_shape, X_LST];
X_COMB_LST_SHAPE_Train = X_COMB_LST_SHAPE_Train(trainMask, :);

X_LST_Train = X_LST(trainMask, :);

X_CNN_Train = X_CNN(trainMask, :);

X_COMB_LST_HOG_Valid = [X_HOG, X_LST];
X_COMB_LST_HOG_Valid = X_COMB_LST_HOG_Valid(validMask, :);

X_COMB_LST_SHAPE_Valid = [X_shape, X_LST];
X_COMB_LST_SHAPE_Valid = X_COMB_LST_SHAPE_Valid(validMask, :);

X_LST_Valid = X_LST(validMask, :);

X_CNN_Valid = X_CNN(validMask, :);

function mdl = createRandomForest(filename, X, Y, fresh)
t = templateTree(Reproducible=true);

if (isfile(filename) & ~fresh)
    fprintf("Loading model: %s\n", filename);
    s = load(filename, "mdl");
    mdl = s.mdl;
    return
end
fprintf("Creating model: %s\n", filename);
mdl = fitcensemble(X, Y, ...
    Method="Bag", ...
    Learners=t, ...
    OptimizeHyperparameters={
    'NumLearningCycles', ...
    'MaxNumSplits', ...
    'MinLeafSize' ...
    }, ...
    HyperparameterOptimizationOptions=struct(AcquisitionFunctionName="expected-improvement-plus", UseParallel=true));
save(filename, "mdl");
end

mdl_HOG_LST = createRandomForest("LST_HOG_RF.mat", X_COMB_LST_HOG_Train, Y_Train, fresh);
mdl_SHAPE_LST = createRandomForest("LST_SHAPE_RF.mat", X_COMB_LST_SHAPE_Train, Y_Train, fresh);
mdl_LST = createRandomForest("LST_RF.mat", X_LST_Train, Y_Train, fresh);
mdl_CNN = createRandomForest("CNN_RF.mat", X_CNN_Train, Y_Train, fresh);

function accuracy = analyzeRandomForest(mdl, X_Valid, Y_Valid)
    label = predict(mdl, X_Valid);
    accuracy = sum(label == Y_Valid) / numel(Y_Valid);
end

accuracies = zeros(4, 1);
accuracies(1) = analyzeRandomForest(mdl_CNN, X_CNN_Valid, Y_Valid);
accuracies(2) = analyzeRandomForest(mdl_LST, X_LST_Valid, Y_Valid);
accuracies(3) = analyzeRandomForest(mdl_HOG_LST, X_COMB_LST_HOG_Valid, Y_Valid);
accuracies(4) = analyzeRandomForest(mdl_SHAPE_LST, X_COMB_LST_SHAPE_Valid, Y_Valid);

disp("Model Accuracies:");
disp(accuracies);