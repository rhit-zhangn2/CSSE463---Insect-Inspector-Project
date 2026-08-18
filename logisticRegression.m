% Logistic Regression 
% Based on https://www.geeksforgeeks.org/machine-learning/implementation-of-logistic-regression-from-scratch-using-python/
% because I couldn't find a matlab library for this lol
clear; 
clc; 
iter = 200; 
learningRate = 0.1; 
fresh = true; 

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

function [y_c, y_f, y_h, y_s] = oneVsAllPartition(y)
    y_c = zeros(size(y, 1), 1); 
    y_c(y == "cicadomorpha") = 1;
    y_f = zeros(size(y, 1), 1); 
    y_f(y == "fulgoromorpha") = 1; 
    y_h = zeros(size(y, 1), 1); 
    y_h(y == "heteroptera") = 1; 
    y_s = zeros(size(y, 1), 1); 
    y_s(y == "sternorrhyncha") = 1; 
end


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


logisticRegressor("logreg_lst.mat", X_train_lst, y_train, X_valid_lst, y_valid, iter, learningRate, "LST"); 
logisticRegressor("logreg_lst_hog.mat", X_train_lst_hog, y_train, X_valid_lst_hog, y_valid, iter, learningRate, "LST-HOG"); 
logisticRegressor("logreg_lst_shape.mat", X_train_lst_shape, y_train, X_valid_lst_shape, y_valid, iter, learningRate, "LST-Shape"); 
logisticRegressor("logreg_cnn_efficientnet.mat", X_train_cnn, y_train, X_valid_cnn, y_valid, iter, learningRate, "CNN"); 


function accuracy = logisticRegressor(filename, X_train, y_train, X_valid, y_valid, iter, learningRate, type)

    X_train = normalize(X_train); 
    X_valid = normalize(X_valid); % I KNOW but it's 12:30 and I'm tired
    
    [y_train_c, y_train_f, y_train_h, y_train_s] = oneVsAllPartition(y_train); 

    [weights_c, bias_c] = fit(X_train, y_train_c, iter, learningRate); 
    [weights_f, bias_f] = fit(X_train, y_train_f, iter, learningRate); 
    [weights_h, bias_h] = fit(X_train, y_train_h, iter, learningRate); 
    [weights_s, bias_s] = fit(X_train, y_train_s, iter, learningRate); 

    y_pred_c = predict_proba(X_valid, weights_c, bias_c); 
    y_pred_f = predict_proba(X_valid, weights_f, bias_f); 
    y_pred_h = predict_proba(X_valid, weights_h, bias_h); 
    y_pred_s = predict_proba(X_valid, weights_s, bias_s); 

    [~, y_pred_num] = max([y_pred_c, y_pred_f, y_pred_h, y_pred_s]'); 
   
    y_pred(y_pred_num == 1) = "cicadomorpha"; 
    y_pred(y_pred_num == 2) = "fulgoromorpha"; 
    y_pred(y_pred_num == 3) = "heteroptera";
    y_pred(y_pred_num == 4) = "sternorrhyncha"; 
    y_pred = categorical(y_pred'); 

    accuracy = sum((y_pred == y_valid), "all") / numel(y_valid); 
    fprintf("\n%s Logistic Regression Results...\nOverall Accuracy: %.4f\n", type, accuracy);

    save(filename, "weights_c"); 
    save(filename, "bias_c"); 
    save(filename, "weights_f"); 
    save(filename, "bias_f"); 
    save(filename, "weights_h"); 
    save(filename, "bias_h"); 
    save(filename, "weights_s"); 
    save(filename, "bias_s"); 

end


function y = sigmoid(x)
    y = (1 ./ (1 + exp(-x))); 
end

function [weights, bias] = fit(X, y, iter, learningRate)
    % y = y'; 
    [m, n] = size(X);
    weights = ones(n, 1);
    bias = 0; 

    for i = 1:iter
        z = (X * weights) + bias; 
        h = sigmoid(z); 

        dw = (1/m) * (X' * (h-y)); % change in weights
        db = (1/m) * sum(h-y); % change in bias          
  
        weights = weights - (learningRate * dw); % update weights
        bias = bias - (learningRate * db); % update bias 
    end
end

function h = predict_proba(X, weights, bias)
        z = (X * weights) + bias; 
        h = sigmoid(z); 
end