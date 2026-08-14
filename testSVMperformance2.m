clear; 
clc; 
%Number of iterations to run the hyperparameter tuning
iterations = 10;
%Whether or not to recreate SVMs from scratch
fresh = false;

%Create masks
load("initial_LST_features.mat", "Y", "dataSplit");
trainMask = (dataSplit == "train");
validMask = (dataSplit == "validation");

Y_Train = Y(trainMask);
Y_Valid = Y(validMask);

function [accuracy, classAUCs] = createSVM_wrapper(filename, X_train, y_train, X_valid, y_valid, type, fresh, iterations, coding)

    if (coding == "onevsone")
        filename = extractBefore(filename, strlength(filename)-3) + "_1v1.mat";
    end
    %fprintf(filename);

    if (isfile(filename) & ~fresh)
        load(filename);
    else
        fprintf("Creating %s SVM...", type)
        svm = createSVM(iterations, X_train, y_train, type, coding);
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

% Define Individual Train Sets
X_HOG_Train = X_HOG(trainMask, :); 
X_LST_Train = X_LST(trainMask, :); 
X_shape_Train = X_shape(trainMask, :); 

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

% Initialize data for figures
featureNames = {
    "HOG"
    "LST"
    "Shape"
    "LST + HOG"
    "LST + Shape"
    "HOG + Shape"
    "LST + HOG + Shape"
    };
methodNames = {
    "One-vs-All"
    "One-vs-One"
    };
accuracies = zeros(7, 2);
meanAUC = zeros(7, 2);

% One vs. all

% Individual SVM
[accuracies(1, 1), auc] = createSVM_wrapper("HOG_SVM.mat", X_HOG_Train, Y_Train, ...
                                    X_HOG_Valid, Y_Valid, "HOG", fresh, iterations, "onevsall"); 
meanAUC(1,1) = mean(auc);

[accuracies(2, 1), auc] = createSVM_wrapper("LST_SVM.mat", X_LST_Train, Y_Train, ...
                                    X_LST_Valid, Y_Valid, "LST", fresh, iterations, "onevsall"); 
meanAUC(2,1) = mean(auc);

[accuracies(3, 1), auc] = createSVM_wrapper("Shape_SVM.mat", X_shape_Train, Y_Train, ...
                                    X_shape_Valid, Y_Valid, "Shape", fresh, iterations, "onevsall"); 
meanAUC(3,1) = mean(auc);

% Pair SVM
[accuracies(4, 1), auc] = createSVM_wrapper("LST_HOG_SVM.mat", X_COMB_LST_HOG_Train, Y_Train, ...
                                    X_COMB_LST_HOG_Valid, Y_Valid, "Combined LST HOG", fresh, iterations, "onevsall"); 
meanAUC(4,1) = mean(auc);

[accuracies(5, 1), auc] = createSVM_wrapper("LST_Shape_SVM.mat", X_COMB_LST_SHAPE_Train, Y_Train, ...
                                    X_COMB_LST_SHAPE_Valid, Y_Valid, "Combined LST Shape", fresh, iterations, "onevsall"); 
meanAUC(5,1) = mean(auc);

[accuracies(6, 1), auc] = createSVM_wrapper("HOG_Shape_SVM.mat", X_COMB_HOG_SHAPE_Train, Y_Train, ...
                                    X_COMB_HOG_SHAPE_Valid, Y_Valid, "Combined HOG Shape", fresh, iterations, "onevsall"); 
meanAUC(6,1) = mean(auc);

% Trio SVM 
[accuracies(7, 1), auc] = createSVM_wrapper("LST_HOG_Shape_SVM.mat", X_COMB_LST_HOG_SHAPE_Train, Y_Train, ...
                                    X_COMB_LST_HOG_SHAPE_Valid, Y_Valid, "Combined LST HOG Shape", fresh, iterations, "onevsall"); 
meanAUC(7,1) = mean(auc);

% One vs One
% Individual SVM
[accuracies(1, 2), auc] = createSVM_wrapper("HOG_SVM.mat", X_HOG_Train, Y_Train, ...
    X_HOG_Valid, Y_Valid, "HOG", fresh, iterations, "onevsone"); 
meanAUC(1,2) = mean(auc);

[accuracies(2, 2), auc] = createSVM_wrapper("LST_SVM.mat", X_LST_Train, Y_Train, ...
    X_LST_Valid, Y_Valid, "LST", fresh, iterations, "onevsone"); 
meanAUC(2,2) = mean(auc);

[accuracies(3, 2), auc] = createSVM_wrapper("Shape_SVM.mat", X_shape_Train, Y_Train, ...
    X_shape_Valid, Y_Valid, "Shape", fresh, iterations, "onevsone"); 
meanAUC(3,2) = mean(auc);

% Pair SVM
[accuracies(4, 2), auc] = createSVM_wrapper("LST_HOG_SVM.mat", X_COMB_LST_HOG_Train, Y_Train, ...
    X_COMB_LST_HOG_Valid, Y_Valid, "Combined LST HOG", fresh, iterations, "onevsone"); 
meanAUC(4,2) = mean(auc);

[accuracies(5, 2), auc] = createSVM_wrapper("LST_Shape_SVM.mat", X_COMB_LST_SHAPE_Train, Y_Train, ...
    X_COMB_LST_SHAPE_Valid, Y_Valid, "Combined LST Shape", fresh, iterations, "onevsone"); 
meanAUC(5,2) = mean(auc);

[accuracies(6, 2), auc] = createSVM_wrapper("HOG_Shape_SVM.mat", X_COMB_HOG_SHAPE_Train, Y_Train, ...
    X_COMB_HOG_SHAPE_Valid, Y_Valid, "Combined HOG Shape", fresh, iterations, "onevsone"); 
meanAUC(6,2) = mean(auc);

% Trio SVM 
[accuracies(7, 2), auc] = createSVM_wrapper("LST_HOG_Shape_SVM.mat", X_COMB_LST_HOG_SHAPE_Train, Y_Train, ...
    X_COMB_LST_HOG_SHAPE_Valid, Y_Valid, "Combined LST HOG Shape", fresh, iterations, "onevsone"); 
meanAUC(7,2) = mean(auc);

figure;
bar(accuracies);

set(gca, ...
    'XTick', 1:length(featureNames), ...
    'XTickLabel', featureNames);

ylabel('Validation Accuracy');
xlabel('Feature Representation');
legend(methodNames, 'Location', 'best');
title('SVM Classification Accuracy');
ylim([0 1]);
grid on;

figure;
bar(meanAUC);

set(gca, ...
    'XTick', 1:length(featureNames), ...
    'XTickLabel', featureNames);

ylabel('Macro-Average AUC');
xlabel('Feature Representation');
legend(methodNames, 'Location', 'best');
title('SVM AUC Performance');
ylim([0 1]);
grid on;
% %Create LST svm if not already present
% 
% 
% %Create HOG svm if not already present
% if (isfile("HOG_SVM_baseline.mat") & ~fresh)
%     load("HOG_SVM_baseline.mat");
% else
%     fprintf("Creating HOG SVM...")
%     load("initial_HOG_features.mat", "X_HOG", "Y", "dataSplit");
% 
%     X_HOG_Train = X_HOG(trainMask);
% 
%     hog_svm = createSVM(iterations, X_HOG_Train, Y_Train, "HOG");
%     save("HOG_SVM_baseline", "hog_svm");
% end
% 
% %Create shape svm if not already present
% if (isfile("shape_SVM_baseline.mat") & ~fresh)
%     load("shape_SVM_baseline.mat");
% else
%     fprintf("Creating Shape SVM...")
%     load("initial_shape_features.mat", "X_shape");
% 
%     X_shape_Train = X_shape(trainMask);
% 
%     shape_svm = createSVM(iterations, X_shape_Train, Y_Train, "Shape");
%     save("shape_SVM_baseline", "shape_svm");
% end
% 
% %Create combined LST-HOG svm if not already present
% if (isfile("LST_HOG_SVM_baseline.mat") & ~fresh)
%     load("LST_HOG_SVM_baseline.mat");
% else
%     fprintf("Creating Combined SVM...")
%     load("initial_HOG_features.mat", "X_HOG");
%     load("initial_LST_features.mat", "X_LST");
% 
%     X_COMB = [X_HOG, X_LST];
%     X_COMB_Train = X_COMB(trainMask);
% 
%     comb_svm = createSVM(iterations, X_COMB_Train, Y_Train, "Combined_LST_HOG");
%     save("comb_SVM_baseline", "comb_svm");
% end
% 
% 
% %Create LST-Shape svm if not already present
% if (isfile("LST_SHAPE_SVM_baseline.mat") & ~fresh)
%     load("LST_SHAPE_SVM_baseline.mat");
% else
%     fprintf("Creating LST-shape SVM...")
%     load("initial_shape_features.mat", "X_shape");
%     load("initial_LST_features.mat", "X_LST");
% 
%     X_COMB = [X_shape, X_LST];
%     X_COMB_Train = X_COMB(trainMask);
% 
%     lst_shape_svm = createSVM(iterations, X_COMB_Train, Y_Train, "Combined_LST_SHAPE");
%     save("LST_SHAPE_SVM_baseline", "shape_svm");
% end
% 
% %Create Shape-HOG svm if not already present
% if (isfile("HOG_SHAPE_SVM_baseline.mat") & ~fresh)
%     load("HOG_SHAPE_SVM_baseline.mat");
% else
%     fprintf("Creating HOG-shape SVM...")
%     load("initial_shape_features.mat", "X_shape");
%     load("initial_HOG_features.mat", "X_HOG");
% 
%     X_COMB = [X_shape, X_HOG];
%     X_COMB_Train = X_COMB(trainMask);
% 
%     hog_shape_svm = createSVM(iterations, X_COMB_Train, Y_Train, "Combined_HOG_Shape");
%     save("HOG_SHAPE_SVM_baseline", "hog_shape_svm");
% end
% 
% %Create Shape-HOG svm if not already present
% if (isfile("LST_HOG_SHAPE_SVM_baseline.mat") & ~fresh)
%     load("LST_HOG_SHAPE_SVM_baseline.mat");
% else
%     fprintf("Creating LST-HOG-Shape SVM...")
%     load("initial_shape_features.mat", "X_shape");
%     load("initial_HOG_features.mat", "X_HOG");
%     load("initial_LST_features.mat", "X_LST");
% 
%     X_COMB = [X_shape, X_HOG, X_LST];
%     X_COMB_Train = X_COMB(trainMask);
% 
%     lst_hog_shape_svm = createSVM(iterations, X_COMB_Train, Y_Train, "Combined_LST_HOG_Shape");
%     save("LST_HOG_SHAPE_SVM_baseline", "lst_hog_shape_svm");
% end
% 
% load("initial_HOG_features.mat", "X_HOG");
% load("initial_LST_features.mat", "X_LST");
% load("initial_shape_features.mat", "X_shape");
% 
% 
% 
% 
% [accuracy, classAUCs] = analyzeSVM(lst_svm, X_LST_Valid, Y_Valid);
% fprintf("\nLST SVM Results...\nOverall Accuracy: " + accuracy);% + "\nLST Class Accuracies: ");
% %celldisp(classAccuracies);
% fprintf("\nLST Class AUCs: ");
% disp(classAUCs);
% 
% [accuracy, classAUCs] = analyzeSVM(hog_svm, X_HOG_Valid, Y_Valid);
% fprintf("\nHOG SVM Results...\nOverall Accuracy: " + accuracy);% + "\nHOG Class Accuracies: ");
% %celldisp(classAccuracies);
% fprintf("\nHOG Class AUCs: ")
% disp(classAUCs);
% 
% [accuracy, classAUCs] = analyzeSVM(shape_svm, X_shape_Valid, Y_Valid);
% fprintf("\nShape SVM Results...\nOverall Accuracy: " + accuracy);% + "\nShape Class Accuracies: ");
% %celldisp(classAccuracies);
% fprintf("\nShape Class AUCs: ");
% disp(classAUCs);
% 
% [accuracy, classAUCs] = analyzeSVM(comb_svm, X_COMB_LST_HOG_Valid, Y_Valid);
% fprintf("\nCombined LST-HOG SVM Results...\nOverall Accuracy: " + accuracy);% + "\nCombined Class Accuracies: ");
% %celldisp(classAccuracies);
% fprintf("\nCombined Class AUCs: ")
% disp(classAUCs);
% 
% 
% [accuracy, classAUCs] = analyzeSVM(comb_svm, X_COMB_LST_HOG_Valid, Y_Valid);
% fprintf("\nCombined LST-HOG SVM Results...\nOverall Accuracy: " + accuracy);% + "\nCombined Class Accuracies: ");
% %celldisp(classAccuracies);
% fprintf("\nCombined Class AUCs: ")
% disp(classAUCs);



