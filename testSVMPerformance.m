%Number of iterations to run the hyperparameter tuning
iterations = 200;
%Whether or not to recreate SVMs from scratch
fresh = false;

%Create masks
load("initial_LST_features.mat", "Y", "dataSplit");
trainMask = (dataSplit == "train");
validMask = (dataSplit == "validation");

Y_Train = Y(trainMask);
Y_Valid = Y(validMask);

%Create LST svm if not already present
if (isfile("LST_SVM_baseline.mat") & ~fresh)
    load("LST_SVM_baseline.mat");
else
    fprintf("Creating LST SVM...")
    load("initial_LST_features.mat", "X_LST");
    
    X_LST_Train = X_LST(trainMask);
    
    lst_svm = createSVM(iterations, X_LST_Train, Y_Train, "LST");
    
    save("LST_SVM_baseline", "lst_svm");
end

%Create HOG svm if not already present
if (isfile("HOG_SVM_baseline.mat") & ~fresh)
    load("HOG_SVM_baseline.mat");
else
    fprintf("Creating HOG SVM...")
    load("initial_HOG_features.mat", "X_HOG", "Y", "dataSplit");

    X_HOG_Train = X_HOG(trainMask);

    hog_svm = createSVM(iterations, X_HOG_Train, Y_Train, "HOG");
    save("HOG_SVM_baseline", "hog_svm");
end

%Create combined svm if not already present
if (isfile("comb_SVM_baseline.mat") & ~fresh)
    load("comb_SVM_baseline.mat");
else
    fprintf("Creating Combined SVM...")
    load("initial_HOG_features.mat", "X_HOG");
    load("initial_LST_features.mat", "X_LST");

    X_COMB = [X_HOG, X_LST];
    X_COMB_Train = X_COMB(trainMask);

    comb_svm = createSVM(iterations, X_COMB_Train, Y_Train, "Combined");
    save("comb_SVM_baseline", "comb_svm");
end

load("initial_HOG_features.mat", "X_HOG");
load("initial_LST_features.mat", "X_LST");

X_HOG_Valid = X_HOG(validMask);
X_LST_Valid = X_LST(validMask);
X_COMB_Valid = [X_HOG, X_LST];
X_COMB_Valid = X_COMB_Valid(validMask);

[accuracy, classAUCs] = analyzeSVM(lst_svm, X_LST_Valid, Y_Valid);
fprintf("\nLST SVM Results...\nOverall Accuracy: " + accuracy);% + "\nLST Class Accuracies: ");
%celldisp(classAccuracies);
fprintf("\nLST Class AUCs: ");
disp(classAUCs);

[accuracy, classAUCs] = analyzeSVM(hog_svm, X_HOG_Valid, Y_Valid);
fprintf("\nHOG SVM Results...\nOverall Accuracy: " + accuracy);% + "\nHOG Class Accuracies: ");
%celldisp(classAccuracies);
fprintf("\nHOG Class AUCs: ")
disp(classAUCs);

[accuracy, classAUCs] = analyzeSVM(comb_svm, X_COMB_Valid, Y_Valid);
fprintf("\nCombined SVM Results...\nOverall Accuracy: " + accuracy);% + "\nCombined Class Accuracies: ");
%celldisp(classAccuracies);
fprintf("\nCombined Class AUCs: ")
disp(classAUCs);