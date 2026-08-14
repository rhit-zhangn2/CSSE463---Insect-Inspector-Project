load("LST_features7blocks.mat", "X_LST", "Y", "dataSplit");
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

createSVM_wrapper("LST_SVM_7blocks.mat", X_LST(trainMask, :), Y_Train, ...
    X_LST(validMask, :), Y_Valid, "LST", false, 25);