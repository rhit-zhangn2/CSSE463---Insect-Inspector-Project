function mdl = createSVM(iter, x, y, type, coding)

mkdir("SavedOutputs/SVM/" + type);

diary("SavedOutputs/SVM/" + type + "/" + iter + "Log");

hpoOptions = hyperparameterOptimizationOptions(AcquisitionFunctionName="expected-improvement-plus", MaxObjectiveEvaluations=iter);
t = templateSVM('Standardize', true);
mdl = fitcecoc(x, y, 'Learners', t, OptimizeHyperparameters={'BoxConstraint', 'KernelScale', 'KernelFunction', 'PolynomialOrder'} ...
    ,HyperparameterOptimizationOptions=hpoOptions, Coding=coding);

diary off
