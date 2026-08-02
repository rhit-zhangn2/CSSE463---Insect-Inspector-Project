function [accuracy, classAUCs] = analyzeSVM(mdl, xValid, yValid)

[label, score] = predict(mdl, xValid);

accuracy = sum((label == yValid), "all") / numel(yValid);
%c = {"c", sum((label == yValid & label == "cicadomorpha"), "all") / numel((yValid == "cicadomorpha"))};
%f = {"f", sum((label == yValid & label == "fulgoromorpha"), "all") / numel((yValid == "fulgoromorpha"))};
%h = {"h", sum((label == yValid & label == "heteroptera"), "all") / numel((yValid == "heteroptera"))};
%s = {"s", sum((label == yValid & label == "sternorrhyncha"), "all") / numel((yValid == "sternorrhyncha"))};

%classAccuracies = {c; f; h; s};

rocObj = rocmetrics(yValid, score, mdl.ClassNames);

classAUCs = auc(rocObj);