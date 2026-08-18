% Confusion matrix
%% Warning: this likes crashing 
clear;  
clc; 
type = "EfficientNetB0"; 

fprintf("Loading features\n");
load("final_CNN_features.mat", "X_CNN", "Y");
fprintf("Loaded features\n")
load("final_dataSplit.mat", "dataSplit");
trainMask = (dataSplit == "train");
validMask = (dataSplit == "validation");
testMask = (dataSplit == "test"); 

y_train = Y(trainMask);
y_valid = Y(validMask);
y_test = Y(testMask); 
[Y_numeric, gnames] = grp2idx(Y);

X_train_cnn = X_CNN(trainMask, :);
X_valid_cnn = X_CNN(validMask, :);
X_test_cnn = X_CNN(testMask, :);

filenames = [dir("initial_subset_2000\initial_subset_2000\images\cicadomorpha\")]

fprintf("Loading trained network\n"); 
nn = load("NN_EfficientnetB0FINAL02.mat"); 
nn = nn.nn; 
fprintf("Loaded trained network\n");  

[accuracy, classAUCs, cm] = analyzeNN(nn, X_test_cnn, y_test, gnames);
fprintf("\n%s Neural Network Results...\nOverall Accuracy: %.4f ", type, accuracy);
fprintf("\n%s Class AUCs: ", type);
disp(classAUCs);
fprintf("\n Confusion Matrix:\n");
disp(cm); 

function confusionMatrix = generateConfusionMatrix(y_valid, y_pred)
% first character: actual
% second character: predicted 

c_c = sum(y_valid == "cicadomorpha" & y_pred == "cicadomorpha"); 
c_f = sum(y_valid == "cicadomorpha" & y_pred == "fulgoromorpha");
c_h = sum(y_valid == "cicadomorpha" & y_pred == "heteroptera");
c_s = sum(y_valid == "cicadomorpha" & y_pred == "sternorrhyncha");

f_c = sum(y_valid == "fulgoromorpha" & y_pred == "cicadomorpha"); 
f_f = sum(y_valid == "fulgoromorpha" & y_pred == "fulgoromorpha");
f_h = sum(y_valid == "fulgoromorpha" & y_pred == "heteroptera");
f_s = sum(y_valid == "fulgoromorpha" & y_pred == "sternorrhyncha");

h_c = sum(y_valid == "heteroptera" & y_pred == "cicadomorpha"); 
h_f = sum(y_valid == "heteroptera" & y_pred == "fulgoromorpha");
h_h = sum(y_valid == "heteroptera" & y_pred == "heteroptera");
h_s = sum(y_valid == "heteroptera" & y_pred == "sternorrhyncha");

s_c = sum(y_valid == "sternorrhyncha" & y_pred == "cicadomorpha"); 
s_f = sum(y_valid == "sternorrhyncha" & y_pred == "fulgoromorpha");
s_h = sum(y_valid == "sternorrhyncha" & y_pred == "heteroptera");
s_s = sum(y_valid == "sternorrhyncha" & y_pred == "sternorrhyncha");


confusionMatrix = [c_c, c_f, c_h, c_s; 
    f_c, f_f, f_h, f_s;
    h_c, h_f, h_h, h_s; 
    s_c, s_f, s_h, s_s; 
    ]; 
end



% Analyze
function [accuracy, classAUCs, confusionMatrix] = analyzeNN(mdl, xValid, yValid, gnames)
    [label, score] = predict(mdl, xValid);

    catLabels = gnames(label);
    classNames = gnames(mdl.ClassNames);
    arr_score = gather(score);

    rocObj = rocmetrics(yValid, arr_score, classNames);
    accuracy = sum((catLabels == yValid), "all") / numel(yValid);
    classAUCs = auc(rocObj);

    confusionMatrix = generateConfusionMatrix(yValid, catLabels); 

end

