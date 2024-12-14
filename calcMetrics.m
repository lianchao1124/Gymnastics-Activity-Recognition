function [avg_precision, avg_recall, avg_f1score] = calcMetrics(confMat)
    % 获取混淆矩阵的大小
    numClasses = size(confMat, 1);  % 类别数
    
    % 初始化精度、召回率和F1-score
    precision = zeros(1, numClasses);
    recall = zeros(1, numClasses);
    f1score = zeros(1, numClasses);

    % 计算每个类的精度、召回率和F1-score
    for i = 1:numClasses
        TP = confMat(i,i);  % True Positive for class i
        FP = sum(confMat(:,i)) - TP;  % False Positive for class i
        FN = sum(confMat(i,:)) - TP;  % False Negative for class i
        TN = sum(confMat(:)) - TP - FP - FN;  % True Negative for class i
        
        % Precision, Recall, F1-score for class i
        precision(i) = TP / (TP + FP);
        recall(i) = TP / (TP + FN);
        f1score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    end
    
    % 计算宏平均
    avg_precision = mean(precision);  % 平均精度
    avg_recall = mean(recall);        % 平均召回率
    avg_f1score = mean(f1score);      % 平均F1-score