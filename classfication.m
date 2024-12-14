clear;
clc;

load('feature3.mat');
load('label.mat');

feature=feature3;
%% 按照受测者划分数据集，7名受测者
for sd=1:7   %%subject
% %% 受测者按照顺序排布
test_subject = 7;

% 留一法分割数据集为训练集和测试集
num_samples = size(feature, 1);
num_test = num_samples / test_subject;
num_train = num_samples - num_test;

test_indices = sd:test_subject:num_samples;% sub1-sub7
train_indices = setdiff(1:num_samples, test_indices);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_features = feature(train_indices,:);
train_labels = label(train_indices);
test_features = feature(test_indices,:);
test_labels = label(test_indices);

% train_features = zscore(train_features);
% test_features = zscore(test_features);
rng(1)
% 使用支持向量机 (SVM) 进行分类
    t = templateSVM('Standardize', true, 'KernelFunction', 'linear');
    svm_model = fitcecoc(train_features, train_labels, 'Learners', t);
    svm_predicted_labels = predict(svm_model, test_features);
    svm_accuracy(sd) = sum(svm_predicted_labels == test_labels) / numel(test_labels);
    
    % 计算混淆矩阵
    [confMat, order] = confusionmat(test_labels, svm_predicted_labels);
    [svm_precision, svm_recall, svm_f1score] = calcMetrics(confMat);  % 计算precision, recall, f1-score
    
    % 存储指标
    svm_precision_values(sd) = svm_precision;
    svm_recall_values(sd) = svm_recall;
    svm_f1score_values(sd) = svm_f1score;
    
    % 输出 SVM 结果
    disp(['SVM Accuracy of subject ', num2str(sd), ': ', num2str(svm_accuracy(sd))]);
    disp(['SVM Precision of subject ', num2str(sd), ': ', num2str(svm_precision)]);
    disp(['SVM Recall of subject ', num2str(sd), ': ', num2str(svm_recall)]);
    disp(['SVM F1-score of subject ', num2str(sd), ': ', num2str(svm_f1score)]);

    % 使用随机森林进行分类
    rf_model = TreeBagger(130, train_features, train_labels);
    rf_predicted_labels = str2double(predict(rf_model, test_features));
    rf_accuracy(sd)  = sum(rf_predicted_labels == test_labels) / numel(test_labels);
    
    [confMat, order] = confusionmat(test_labels, rf_predicted_labels);
    [rf_precision, rf_recall, rf_f1score] = calcMetrics(confMat);
    
    % 存储指标
    rf_precision_values(sd) = rf_precision;
    rf_recall_values(sd) = rf_recall;
    rf_f1score_values(sd) = rf_f1score;
    
    % 输出随机森林结果
    disp(['Random Forest Accuracy of subject ', num2str(sd), ': ', num2str(rf_accuracy(sd))]);
    disp(['Random Forest Precision of subject ', num2str(sd), ': ', num2str(rf_precision)]);
    disp(['Random Forest Recall of subject ', num2str(sd), ': ', num2str(rf_recall)]);
    disp(['Random Forest F1-score of subject ', num2str(sd), ': ', num2str(rf_f1score)]);

    % 使用XGBoost进行分类
    xgb_model = fitcensemble(train_features, train_labels, 'Method', 'AdaBoostM2', 'NumLearningCycles', 400, 'Learners', 'Tree');
    xgb_predicted_labels = predict(xgb_model, test_features);
    xgb_accuracy(sd)  = sum(xgb_predicted_labels == test_labels) / numel(test_labels);
    
    [confMat, order] = confusionmat(test_labels, xgb_predicted_labels);
    [xgb_precision, xgb_recall, xgb_f1score] = calcMetrics(confMat);
    
    % 存储指标
    xgb_precision_values(sd) = xgb_precision;
    xgb_recall_values(sd) = xgb_recall;
    xgb_f1score_values(sd) = xgb_f1score;
    
    % 输出XGBoost结果
    disp(['XGBoost Accuracy of subject ', num2str(sd), ': ', num2str(xgb_accuracy(sd))]);
    disp(['XGBoost Precision of subject ', num2str(sd), ': ', num2str(xgb_precision)]);
    disp(['XGBoost Recall of subject ', num2str(sd), ': ', num2str(xgb_recall)]);
    disp(['XGBoost F1-score of subject ', num2str(sd), ': ', num2str(xgb_f1score)]);

    % 使用多层感知器（MLP）进行分类
    mlp_model = fitcnet(train_features, train_labels, "LayerSizes", [100 20]);
    mlp_predicted_labels = predict(mlp_model, test_features);
    mlp_accuracy(sd)  = sum(mlp_predicted_labels == test_labels) / numel(test_labels);
    
    [confMat, order] = confusionmat(test_labels, mlp_predicted_labels);
    [mlp_precision, mlp_recall, mlp_f1score] = calcMetrics(confMat);
    
    % 存储指标
    mlp_precision_values(sd) = mlp_precision;
    mlp_recall_values(sd) = mlp_recall;
    mlp_f1score_values(sd) = mlp_f1score;
    
    % 输出MLP结果
    disp(['MLP Accuracy of subject ', num2str(sd), ': ', num2str(mlp_accuracy(sd))]);
    disp(['MLP Precision of subject ', num2str(sd), ': ', num2str(mlp_precision)]);
    disp(['MLP Recall of subject ', num2str(sd), ': ', num2str(mlp_recall)]);
    disp(['MLP F1-score of subject ', num2str(sd), ': ', num2str(mlp_f1score)]);

    % 使用决策树进行分类
    lda_model = fitcdiscr(train_features, train_labels);
    lda_predicted_labels = predict(lda_model, test_features);
    lda_accuracy(sd) = sum(lda_predicted_labels == test_labels) / numel(test_labels);
    
    [confMat, order] = confusionmat(test_labels, lda_predicted_labels);
    [lda_precision, lda_recall, lda_f1score] = calcMetrics(confMat);
    
    % 存储指标
    lda_precision_values(sd) = lda_precision;
    lda_recall_values(sd) = lda_recall;
    lda_f1score_values(sd) = lda_f1score;
    
    % 输出LDA结果
    disp(['LDA Accuracy of subject ', num2str(sd), ': ', num2str(lda_accuracy(sd))]);
    disp(['LDA Precision of subject ', num2str(sd), ': ', num2str(lda_precision)]);
    disp(['LDA Recall of subject ', num2str(sd), ': ', num2str(lda_recall)]);
    disp(['LDA F1-score of subject ', num2str(sd), ': ', num2str(lda_f1score)]);

    % 使用最近邻进行分类
    knn_model = fitcknn(train_features, train_labels, 'NumNeighbors', 3, 'Standardize', 0);
    knn_predicted_labels = predict(knn_model, test_features);
    knn_accuracy(sd)  = sum(knn_predicted_labels == test_labels) / numel(test_labels);
    
    [confMat, order] = confusionmat(test_labels, knn_predicted_labels);
    [knn_precision, knn_recall, knn_f1score] = calcMetrics(confMat);
    
    % 存储指标
    knn_precision_values(sd) = knn_precision;
    knn_recall_values(sd) = knn_recall;
    knn_f1score_values(sd) = knn_f1score;
    
    % 输出KNN结果
    disp(['K-Nearest Neighbors Accuracy of subject ', num2str(sd), ': ', num2str(knn_accuracy(sd))]);
    disp(['K-Nearest Neighbors Precision of subject ', num2str(sd), ': ', num2str(knn_precision)]);
    disp(['K-Nearest Neighbors Recall of subject ', num2str(sd), ': ', num2str(knn_recall)]);
    disp(['K-Nearest Neighbors F1-score of subject ', num2str(sd), ': ', num2str(knn_f1score)]);

    % 使用朴素贝叶斯进行分类
    nb_model = fitcnb(train_features, train_labels);
    nb_predicted_labels = predict(nb_model, test_features);
    nb_accuracy(sd)  = sum(nb_predicted_labels == test_labels) / numel(test_labels);
    
    [confMat, order] = confusionmat(test_labels, nb_predicted_labels);
    [nb_precision, nb_recall, nb_f1score] = calcMetrics(confMat);
    
    % 存储指标
    nb_precision_values(sd) = nb_precision;
    nb_recall_values(sd) = nb_recall;
    nb_f1score_values(sd) = nb_f1score;
    
    % 输出Naive Bayes结果
    disp(['Naive Bayes Accuracy of subject ', num2str(sd), ': ', num2str(nb_accuracy(sd))]);
    disp(['Naive Bayes Precision of subject ', num2str(sd), ': ', num2str(nb_precision)]);
    disp(['Naive Bayes Recall of subject ', num2str(sd), ': ', num2str(nb_recall)]);
    disp(['Naive Bayes F1-score of subject ', num2str(sd), ': ', num2str(nb_f1score)]);

end

macc=[svm_accuracy;rf_accuracy;xgb_accuracy;mlp_accuracy;lda_accuracy;knn_accuracy;nb_accuracy];
mean_acc=mean(macc,2);

mprecision = [svm_precision_values; rf_precision_values; xgb_precision_values; mlp_precision_values; lda_precision_values; knn_precision_values; nb_precision_values];
mean_precision = mean(mprecision, 2);  % 按行计算平均值，表示每个模型的平均精度

mrecall = [svm_recall_values; rf_recall_values; xgb_recall_values; mlp_recall_values; lda_recall_values; knn_recall_values; nb_recall_values];
mean_recall = mean(mrecall, 2);  % 按行计算平均值，表示每个模型的平均召回率

mf1score = [svm_f1score_values; rf_f1score_values; xgb_f1score_values; mlp_f1score_values; lda_f1score_values; knn_f1score_values; nb_f1score_values];
mean_f1score = mean(mf1score, 2);  % 按行计算平均值，表示每个模型的平均F1-score
