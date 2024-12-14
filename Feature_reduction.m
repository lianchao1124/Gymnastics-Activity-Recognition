clear;
clc;

load('feature3.mat');
load('label.mat');
label=label;

%% 随机划分数据集

mm=1;
for k=1:10:300
    sm=size(feature3(1,1:k:end),2);
    idx = fscchi2(feature3,label);
    % W = ld_low(X, y, k);
    %idx = fscmrmr(feature3,label);
    W=feature3(:,idx(1:sm));

feature=W;

for sd=1:7 % subject1-7

test_interval = 7;

rng(1); % 设置随机数种子，以确保结果可重复
num_samples = size(feature, 1);
num_test = ceil(num_samples / test_interval);
num_train = num_samples - num_test;

test_indices = sd:test_interval:num_samples;
train_indices = setdiff(1:num_samples, test_indices);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_features = feature(train_indices,:);
train_labels = label(train_indices);
test_features = feature(test_indices,:);
test_labels = label(test_indices);

% 使用最近邻进行分类
knn_model = fitcknn(train_features, train_labels,'NumNeighbors',3,'Standardize',0);
knn_predicted_labels = predict(knn_model, test_features);
knn_accuracy(sd)  = sum(knn_predicted_labels == test_labels) / numel(test_labels);
disp(['K-Nearest Neighbors Accuracy: ', num2str(knn_accuracy)]);

end
clear feature
pa(mm)=mean(knn_accuracy);
mm=mm+1;
end



