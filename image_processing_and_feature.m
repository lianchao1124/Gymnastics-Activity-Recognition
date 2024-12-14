clear;
clc;
%%%%%extract feature from different images%%%%%%
% 指定包含图片的文件夹路径
parentFolder1 = 'C:\Users\CRY\Desktop\纹理TIM\image\mexh\';
% parentFolder2 = 'C:\Users\CRY\Desktop\ticao\other\cwty\';
% parentFolder3 = 'C:\Users\CRY\Desktop\ticao\other\cwtz\';

% % 循环遍历六个子文件夹
q=1;
for i = 1:6
    % 构造当前子文件夹路径3
    subFolder1 = fullfile(parentFolder1, sprintf('%d', i));
    % subFolder2 = fullfile(parentFolder2, sprintf('%d', i));
    % subFolder3 = fullfile(parentFolder3, sprintf('%d', i));

    % 获取当前子文件夹下的所有图片文件
    imageFiles1 = dir(fullfile(subFolder1, '*.png')); 
    % imageFiles2 = dir(fullfile(subFolder2, '*.png')); 
    % imageFiles3 = dir(fullfile(subFolder3, '*.png')); 

    % 循环遍历当前子文件夹下的所有图片
    % for j = 1:2
    for j = 1:280
        % 读取图片
        imagePath1 = fullfile(subFolder1, imageFiles1(j).name);
        % imagePath2 = fullfile(subFolder2, imageFiles2(j).name);
        % imagePath3 = fullfile(subFolder3, imageFiles3(j).name);

        img1 = imread(imagePath1);
        % img2 = imread(imagePath2);
        % img3 = imread(imagePath3);

        img1=im2gray(img1);

        thresh = graythresh(img1);
        s = im2bw(img1,0.45);
        se = strel('disk',5);
        closeBW = imclose(s,se);

        [horizontal_count, vertical_count] = count_pixels2(closeBW, 0,1);

        feature1(q,:) = [horizontal_count,vertical_count];


        label(q)=i;
        q=q+1;

    end
end

% feature=[feature1,feature2,feature3];

label=label';

feature2 = mapminmax(feature1',0,1);
feature3 =feature2';


%%%%%%%%%%%feature 3是全部的,特征选择前%%%%%%%%%

save('feature3.mat',"feature3");
save('label.mat',"label");
