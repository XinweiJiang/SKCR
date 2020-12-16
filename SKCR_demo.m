%     demo for SKCR classification algorithm
%--------------Brief description-------------------------------------------
%
%
% This demo implements the  SKCR hyperspectral image classification [1]
%
%
% More details in:
%
% [1]B. Tu, C. Zhou, X. Liao, G. Zhang and Y. Peng, "Spectral-Spatial 
% Hyperspectral Classification via Structural-Kernel Collaborative Representation," 
% in IEEE Geoscience and Remote Sensing Letters, doi: 10.1109/LGRS.2020.2988124.%
%
% contact: tubing@hnist.edu.cn (Bing Tu); chengle_zhou@foxmail.com (Chengle Zhou)
close all; clear all; clc
CA_new = [];

%% add related fuctions to path 
addpath data
addpath dual_kernel
addpath utilities

%% load indian_pines  data
load IndiaP
load Indian_pines_gt

tim2 = indian_pines_gt;
tim1 = double(tim2);
[xi,yi] = find(tim1 == 0);
xisize = size(xi);

no_class = max(indian_pines_gt(:));
[rows, cols, band_ori] = size(img);
img_ori = img;
img = reshape(img, rows * cols, band_ori);

%% PCA reduce dimension
if(~exist('superpixel_labels.mat'))
    img_pca = compute_mapping(img,'PCA',3);     %
    [img_pca] = scale_new(img_pca);
    superpixel_data = reshape(img_pca,[rows, cols, 3]);
    superpixel_data=mat2gray(superpixel_data);
    superpixel_data=im2uint8(superpixel_data);
    % superpixels
    number_superpixels = 200;
    lambda_prime = 0.8;  sigma = 10;  conn8 = 1;
    SuperLabels = mex_ers(double(superpixel_data),number_superpixels,lambda_prime,sigma,conn8);
    save SuperLabels superpixel_labels
else
    load superpixel_labels
end

%% Generate mean feature map
k = 0.5;
[mean_matix,super_img,indexes] = Kmean_feature(img_ori,SuperLabels,k);

%% Construct training and test datasets
train_num = [3,20,13,4,8,10,3,8,3,14,31,9,4,17,10,4]; % the rate of training is set to 1.5% of ground thruth
indexes = train_random_select(GroundT(2,:),train_num); % based on 24 for each class
train_SL = GroundT(:,indexes);
test_SL = GroundT;
test_SL(:,indexes) = [];

train_samples = img(train_SL(1,:),:);
train_labels = train_SL(2,:);
test_samples = img(test_SL(1,:),:);
GroudTest = test_SL(2,:);

%% Generate spectral feature
train_img = zeros(rows,cols);          
train_img(train_SL(1,:)) = train_SL(2,:);

%%
in_param.nfold = 5;  % cross validation for parameters
in_param.alpha = 0.05; 
in_param.beta = 1 - in_param.alpha; 
[SKCR_results, SKCE_out_param] = classify_svm_mykernel(img_ori,mean_matix,train_img,in_param);
[OA,kappa,AA,CA] = calcError(test_SL(2,:)-1,SKCR_results(test_SL(1,:))'-1,[1:no_class]);

%% Display the result of SKCR
results  = reshape(SKCR_results,[rows, cols]);
for i = 1:xisize
      results (xi(i),yi(i)) = 0;
 end
results = reshape(results,[rows, cols]); 
SKCR_map = label2color(results,'india');
figure,imshow(SKCR_map);
