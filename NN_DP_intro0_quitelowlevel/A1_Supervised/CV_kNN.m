function [k_opt,acc_cv]=CV_kNN(XTrain,LTrain,partitions,k_max)
%Crossvalidation used to choose optimal parameter k.
%   Inputs:
%           XTrain  -  sample features
%           LTrain  -  sample labels
%           partitions      the number of partions of the data
%           k_max       % maximum number of k
% 
%   Outputs:
%           k_opt  - the k value with best accuracy
%           acc_cv     -  best accuracy with regard to k_opt
% 

% -----------------just for testing
% 
%[k,mean_acc]= CV_kNN(X_All, L_All,partitions,k_max)
 %XTrain = X_All;
 %LTrain = L_All;
 %partitions=3;
 %k_max=30;
% ----------------------

acc=zeros(k_max,1);
acc_tmp=zeros(partitions,1);
[N,~] = size(XTrain);
part_size = floor(N/partitions);  %number of points for each parts

for k=1:k_max
    for i=1:partitions
        % take one subset as test data
        XTest = XTrain(((i-1)*part_size+1):i*part_size,:);   
        LTest = LTrain(((i-1)*part_size+1):i*part_size,:);
        
        XTrain_ = XTrain;
        LTrain_ = LTrain;
        
        % rest of data for training
        XTrain_(((i-1)*part_size+1):i*part_size,:) = []; 
        LTrain_(((i-1)*part_size+1):i*part_size,:) = [];
        %calculate accuracy for each iteration
        LPred = kNN(XTest, k, XTrain_, LTrain_);
        cM = calcConfusionMatrix( LPred, LTest );
        acc_tmp(i)  = calcAccuracy( cM );  
    end
    % calculate mean accuracy for certain k.
    acc(k) = mean(acc_tmp);

end
% sort the accuracy and get the optimal k with highest mean accuracy
[value,ind] = sort(acc,'descend');
k_opt = ind(1);
acc_cv = value(1);

end
