function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);

% L_NN  -   The classes belonging to the k closest neighbours
% class_num  -   The total amount of each class in L_NN
% avg_dist_c  -   The average-distance for each class to the sample point
% dist_c -   The class, distance and index of each point for the sample point
% Xrow   -   sample size of X
% Xcol    -   X dimension

LPred  = zeros(size(X,1),1);
L_NN = zeros(k,1);
class_num = zeros(NClasses, 1);
[Xrow,Xcol]=size(X);

% here the distance measure is euclidean distance

for i = 1:Xrow
    Dist=[];
    for j = 1:Xcol
        Dist = [Dist (XTrain(:,j)-X(i,j)).^2];
    end
    dist=sqrt(sum(Dist,2));
    [~,index] = sort(dist,1,'ascend');
    L_NN(:,1) = LTrain(index(1:k));
    for n=1:NClasses
        class_num(n,:) = sum(L_NN == classes(n));
    end
    if unique(L_NN) == 1
        %one point
        LPred(i,:) = L_NN(1,1);
    else
        if NClasses == length(unique(class_num))
            %no tie
            [~,I] = max(class_num);
            LPred(i,:) = I;
        else
            % when a tie happens
            avg_dist_c = zeros(NClasses, 1);
            dist_c = [L_NN, dist(index(1:k)), index(1:k)];
            for n=1:NClasses
                idx=dist_c(:,1)==n;
                avg_dist_c(n,:) = mean(dist_c(idx,2));
            end
            [~,I] = min(avg_dist_c);
            LPred(i,:) = I;
        end
    end
  
end
end

