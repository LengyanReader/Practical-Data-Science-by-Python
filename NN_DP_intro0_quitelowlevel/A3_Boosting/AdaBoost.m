clear

%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 75;   % random integers
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 1000;   % set
% Number of weak classifiers
nbrWeakClassifiers = 35;   % set

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

% ctrl+r/t, about adding or deleting multiple lines of annotations

% printing picture samples

% figure(1);
% colormap gray;
% for k=1:25
%     subplot(5,5,k), imagesc(faces(:,:,10*k));
%     axis image;
%     axis off;
% end
% 
% figure(2);
% colormap gray;
% for k=1:25
%     subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
%     axis image;
%     axis off;
% end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

% figure(3);
% colormap gray;
% for k = 1:25
%     subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
%     axis image;
%     axis off;
% end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError

P = 1; %here let the polarity be +1 at this step. 
N = size(xTrain,1); %feature numbers/dimensions
M_train = size(xTrain,2); %number of training samples
Dt(1:M_train) = 1/M_train; % weight of x

pn=2;  % a paramater to T from nbrWeakClassifiers
T = floor((nbrWeakClassifiers)*pn); %number of base classifiers, using T is only for convenience

h = zeros(T,M_train);% store the h(x), namely, threshold function of each time(iteration)
%E_sum = zeros(1,T);    % store...
%Err_N = zeros(1,N);----------------------
alpha = zeros(1,T);  % weights sth
% best_thred_n = zeros(1,N);-------------
best_thred_t = zeros(1,T);  % for the best
best_polarity_t = zeros(1,T);
best_feature = zeros(1,T);
best_thred_n = 0;% threshold function for the responding feature
Err_N = zeros(1,N);  % for all the dataset
Err_M = 0;  % the mean value of errors
best_polarity_n = P;%n
best_polarity_m = P;%m
best_m = 0; %
best_f = 0; %?

for t=1:T %weak classifiers begin
    min_err = inf;
    for f=1:N %features/dimensions, one after one, to look for which feature is best to be the classification      
        Err_M_min = inf;
        %-sample points----------------------------------------------------------------->
        for m=1:M_train %thresholds (& samples), sample points ,one by one
            P = 1;  % set a polarity
            h(t,:) = WeakClassifier(xTrain(f,m), P, xTrain(f,:)); % getting h(t)
            Err_M = WeakClassifierError(h(t,:), Dt, yTrain);
            % Err_M(s) = Err_M;
            if Err_M > 0.5
                P = -1;
                Err_M = 1-Err_M;   
            end
            if Err_M_min > Err_M  % minizing errors
                best_m = m;
                Err_M_min = Err_M;
                best_polarity_m = P;
            end
        %-sample points-----------------------------------------------------------------|
        end
        %[m,best_m]=min(Err_M);
        Err_N = Err_M_min;  % for all features
        if min_err > Err_N
            min_err = Err_N;
            best_f = f;
            best_polarity_n = best_polarity_m;
            best_thred_n = xTrain(best_f,best_m);
        end
    end
    %[min_err,best_f] = min(Err_N);
    
    % the processing 
    % what have done above is to find best threshold, polarity and best feature for each weak classifier
    % now entering 
    best_thred_t(t) = best_thred_n;
    best_polarity_t(t) = best_polarity_n;
    best_feature(t) = best_f;
    %E_sum(t) = min_err;
    
    %% update alpha
    alpha(t) = 0.5*log((1-min_err)/(min_err+0.00001));
    
    %% update weight
    h(t,:) = WeakClassifier(best_thred_t(t), best_polarity_t(t), xTrain(best_f,:));
    Dt = Dt.*exp(-alpha(t).*yTrain.*h(t,:));
    Dt = Dt/sum(Dt);
end

y_pred = zeros(1,size(xTrain,2));
M_train = size(xTrain,2);
for i=1:M_train
    y_temp = zeros(1,T);
    for t=1:(T)
        y_temp(t) = WeakClassifier(best_thred_t(t), best_polarity_t(t), xTrain(best_feature(t),i));    
    end
    H = sign(alpha*y_temp');
    y_pred(i) = H;
end



%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

y_pred = zeros(1,size(xTrain,2));
M_train = size(xTrain,2);
for i=1:M_train
    y_temp = zeros(1,T);
    for t=1:(T)
        y_temp(t) = WeakClassifier(best_thred_t(t), best_polarity_t(t), xTrain(best_feature(t),i));    
    end
    H = sign(alpha*y_temp');
    y_pred(i) = H;
end

cM_train = confusionmat(yTrain,y_pred)
diagonalSum = trace(cM_train);
acc_train = diagonalSum / sum(sum(cM_train))

y_pred_test = zeros(1,size(xTest,2));
M_test = size(xTest,2);
for i=1:M_test
    y_temp_test = zeros(1,T);
    for t=1:(T)
        y_temp_test(t) = WeakClassifier(best_thred_t(t), best_polarity_t(t), xTest(best_feature(t),i));    
    end
    H = sign(alpha*y_temp_test');
    y_pred_test(i) = H;
end

cM_test = confusionmat(yTest, y_pred_test)
diagonalSum_test = trace(cM_test);
acc_test = diagonalSum_test / sum(sum(cM_test))


%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

acc_train_all = zeros(1,T);
for num_weak = 1:T
    M_train = size(xTrain,2); 
    y_pred_1 = zeros(1,M_train);
    for i=1:M_train
        y_temp_1 = zeros(1,T);
        for t=1:num_weak
            y_temp_1(t) = WeakClassifier(best_thred_t(t), best_polarity_t(t), xTrain(best_feature(t),i));    
        end
        H_1 = sign(alpha*y_temp_1');
        y_pred_1(i) = H_1;
    end
cM_train_1 = confusionmat(yTrain,y_pred_1);
diagonalSum_1 = trace(cM_train_1);
acc_train_1 = diagonalSum_1 / sum(sum(cM_train_1));
acc_train_all(num_weak) = acc_train_1;
end

acc_test_all = zeros(1,T);
for num_weak = 1:T
    M_test = size(xTest,2); 
    y_pred_1 = zeros(1,M_test);
    for i=1:M_test
        y_temp_1 = zeros(1,T);
        for t=1:num_weak
            y_temp_1(t) = WeakClassifier(best_thred_t(t), best_polarity_t(t), xTest(best_feature(t),i));    
        end
        H_1 = sign(alpha*y_temp_1');
        y_pred_1(i) = H_1;
    end
cM_train_1 = confusionmat(yTest,y_pred_1);
diagonalSum_1 = trace(cM_train_1);
acc_train_1 = diagonalSum_1 / sum(sum(cM_train_1));
acc_test_all(num_weak) = acc_train_1;
end

figure(4)
plot(acc_train_all*100)
hold on
plot(acc_test_all*100)
hold off
grid on

title('\fontsize{15}Accuracy versus Number of Weak Classifier----')
xlabel('\fontsize{12}the number of weak classifier')
ylabel('\fontsize{12}Accuracy(%)')
text1 = "The amout of the test data set:"+string(M_test);
text2 = "The amout of the train data set:"+string(M_train);
text3 = "The features used here:"+string(num_unique_feature);
text(40,86,text1)
text(40,84,text2)
text(40,82,text3)
% M_test
% M_train
% num_unique_feature = size(unique_feature,2)

clear y_pred_1 y_temp_1 H_1 cM_train_1 diagonalSum_1 acc_train_1
%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.

err_index = find(y_pred ~= yTrain);
colormap gray;
%imagesc(trainImages(:,:,err_index(size(err_index,2))))
%imagesc(trainImages(:,:,err_index(size(err_index,2))));
%err_index() %changed this because the error was not always 1x25

figure(5)
for k=1:min(size(err_index,2),25)  %changed this, because the error was not always 1x25
    subplot(5,5,k), imagesc(trainImages(:,:,err_index(k)));
    axis image;
    axis off;
end
clear err_index
%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.

unique_feature = unique(best_feature);
num_unique_feature = size(unique_feature,2);
dim = ceil(sqrt(num_unique_feature));

figure(6)
colormap gray;
for k = 1:num_unique_feature
    subplot(dim,dim,k),imagesc(haarFeatureMasks(:,:,unique_feature(k)));
    axis image;
    axis off;
end
clear unique_features