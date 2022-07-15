function [ cM ] = calcConfusionMatrix( LPred, LTrue )
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels

classes  = unique(LTrue);
NClasses = length(classes);

% %the easy way:(-----------------------
% cM=crosstab(LPred, LTrue );
% cM = confusionmat(LTrue, LPred);
% %--------------------------------------

% Add your own code here
cM = zeros(NClasses);

for n=1:NClasses
    for i=1:length(LPred)
        %
        if n == LTrue(i) == LPred(i)
            cM(n,n) = cM(n,n) + 1;
        elseif n == LTrue(i)
            cM(LPred(i),n) = cM(LPred(i),n) +1;
        elseif n == LPred(i)
            cM(n, LTrue(i)) = cM(n, LTrue(i)) + 1;
        end
    end
end
end


