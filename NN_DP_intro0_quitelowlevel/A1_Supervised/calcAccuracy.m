function [acc] = calcAccuracy( cM )
% CALCACCURACY Takes a confusion matrix amd calculates the accuracy

acc = trace(cM)/ sum(sum(cM));

end

