function [ cM ] = calcConfusionMatrix( LPred, LTrue )
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels

classes  = unique(LTrue);
NClasses = length(classes);

%oc0 = sum(LPred(:) == 0)
%oc1 = sum(LPred(:) == 1)

% Add your own code here
cM = zeros(NClasses);

for i=1:length(LPred)
    CPred = LPred(i);
    CTrue = LTrue(i);
    cM(CTrue,CPred) = cM(CTrue,CPred) + 1; 
end
end

