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

%classes = unique(LTrain);
%NClasses = length(classes);

% Add your own code here
LPred  = zeros(size(X,1),1);

for i=1:size(X, 1)
    D = pdist2(X(i,:),XTrain);
    [D1, I] = sort(D);
    N = LTrain(I(:,1:k));
    LPred(i) = mode(N);
end

  
end

