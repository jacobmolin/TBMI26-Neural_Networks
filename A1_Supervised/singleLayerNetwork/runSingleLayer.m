function [ Y, L ] = runSingleLayer(X, W)
% RUNSINGLELAYER Performs one forward pass of the single layer network, i.e
% it takes the input data and calculates the output for each sample.
%
%    Inputs:
%              X - Samples to be classified (matrix)
%              W - Weights of the neurons (matrix)
%
%    Output:
%              Y - Output for each sample and class (matrix)
%              L - The resulting label of each sample (vector) 

% Add your own code here
X1 = X;
X1(end,:) = [];
% X1
% size(X1)
% size(W)

Y = X1.*W;
% size(Y)

% trans([1000 x 2] * [1000 x 2]

% Calculate labels
[~, L] = max(Y, [], 2);

end

