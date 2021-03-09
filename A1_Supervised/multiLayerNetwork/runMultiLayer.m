function [ Y, L, H ] = runMultiLayer( X, W, V )
% RUNMULTILAYER Calculates output and labels of the net
%
%    Inputs:
%              X - Data samples to be classified (matrix)
%              W - Weights of the hidden neurons (matrix)
%              V - Weights of the output neurons (matrix)
%
%    Output:
%              Y - Output for each sample and class (matrix)
%              L - The resulting label of each sample (vector) 
%              U - Activation of hidden neurons (vector)

% Add your own code here
S = X * W; % Calculate the weighted sum of input signals (hidden neuron)
biasCol = ones(size(S, 1), 1);
H = [tanh(S) biasCol]; % Calculate the activation of the hidden neurons (use hyperbolic tangent)
Y = H * V; % Calculate the weighted sum of the hidden neurons


% Calculate labels
[~, L] = max(Y,[],2);

end

