function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% need to find the a2 first 
% initialize a2 with all 1's as 1*1 = 1 or 1*0 =0 
% S0, a0 in 1st level will have 1 and also other will have no value.
% adding the +1 as first element
a1 = [ones(size(X, 1), 1) X];
%pridict a2 in level 2
Predictiona2 = sigmoid(a1 * Theta1');
% adding the +1 as first element
a2 = [ones(size(Predictiona2, 1), 1) Predictiona2];
Predictiona3 = sigmoid(a2 * Theta2');
[pridictmaxval, index] = max (Predictiona3, [], 2);
p = index;







% =========================================================================


end
