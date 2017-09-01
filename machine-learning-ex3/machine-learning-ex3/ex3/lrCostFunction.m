function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

hypothesisx = sigmoid(X * theta);    %Prediction for the given x on sigmoid
%cost funtion on y= 0 or 1  should work.
prod1 = -1 * (y .* log(hypothesisx));  
prod2 = (1 - y) .* log(1 - hypothesisx);
% and remove the first theta value for regularization
theta_r = theta(2:end, :);
  
J = (1/(m) * sum( prod1 - prod2) )  + ((lambda / (2 * m))  * sum (theta_r .^2));  

%theatalength = length(theta);

% updating Gradients
printf('updating gradient\n');
%fullgradsbefore = (X' * (hypothesisx - y)) * (1/m);
gradwithoutlamda = (X' * (hypothesisx - y)) * (1/m);
gradwithlamda = ((X' * (hypothesisx - y)) * (1/m) ) + ((lambda/ m)  * theta);
grad(1, : ) = gradwithoutlamda(1, : ) ;
grad(2:end, : ) = gradwithlamda(2:end, : );
%for graditer = 2 : theatalength
%  grad(graditer) = (fullgradsbefore(graditer))  + ((lambda /m) * theta(graditer));  
%  end

% grad = ((1/m) * (X' * (hypothesisx - y))  )+ ((lambda/2 * m)  * sum(theta_r));

% =============================================================

grad = grad(:);

end
