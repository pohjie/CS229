function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = rows(theta); % number of features
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
for i = 1:m
	J += (-1 * y(i) * log(sigmoid(theta' * X(i, :)')) - (1 - y(i)) * log(1 - sigmoid(theta' * X(i, :)')));

end

J /= m;

dummy = 0; % to separate the computation into two distinct parts
for j = 2:n
	dummy += (theta(j))^2;
end

dummy *= lambda;
dummy /= (2 * m);

J += dummy;

for i = 1 : m
	grad += ((sigmoid(X(i, :) * theta)) - y(i)) * X(i, :)';
end

grad = grad ./ m;

for j = 2 : n
	grad(j) += (theta(j) * lambda / m);
end
% =============================================================

end