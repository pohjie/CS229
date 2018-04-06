function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% First, we need to add ones to X
a1 = [ones(m, 1) X];

% Then, we need to compute a(2)
% a(2) is a 5000 by 25 matrix
a2 = sigmoid(a1 * Theta1');
a2 = [ones(m, 1) a2];

% Then we get a(3), which is a 5000 by 10 matrix
a3 = sigmoid(a2 * Theta2');

% Now we enter part 1, whereby we calculate cost function w/o regularization
K = columns(a3);
bigY = zeros(m, K);

for h = 1 : m
	bigY(h, y(h)) = 1;
end

for i = 1 : m
	for k = 1 : K
		J += -bigY(i,k) * log(a3(i, k)) - (1 - bigY(i, k)) * log(1 - a3(i, k));
	end
end

J /= m;

% regularization here
subset = 0;

for j = 1 : rows(Theta1)
	for k = 2 : columns(Theta1)
		subset += (Theta1(j, k))^2;
	end
end

for j = 1 : rows(Theta2)
	for k = 2 : columns(Theta2)
		subset += (Theta2(j, k))^2;
	end
end

subset = subset * lambda / (2 * m);
J += subset;
% -------------------------------------------------------------
% Backpropagation algorithm

% vectorized approach: step 1
diff3 = a3 - bigY;

% step 2
diff2 = diff3 * Theta2(:, 2:end) .* sigmoidGradient(a1 * Theta1');

% step 3
Delta1 = diff2' * a1;
Delta2 = diff3' * a2;

Theta1_grad = Delta1 ./ m;
Theta2_grad = Delta2 ./ m;

% regularization step
Theta1_reg = [zeros(rows(Theta1), 1) Theta1(:, 2:end)];
Theta1_reg = Theta1_reg .* (lambda / m);
Theta2_reg = [zeros(rows(Theta2), 1) Theta2(:, 2:end)];
Theta2_reg = Theta2_reg .* (lambda / m);

Theta1_grad += Theta1_reg;
Theta2_grad += Theta2_reg;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
