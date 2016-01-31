function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

% Cost function
for i = 1:m
    h = sigmoid(X(i,:) * theta);
    J += -y(i)*log(h) - (1 - y(i))*log(1-h);
end

J = J / m + norm(theta(2:size(theta)))^2 * lambda / (2*m);


% Gradient (loop version)
% for j = 1:size(theta)
%     for i = 1:m
%         h = sigmoid(X(i,:) * theta);
%         grad(j) += (h - y(i)) * X(i,j);
%     end
%     if j > 1
%       grad(j) += lambda * theta(j);
%     end
% end

% grad = grad / m;

% Gradient (vectorize)
n = size(theta);

s = (sigmoid(X * theta) - y) / m;
grad(1) = X(:,1)' * s;
grad(2:n) = X(:,2:n)' * s + lambda * theta(2:n) / m;

% =============================================================

end
