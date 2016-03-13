function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vec = sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
errors = zeros(length(C_vec));

for i = 1:length(C_vec)
  for j = 1:length(sigma_vec)
    model= svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j)));
    predictions = svmPredict(model, Xval);
    errors(i,j) = mean(double(predictions ~= yval));
  end
end

[_,I] = min(errors(:));
[I_row,I_col] = ind2sub(size(errors),I);

C = C_vec(I_row)          % 1
sigma = sigma_vec(I_col)  % 0.1

% =========================================================================
% plot
p = zeros(length(sigma_vec),1);
colors = ['k','r','g','b','m','c','k','r'];
styles = ['-','-','-','-','-','-','--','--'];
for j = 1:length(sigma_vec)
  p(j) = semilogx(C_vec, errors(j,:),'color',colors(j),'linestyle',styles(j));
  legend(p(j),sprintf("%0.2f",sigma_vec(j)));
  hold on;
end
xlabel('C');
ylabel('Error');
title('sigma')
hold off;

end
