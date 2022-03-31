function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS: 
% data = n * (d+1) matrix with n samples and d features, where
% column d+1 is all ones (corresponding to the intercept term) 
% labels = n * 1 vector of class labels (taking values 0 or 1)
% epsilon = optional argument specifying the convergence
% criterion - if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
% (if unspecified, use a default value of 1e-5)
% maxiter = optional argument that specifies the maximum number of
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
%
% OUTPUT: 
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"

if nargin == 2
  epsilon = 1e-5;
  maxiter = 1000;
end
N = size(data, 1);

weights = zeros(size(data, 2), 1);

for i = 1:maxiter
  % Calculte the soft weights, shape: n*1
  soft = 1./(1 + exp(labels.* (data * weights)) );
  % Calculate y*x
  y_times_x = diag(labels) * data;
  % Calculate the gradient 
  grad = - (y_times_x' * soft)/N;
  weights = weights - grad;
  
  if mean(abs(grad)) < epsilon
    break
  end
  
end

end