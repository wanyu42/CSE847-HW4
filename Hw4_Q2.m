clear;
load('ad_data.mat');
par  = [1e-8, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];


no_zeros_list = [];
AUC_list = [];
for par_value = par
  [w,c] = logistic_l1_train(X_train, y_train, par_value);
  % none zeros weights
  no_zeros = sum(w~=0);
  no_zeros_list = [no_zeros_list no_zeros];
  
  % Get the probability of one
  scores = 1./(1+exp(-(X_test*w+c)));
  [X,Y,T,AUC] = perfcurve(y_test,scores,1);
  AUC_list = [AUC_list AUC];
end

figure(1)
plot(par, no_zeros_list)
xticks(par)
title('none zero features')
figure(2)
plot(par, AUC_list)
xticks(par)
title('AUC')


function [w, c] = logistic_l1_train(data, labels, par)
% OUTPUT w is equivalent to the first d dimension of weights in logistic train
% c is the bias term, equivalent to the last dimension in weights in logistic train.
% Specify the options (use without modification).
opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000; % maximum iterations

[w, c] = LogisticR(data, labels, par, opts);
end

