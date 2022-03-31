clear;
load('data.txt');
load('labels.txt');

% Add ones to last column for intercept
data = [data, ones(size(data, 1), 1)];
% Use +1/-1 as labels
labels = 2*labels-1;

train_x = data(1:2000, :);
train_y = labels(1:2000);
test_x = data(2001:end, :);
test_y = labels(2001:end);

n_list = [200, 500, 800, 1000, 1500, 2000];
acc_list = [];
for n=n_list
  w = logistic_train(train_x(1:n, 1:end), train_y(1:n));
  
  %Test on testset
  pred = 2*(test_x * w >= 0)-1;
  acc = mean(pred == test_y);
  acc
  acc_list = [acc_list, acc];
end

figure(1)
plot(n_list, acc_list)
xticks(n_list)
grid()
title('Acc vs TrainSize')