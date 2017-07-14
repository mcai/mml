clear; clc; close all; format compact

num_inputs = 784;
num_hidden = 100;
num_outputs = 10;

learning_rate = 0.3;

nn = NeuralNetwork(num_inputs, num_hidden, num_outputs, learning_rate);

disp(nn);

% mnist_train = csvread('data/mnist_train_100.csv');
% mnist_test = csvread('data/mnist_test_10.csv');

disp('Loading data..')

mnist_train = csvread('data/mnist_train.csv');
mnist_test = csvread('data/mnist_test.csv');

disp('Training..')

for i = 1:size(mnist_train, 1)
    inputs = mnist_train(i, 2:end) / 255.0 * 0.99 + 0.01;

    targets = zeros(1, num_outputs) + 0.01;
    targets(1, mnist_train(i, 1) + 1) = 0.99;

    train(nn, inputs, targets);
end

disp('Testing..')

scores = zeros(size(mnist_test, 1), 1);

for i = 1:size(mnist_test, 1)
    inputs = mnist_test(i, 2:end) / 255.0 * 0.99 + 0.01;

    targets = zeros(1, num_outputs) + 0.01;
    targets(1, mnist_test(i, 1) + 1) = 0.99;
    
    predicted_targets = test(nn, inputs);
    
    [~, predicted] = max(predicted_targets);
    [~, actual] = max(targets);
     
    scores(i, 1) = (predicted == actual);
end

acc = sum(scores == 1) / size(scores, 1);

fprintf("Accuracy: %.3f\n", acc);