clear; clc; close all; format compact

num_inputs = 784;
num_hidden = 100;
num_outputs = 10;

learning_rate = 0.3;

nn = NeuralNetwork(num_inputs, num_hidden, num_outputs, learning_rate);

disp(nn);

mnist_train = csvread('data/mnist_train_100.csv');
mnist_test = csvread('data/mnist_test_10.csv');

for i = 1:size(mnist_train, 1)
    inputs = mnist_train(i, 2:end) / 255.0 * 0.99 + 0.01;
    label = mnist_train(i, 1);

    targets = zeros(1, num_outputs) + 0.01;
    targets(label + 1) = 0.99;

    train(nn, inputs, targets)
end

for i = 1:size(mnist_test, 1)
    inputs = mnist_test(i, 2:end) / 255.0 * 0.99 + 0.01;
    label = mnist_test(i, 1);

    targets = zeros(1, num_outputs) + 0.01;
    targets(label + 1) = 0.99;
    
    predicted_targets = test(nn, inputs);
    
    [~, predicted_targets] = max(predicted_targets);
    [~, targets] = max(targets);
    
    fprintf('predicted(%d):', i);
    disp(predicted_targets);
    
    fprintf('actual(%d):', i);
    disp(targets);
end