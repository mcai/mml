clear; clc; close all; format compact

disp('Loading data..')

mnist_train = csvread('data/mnist_train.csv');
mnist_test = csvread('data/mnist_test.csv');

%%
num_inputs = 784;
num_hidden = 100;
num_outputs = 10;

learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.2, 2.4]';

accuracies = zeros(size(learning_rates, 1), 1);

for learning_rate_i = 1:size(learning_rates, 1)
    learning_rate = learning_rates(learning_rate_i);
    nn = NeuralNetwork(num_inputs, num_hidden, num_outputs, learning_rate);

    disp(nn);

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

        outputs = test(nn, inputs);

        [~, predicted] = max(outputs);
        [~, actual] = max(targets);

        scores(i, 1) = (predicted == actual);
    end

    accuracy = mean(scores);

    fprintf("Learning rate: %.4f, accuracy: %.4f\n", learning_rate, accuracy);
    
    accuracies(learning_rate_i) = accuracy;
end

%%
[best_accuracy, best_accuracy_index] = max(accuracies);
best_learning_rate = learning_rates(best_accuracy_index);

fprintf("Best Learning rate: %.4f, best accuracy: %.4f\n", best_learning_rate, best_accuracy);

%%
plot(learning_rates, accuracies, '-o')
title('Learning Rate vs. Accuracy')
xlabel('Learning Rate')
ylabel('Accuracy')

