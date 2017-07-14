clear; clc; close all; format compact

disp('Loading data..')

mnist_train_data = csvread('data/mnist_train.csv');
mnist_test_data = csvread('data/mnist_test.csv');

%%
num_inputs = 784;
num_hidden_range = [5, 10, 20, 50, 100, 200, 500]';
num_outputs = 10;

learning_rate = 0.1;

num_epochs = 10;

accuracies = zeros(size(num_hidden_range, 1), 1);

for i = 1:size(num_hidden_range, 1)
    num_hidden = num_hidden_range(i);
    accuracy = train_and_test(mnist_train_data, mnist_test_data, num_inputs, num_hidden, num_outputs, learning_rate, num_epochs);

    accuracies(i) = accuracy;
end

%%
[best_accuracy, best_accuracy_index] = max(accuracies);
best_num_hidden = num_hidden_range(best_accuracy_index);

fprintf("Best number of hidden nodes: %d, best accuracy: %.4f\n", best_num_hidden, best_accuracy);

%%
plot(num_hidden_range, accuracies, '-o')
title('Number of Hidden Nodes vs. Accuracy')
xlabel('Number of Hidden Nodes')
ylabel('Accuracy')

%%
results = array2table([num_hidden_range, accuracies], 'VariableNames', {'num_hidden', 'accuracy'});

writetable(results, 'results/num_hidden-accuracy.csv', 'Delimiter', ',', 'QuoteStrings', true)

