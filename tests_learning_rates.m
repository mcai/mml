clear; clc; close all; format compact

disp('Loading data..')

mnist_train_data = csvread('data/mnist_train.csv');
mnist_test_data = csvread('data/mnist_test.csv');

%%
num_inputs = 784;
num_hidden = 100;
num_outputs = 10;

learning_rates_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.2, 2.4]';

num_epochs = 10;

accuracies = zeros(size(learning_rates_range, 1), 1);

for i = 1:size(learning_rates_range, 1)
    learning_rate = learning_rates_range(i);
    accuracy = train_and_test(mnist_train_data, mnist_test_data, num_inputs, num_hidden, num_outputs, learning_rate, num_epochs);

    accuracies(i) = accuracy;
end

%%
[best_accuracy, best_accuracy_index] = max(accuracies);
best_learning_rate = learning_rates_range(best_accuracy_index);

fprintf("Best Learning rate: %.4f, best accuracy: %.4f\n", best_learning_rate, best_accuracy);

%%
plot(learning_rates_range, accuracies, '-o')
title('Learning Rate vs. Accuracy')
xlabel('Learning Rate')
ylabel('Accuracy')

%%
results = array2table([learning_rates_range, accuracies], 'VariableNames', {'learning_rate', 'accuracy'});

writetable(results, 'results/learning_rate-accuracy.csv', 'Delimiter', ',', 'QuoteStrings', true)

