clear; clc; close all; format compact

disp('Loading data..')

mnist_train_data = csvread('data/mnist_train.csv');

img = imread('data/3.png');

img = rgb2gray(img);

img = imresize(img, [28, 28]);

img = 255 - img;

img = reshape(img, 1, 784);

img = double(img);

custom_test_data = [3, img];

%%
num_inputs = 784;
num_hidden = 200;
num_outputs = 10;

learning_rate = 0.1;

num_epochs = 10;

train_and_test(mnist_train_data, custom_test_data, num_inputs, num_hidden, num_outputs, learning_rate, num_epochs);

