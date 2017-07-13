clear; clc; format compact

numInputs = 3;
numHidden = 3;
numOutputs = 3;

learningRate = 0.3;

nn = NeuralNetwork(numInputs, numHidden, numOutputs, learningRate);

disp(nn);

nn.weights_inputs_hidden            
nn.weights_hidden_outputs

train(nn)

test(nn, [1.0, 0.5, -1.5])

