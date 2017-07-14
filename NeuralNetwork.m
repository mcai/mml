% Neural network.
classdef NeuralNetwork
    properties
        numInputs
        numHidden
        numOutputs
        learningRate
    end
    
    properties % (Access = private)
        weights_inputs_hidden
        weights_hidden_outputs
    end
    
    properties
        activation_function
    end
    
    methods
        % Initialize the neural network.
        function nn = NeuralNetwork(numInputs, numHidden, numOutputs, learningRate)
            nn.numInputs = numInputs;
            nn.numHidden = numHidden;
            nn.numOutputs = numOutputs;
            nn.learningRate = learningRate;
            
            nn.weights_inputs_hidden = random('Normal', 0, power(nn.numHidden, -0.5), [nn.numHidden, nn.numInputs]);
            nn.weights_hidden_outputs = random('Normal', 0, power(nn.numOutputs, -0.5), [nn.numOutputs, nn.numHidden]);
            
            nn.activation_function = @logsig;
        end
        
        % Train the neural network.
        function train(nn, inputs, targets)
            inputs = inputs';
            targets = targets';
            
            hidden_in = nn.weights_inputs_hidden * inputs;
            hidden_out = nn.activation_function(hidden_in);
            
            outputs_in = nn.weights_hidden_outputs * hidden_out;
            outputs_out = nn.activation_function(outputs_in);
            
            outputs_errors = targets - outputs_out;
            hidden_errors = nn.weights_hidden_outputs' * outputs_errors;
            
            nn.weights_hidden_outputs = nn.weights_hidden_outputs + nn.learningRate .* outputs_errors .* (1 - outputs_errors) * hidden_out';
            nn.weights_inputs_hidden = nn.weights_inputs_hidden + nn.learningRate .* hidden_errors .* (1 - hidden_errors) * inputs';
        end
        
        % Test the neural network.
        function r = test(nn, inputs)
            inputs = inputs';
            
            hidden_in = nn.weights_inputs_hidden * inputs;
            hidden_out = nn.activation_function(hidden_in);
            
            outputs_in = nn.weights_hidden_outputs * hidden_out;
            outputs_out = nn.activation_function(outputs_in);
            
            r = outputs_out';
        end
    end
end

