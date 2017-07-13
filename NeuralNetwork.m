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
        function obj = NeuralNetwork(numInputs, numHidden, numOutputs, learningRate)
            obj.numInputs = numInputs;
            obj.numHidden = numHidden;
            obj.numOutputs = numOutputs;
            obj.learningRate = learningRate;
            
            obj.weights_inputs_hidden = random('Normal', 0, power(obj.numHidden, -0.5), [obj.numHidden, obj.numInputs]);
            obj.weights_hidden_outputs = random('Normal', 0, power(obj.numOutputs, -0.5), [obj.numOutputs, obj.numHidden]);
            
            obj.activation_function = @logsig;
        end
        
        % Train the neural network.
        function train(obj)
        end
        
        % Query the neural network.
        function outputs_out = test(obj, inputs)
            hidden_in = obj.weights_inputs_hidden * inputs;
            hidden_out = obj.activation_function(hidden_in);
            
            outputs_in = obj.weights_hidden_outputs * hidden_out;
            outputs_out = obj.activation_function(outputs_in);
        end
    end
end

