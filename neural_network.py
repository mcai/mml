import numpy
import scipy.special


class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs, learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.weights_inputs_hidden = numpy.random.normal(0.0, pow(self.num_inputs, -0.5), (self.num_hidden, self.num_inputs))
        self.weights_hidden_outputs = numpy.random.normal(0.0, pow(self.num_hidden, -0.5), (self.num_outputs, self.num_hidden))

        self.learning_rate = learning_rate

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_in = numpy.dot(self.weights_inputs_hidden, inputs)
        hidden_out = self.activation_function(hidden_in)

        outputs_in = numpy.dot(self.weights_hidden_outputs, hidden_out)
        outputs_out = self.activation_function(outputs_in)

        output_errors = targets - outputs_out
        hidden_errors = numpy.dot(self.weights_hidden_outputs.T, output_errors)

        self.weights_hidden_outputs += self.learning_rate * numpy.dot((output_errors * outputs_out * (1.0 - outputs_out)),
                                                                      numpy.transpose(hidden_out))
        self.weights_inputs_hidden += self.learning_rate * numpy.dot((hidden_errors * hidden_out * (1.0 - hidden_out)),
                                                                     numpy.transpose(inputs))

    def test(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_in = numpy.dot(self.weights_inputs_hidden, inputs)
        hidden_out = self.activation_function(hidden_in)

        outputs_in = numpy.dot(self.weights_hidden_outputs, hidden_out)
        outputs_out = self.activation_function(outputs_in)

        return outputs_out


num_inputs = 784
num_hidden = 100
num_outputs = 10

learning_rate = 0.3

n = NeuralNetwork(num_inputs, num_hidden, num_outputs, learning_rate)

training_data_file = open("data/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(num_outputs) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

test_data_file = open("data/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scores = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.test(inputs)
    label = numpy.argmax(outputs)
    if label == correct_label:
        scores.append(1)
    else:
        scores.append(0)

scores_array = numpy.asarray(scores)
print("performance = ", scores_array.mean())
