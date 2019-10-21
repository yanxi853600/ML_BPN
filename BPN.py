# (11/29)畫出BPN向前傳遞 與向後傳遞 方法與計算

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import math
import random

#載入數據集
avocado=pd.read_csv("avocado.csv")
x = pd.DataFrame(avocado,columns=["Total Volume","AveragePrice"])

#資料預處理
label_Encoder=preprocessing.LabelEncoder()
y=label_Encoder.fit_transform(avocado["type"])

#split our data
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.33,random_state=0)

#
class NeuralNetwork(object):

    def __init__(avocado, learning_rate=0.5, debug=False):
        """
        Train NeuralNetwork by fixed learning rate
        """
        avocado.neuron_layers = []
        avocado.learning_rate = learning_rate
        avocado.debug = debug

    def train(avocado, dataset):
        for inputs, outputs in dataset:
            avocado.feed_forward(inputs)
            avocado.feed_backword(outputs)
            avocado.update_weights(avocado.learning_rate)

    def feed_forward(avocado, inputs):
        s = inputs
        for (i, l) in enumerate(avocado.neuron_layers):
            s = l.feed_forward(s)
            if avocado.debug:
                print ("Layer %s:" % (i+1), " output:%s" % s)
        return s

    def feed_backword(avocado, outputs):
        layer_num = len(avocado.neuron_layers)
        l = layer_num
        previous_deltas = [] 
        while l != 0:
            current_layer = avocado.neuron_layers[l - 1]
            if len(previous_deltas) == 0:
                for i in range(len(current_layer.neurons)):
                    error = -(outputs[i] - current_layer.neurons[i].output)
                    current_layer.neurons[i].calculate_delta(error)
            else:
                previous_layer = avocado.neuron_layers[l]
                for i in range(len(current_layer.neurons)):
                    error = 0
                    for j in range(len(previous_deltas)):
                        error += previous_deltas[j] * previous_layer.neurons[j].weights[i]
                    current_layer.neurons[i].calculate_delta(error)
            previous_deltas = current_layer.get_deltas()
            if avocado.debug:
                print ("Layer %s:" % l, "deltas:%s" % previous_deltas)
            l -= 1

    def update_weights(avocado, learning_rate):
        for l in avocado.neuron_layers:
            l.update_weights(learning_rate)

    def calculate_total_error(avocado, dataset):
        """
        Return mean squared error of dataset
        """
        total_error = 0
        for inputs, outputs in dataset:
            actual_outputs = avocado.feed_forward(inputs)
            for i in range(len(outputs)):
                total_error += (outputs[i] - actual_outputs[i]) ** 2
        return total_error / len(dataset)

    def get_output(avocado, inputs):
       return avocado.feed_forward(inputs)

    def add_layer(avocado, neruon_layer):
        avocado.neuron_layers.append(neruon_layer)

    def dump(avocado):
        for (i, l) in enumerate(avocado.neuron_layers):
            print ("Dump layer: %s" % (i+1))
            l.dump()


class NeuronLayer(object):

    def __init__(avocado, input_num, neuron_num, init_weights=[], bias=1):
        avocado.neurons = []
        weight_index = 0
        for i in range(neuron_num):
            n = Neuron(input_num)
            for j in range(input_num):
                if weight_index < len(init_weights):
                    n.weights[j] = init_weights[weight_index]
                    weight_index += 1
            n.bias = bias
            avocado.neurons.append(n)

    def feed_forward(avocado, inputs):
        outputs = []
        for n in avocado.neurons:
            outputs.append(n.calculate_output(inputs))
        return outputs

    def get_deltas(avocado):
        return [n.delta for n in avocado.neurons]

    def update_weights(avocado, learning_rate):
        for n in avocado.neurons:
            n.update_weights(learning_rate)

    def dump(avocado):
        for (i, n) in enumerate(avocado.neurons):
            print ("-Dump neuron: %s" % (i+1))
            n.dump()


class Neuron(object):

    def __init__(avocado, weight_num):
        avocado.weights = []
        avocado.bias = 0
        avocado.output = 0
        avocado.delta = 0
        avocado.inputs = []
        for i in range(weight_num):
            avocado.weights.append(random.random())

    def calculate_output(avocado, inputs):
        avocado.inputs = inputs
        if len(inputs) != len(avocado.weights):
            raise Exception("Input number not fit weight number")
        avocado.output = 0
        for (i, w) in enumerate(avocado.weights):
            avocado.output += w * inputs[i]
        avocado.output = avocado.activation_function(avocado.output + avocado.bias)
        return avocado.output

    def activation_function(avocado, x):
        """Using sigmoid function"""
        return 1 / (1 + math.exp(-x))

    def calculate_delta(avocado, error):
        """ Using g' of sigmoid """
        avocado.delta = error * avocado.output * (1 - avocado.output)

    def update_weights(avocado, learning_rate):
        for (i, w) in enumerate(avocado.weights):
            new_w = w - learning_rate * avocado.delta * avocado.inputs[i]
            avocado.weights[i] = new_w
        avocado.bias = avocado.bias - learning_rate * avocado.delta

    def dump(avocado):
        print ("-- weights:", avocado.weights)
        print ("-- bias:", avocado.bias)
