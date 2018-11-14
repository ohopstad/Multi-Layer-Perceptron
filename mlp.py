
import numpy as np
from random import random, seed
from math import exp

class mlp:
    def __init__(self, inputs, targets, nhidden):
        self.beta = 1
        self.eta = 0.1
        self.momentum = 0.0
        self.network = self.init_network(len(inputs[0]), len(targets[0]), nhidden) 

#       makes the two-layer network. Also keeps _init_ nice and tidy.
    def init_network(self, n_inputs, n_outputs, n_hidden):
        network = list()
        h_layer = list()
        for _ in range(n_hidden):
            h_layer.append({'weights':[random() for i in range (n_inputs +1)]})
        network.append(h_layer)
        out_layer = list()
        for _ in range(n_outputs):
            out_layer.append({'weights':[random() for i in range (n_hidden +1)]})
        network.append(out_layer)
        return network

#       Sigmoid transfer function and its derivative
    def transfer(self, activation):
        return 1.0/(1.0 + exp(-activation))  
    def transfer_deriv(self, output):
        return output * (1.0 - output)

#       calculates output of a neuron.
    def activate(self, weights, inputs):
        val = weights[-1] * self.beta #bias
        for i in range(len(weights) -1):
            val += weights[i] * inputs[i] #inputs
        return self.transfer(val)

    def forward(self, inputs):
        # activation
        # transfer
        # propagation 
        for layer in self.network: # one layer at a time. Use output this layer for input next layer
            new_inp = []
            for neuron in layer: 
                neuron['output'] = self.activate(neuron['weights'], inputs)
                new_inp.append(neuron['output'])
            inputs = new_inp
        return inputs

#       Calculates the backwards propagation errors.
    def back_prop_err(self, target): 
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network) -1:   # not output layer
                for j in range(len(layer)):
                    err = 0.0
                    for neuron in self.network[i +1]:
                        err += (neuron['weights'][j] * neuron['delta'])
                    errors.append(err)
            else:                           # output layer
                for j in range (len(layer)):
                    neuron = layer[j]
                    errors.append(target[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j]* self.transfer_deriv(neuron['output'])

#       updates the weights of the neurons in the network.
#       run self.back_prop_err() first.
    def update_weights(self, inputs):
        for i in range(len(self.network)):
            if i != 0:  # not input
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)): # W = eta * delta * inp
                    neuron['weights'][j] += self.eta * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += self.eta * neuron['delta'] # bias weight.

#       training algorithm. Updates weights for every output.
#       runs by default 100 times on every input.
    def train(self, inputs, targets, iterations=100):
        # update weights
        # train network
        sum_err = 0
        for epoch in range(iterations):
            sum_err = 0
            for i in range(len(inputs)):
                outp = self.forward(inputs[i])
                sum_err += sum([(targets[i][j] - outp[j])**2 for j in range(len(targets[i]))])
                self.back_prop_err(targets[i])
                self.update_weights(inputs[i])
            #print('>>epoch=%d, error=%.3f' % (epoch, sum_err))             # <- visual aid 
        return sum_err

#       defines one iteration as one call to self.train(). I chose a lower number here to speed up things.
#       runs training algorithm on training set, then checks the validation error.
    def earlystopping(self, inputs, targets, valid, validtargets):
        epochs = 20
        err = 100000
        err_prev = 100001 
        iteration = 0
        while (err +0.1) < err_prev :        # <- don't run for too long   (I just couldn't wait that long.)
            iteration += 1
            self.train(inputs, targets, epochs)
            err_prev = err
            err = 0
            for i in range(len(valid)):
                outp = self.forward(valid[i])
                err += sum([(validtargets[i][j] - outp[j])**2 for j in range(len(validtargets[i]))])
            print('>iteration=%d, valid_error=%.3f\n' % (iteration, err))     # <- visual aid (recommended)
            
#       Matrix depicting the accuracy of the preceptron.
    def confusion(self, inputs, targets):
        matrix = np.zeros(shape=(len(targets[0]), len(targets[0])))
        right = 0.0
        wrong = 0.0
        for i in range(len(inputs)):
            output = self.forward(inputs[i])
            col = output.index(max(output))
            row = np.nonzero(targets[i])[0][0]
            matrix[row,col] += 1
            #print('>>> col=%d, row%d' % (col, row))
            if col != row: 
                wrong += 1.0
            else: 
                right += 1.0
        print("V confusion matrix: V")
        print(matrix)
        if right != 0.0 and wrong != 0.0:
            percent = right/(right+wrong)
        elif right != 0.0:
            percent = 1.0
        else :
            percent = 0.0
        print('> right=%.3f, wrong=%.3f, accuracy=%.3f \n' % (right, wrong, percent))
        return matrix


# :::::::::::::::::::::
# ::: TESTING BELOW :::

if __name__ == "__main__": # Dont run unless this specific file is called.
    print("\ninit:")
    thing = mlp([[0, 0],[1, 1]], [[1]], 2)
    for layer in thing.network:
        print(layer)

    print("\nforward propagation:")
    inp = [1, 0]
    print(str(thing.forward(inp)) + "\n")
    for layer in thing.network:
        print(layer)


    print("\nbackward propagation:")
    outp = [1]
    thing.back_prop_err(outp)
    for layer in thing.network:
        print(layer)

    print("\ntrain:")  
    inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    targets = [[1, 0], [0, 1], [0, 1], [1, 0]]
    thing = mlp(inputs, targets, 3)
    thing.train(inputs, targets ,10000) 
    #for layer in thing.network:
     #   print(layer)

    print("\nconfusion:")
    thing.confusion(inputs, targets) # should be good, we have JUST trained for these values.
    