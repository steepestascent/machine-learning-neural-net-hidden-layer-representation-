import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Learner():

    def __init__(self, sizes):
        self.number_of_layers = len(sizes)
        self.sizes = sizes
        np.random.seed(60)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        activations = [a]
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w,a) + b)
            activations.append(a)
        return activations

    def SGD(self, traning_data, epochs, mini_batch_size, eta, test_data):
        # plot data
        self.sum_of_squared_errors = pd.DataFrame(np.zeros(shape=(epochs,8)))

        score = 0 
        for self.epcoh_iteration in range(epochs):
            if score == 8:
                break
            n = len(traning_data)
            # random.shuffle(traning_data)
            mini_batches = [traning_data[k:k+mini_batch_size]
            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
           
            print "epoch {0} complete".format(self.epcoh_iteration)
            score = self.evaluate(test_data)
            print "score: {0}".format(score)
        
        self.print_hidden_layer_representation(traning_data)
        self.plot_sum_of_squared_errors()
        print 'finished'
    
    def update_mini_batch(self, mini_batch, eta):
        grad_b, grad_w = self.get_zeroed_biases_and_weights()

        for x, y in mini_batch:
            x = x.reshape(8,1)
            y = y.reshape(8,1)
            delta_grad_b, delta_grad_w = self.backprop(x, y)
            grad_b = [b + nb for b, nb in zip(grad_b, delta_grad_b)]
            grad_w = [w + nw for w, nw in zip(grad_w, delta_grad_w)]
        self.biases = [b-(eta/len(mini_batch))* nb for b, nb in zip(self.biases, grad_b)]   
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, grad_w)]   

    def backprop(self, x, y):
        grad_b, grad_w = self.get_zeroed_biases_and_weights()

        # feed forward
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # record sum of squared error
        self.sum_of_squared_errors.iloc[self.epcoh_iteration] += np.reshape(np.sqrt((activations[-1] - y)**2), (8,))
        
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.number_of_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (grad_b, grad_w)

    def get_zeroed_biases_and_weights(self):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        return grad_b, grad_w

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
       
    def cost_derivative(self, output_activations, y):
        # (o - t)
        return (output_activations - y)

    def sigmoid_prime(self, z):
        return self.sigmoid(z)* (1 - self.sigmoid(z))

    def print_hidden_layer_representation(self, traning_data):
        hidden_layer_activations = np.zeros(shape=(8,3))
        for index, (x, y) in enumerate(traning_data):
            activations = self.feed_forward(x.reshape(8,1))
            hidden_layer_activations[index] =  np.reshape(activations[-2], (3,))

        hidden_layer_activations_medians = pd.DataFrame(hidden_layer_activations).median(axis=0).values     
        for x, y in traning_data:
            activations = self.feed_forward(x.reshape(8,1))
            print "{0} \t {1}, \t {2}, \t {3}: \t Target: {4}".format(self.get_binary(activations[-2], hidden_layer_activations_medians), activations[-2][0], activations[-2][1], activations[-2][2], np.argmax(y))

    def get_binary(self, vector, column_medians):
        binary = ""
        for i, x in enumerate(vector):
            if x >= column_medians[i]:
                binary += "1"
            else:
                binary += "0"
        return binary

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)[-1]),np.argmax(y)) for (x, y) in test_data]
        return sum(int(x==y) for x, y in test_results)

    
    def plot_sum_of_squared_errors(self):        
        df = self.sum_of_squared_errors.iloc[:self.epcoh_iteration,:]
        # min max norm
        df = (df-df.min(axis=0))/(df.max(axis=0)-df.min(axis=0))

        plt.title('Sum of squared errors for each output unit')

        for i in range(len(df.columns)):
            plt.plot(df.index, df.iloc[:, i])

        plt.xlabel('epoch')
        plt.ylabel('error')

        plt.show()

if __name__ == "__main__":
    l = Learner([8,3,8])
    training_data =  [x.reshape(8,1) for x in np.zeros((8, 8))]
    for i in xrange(8):
        training_data[i][i] = 1
    training_data = zip(training_data, training_data)       
    l.SGD(training_data,3000,8,0.3, test_data=training_data)
