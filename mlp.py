# -*- coding: utf-8 -*-

import numpy as np
import random
import time
from sys import stdout

class MLP():
    """ Multilayer perceptron """

    def __init__(self, sizes):
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        self.biases_v = [np.zeros((y,1)) for y in sizes[1:]]
        self.weights_v = [np.zeros((y,x)) for x,y in zip(sizes[:-1], sizes[1:])]
        print "MLP net={}".format(self.sizes)

    def feedforward(self, a):
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,a) + b
            a = sigmoid_vec(z)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, test_data, learning_rate, lmbda=0.0):
        self.durations = np.array([])
        self.t0 = time.time()
        print "SGD. epochs={}, learning_rate={}, l2reg={}, mini_batch_size={}".format(epochs, learning_rate,lmbda, mini_batch_size)
        if test_data:
            print "Init test error={}".format(self.evaluate(test_data))
        for epidx in xrange(epochs):
            self.sgd_epoch( training_data, epochs, epidx, mini_batch_size, learning_rate, lmbda)
            if test_data:
                print "Test error={}".format(self.evaluate(test_data))

    def sgd_epoch(self, training_data, epochs, epidx, mini_batch_size, learning_rate, lmbda):
        n = len(training_data)
        mini_batches = self.select_mini_batches(training_data, mini_batch_size)
        counter = 0
        t1 = time.time()
        duration = 0
        for idx, mini_batch in enumerate(mini_batches):
            self.update_mini_batch(mini_batch, learning_rate, lmbda, n)
            counter += len(mini_batch)
            t = time.time()
            ep_duration = t - t1
            progress = counter * 1.0 / n
            ep_estimate = ep_duration / progress
            total_progress = ((epidx)*n + counter) * 1.0/(epochs*n)
            total_duration = t - self.t0
            total_estimate = total_duration / total_progress
            printi("T={}s/{}s. t={}/s{}s. Epoch {}/{}. Train {}/{}. ".format(int(total_duration), int(total_estimate), int(ep_duration), int(ep_estimate), epidx+1, epochs, counter,n))

    def evaluate(self, test_data):
        matches = sum([int(np.argmax(self.feedforward(x)) == y) for (x, y) in test_data])
        return 1.0 - matches * 1.0 / len(test_data)

    def select_mini_batches(self, training_data, mini_batch_size):
        random.shuffle(training_data)
        return [training_data[k:k+mini_batch_size] for k in xrange(0,len(training_data),mini_batch_size)]

    def update_mini_batch(self, mini_batch, learning_rate, lmbda, n):
        nabla_w, nabla_b = self.gradient(mini_batch)
        m1 = 0.5
        m2 = 1 - m1
        eps = learning_rate/len(mini_batch)
        self.weights = [ (1-learning_rate*(lmbda/n))*w + w_v for w, w_v in zip(self.weights, self.weights_v) ]
        self.biases = [ b + b_v for b, b_v in zip(self.biases, self.biases_v)]
        self.weights_v = [m1 * w_v - m2*eps*nw for w_v, nw in zip(self.weights_v, nabla_w)]
        self.biases_v = [m1 * b_v - m2*eps*nb for b_v, nb in zip(self.biases_v, nabla_b)]
        return

    def gradient(self, mini_batch):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for (x,y) in mini_batch:
            delta_nabla_w, delta_nabla_b, delta_cost = self.backprop(x,y)
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        return nabla_w, nabla_b

    def backprop(self, x, y):
        activations, zs = self.forward_pass(x)
        cost_prime = (activations[-1] - y)# * sigmoid_prime_vec(zs[-1])
        cost = np.linalg.norm(cost_prime)
        nabla_w, nabla_b = self.backward_pass(activations, zs, cost_prime)
        return nabla_w, nabla_b, cost

    def forward_pass(self, x):
        a = x
        activations = [a]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w,a) + b
            a = sigmoid_vec(z)
            zs.append(z)
            activations.append(a)
        return activations, zs

    def backward_pass(self, activations, zs, nabla_cost):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta = nabla_cost
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta
        for k in xrange(2, len(self.sizes)):
            l = -k
            w_t = self.weights[l+1].transpose()
            nabla_c = np.dot(w_t, delta)
            delta = nabla_c * sigmoid_prime_vec(zs[l])
            a_t = activations[l-1].transpose()
            nabla_w[l] = np.dot(delta, a_t)
            nabla_b[l] = delta
        return nabla_w, nabla_b

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)

def printi(str):
    stdout.write("\r" + str)
    stdout.flush()
