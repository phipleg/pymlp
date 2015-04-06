# -*- coding: utf-8 -*-

import numpy as np
import random
import time

class MLP():
    """ Multilayer perceptron """

    def __init__(self, sizes):
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    def feedforward(self, a):
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,a) + b
            a = sigmoid_vec(z)
        return a

    def sgd(self, tr_d, epochs, mb_len, te_d, eta, lmbda=0.0):
        print "SGD. epochs={}, eta={}, l2reg={}, net={}".format(epochs,eta,lmbda, self.sizes)
        for epidx in xrange(epochs):
            self.sgd_epoch( tr_d, epochs, epidx, mb_len, te_d, eta, lmbda)

    def sgd_epoch(self, tr_d, epochs, epidx, mb_len, te_d, eta, lmbda):
        n = len(tr_d)
        mini_batches = self.select_mini_batches(tr_d, mb_len)
        for idx, mini_batch in enumerate(mini_batches):
            self.update_mini_batch(mini_batch, eta, lmbda, n)
        if te_d:
            acc = self.evaluate(te_d)
            print "Epoch {}/{}, test accuracy={}".format(
                epidx, epochs, acc)

    def evaluate(self, te_d):
        matches = sum([int(np.argmax(self.feedforward(x)) == y) for (x, y) in te_d])
        return matches * 1.0 / len(te_d)

    def select_mini_batches(self, tr_d, mb_len):
        random.shuffle(tr_d)
        return [tr_d[k:k+mb_len] for k in xrange(0,len(tr_d),mb_len)]

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_w, nabla_b = self.gradient(mini_batch)
        eps = eta/len(mini_batch)
        self.weights = [ (1-eta*(lmbda/n))*w - eps*nw for w, nw in zip(self.weights, nabla_w) ]
        self.biases = [ b - eps*nb for b, nb in zip(self.biases, nabla_b)]
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

