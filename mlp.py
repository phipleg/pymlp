# -*- coding: utf-8 -*-

import numpy as np
import random
import time

import printutils as pu

import matplotlib.pyplot as plt
import plt_pixels
import plt_confusion

class MLP():
    """ Multilayer perceptron """

    def __init__(self, sizes):
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        self.biases_v = [np.zeros((y,1)) for y in sizes[1:]]
        self.weights_v = [np.zeros((y,x)) for x,y in zip(sizes[:-1], sizes[1:])]
        print "MLP net={}".format(self.sizes)
        plt.ion()
        plt.show()
        self.fig = plt.figure(num=None, figsize=(14, 6), dpi=80)

    def feedforward(self, a):
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,a) + b
            a = sigmoid_vec(z)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, test_data, learning_rate, lmbda=0.0, drop_prob=0.0):
        self.t0 = time.time()
        print "SGD. epochs={}, mini_batch_size={}, learning_rate={}, l2reg={}, dropout_probability={}".format(epochs, mini_batch_size, learning_rate,lmbda, drop_prob)
        if test_data:
            print "Initial acc={}".format(self.evaluate(test_data)[0])
        for epidx in xrange(epochs):
            self.sgd_epoch( training_data, epochs, epidx, mini_batch_size, learning_rate, lmbda, drop_prob)
            acc, conf_matrix =  self.evaluate(test_data)
            print " Test acc={}".format(acc)
            plt.clf()
            ax_count = len(self.sizes)
            ax_conf = self.fig.add_subplot(1,ax_count,1)
            ax_conf.set_title("Confusion matrix")
            plt_confusion.draw_confusion_matrix(self.fig, ax_conf, conf_matrix)
            for i, (x,y) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
                xx = int(np.ceil(np.sqrt(x)))
                yy = int(np.ceil(np.sqrt(y)))
                ax = self.fig.add_subplot(1,ax_count,i+2)
                ax.set_title("{}. layer weights".format(i+1))
                plt_pixels.draw_pixels(self.fig, ax, self.weights[i], [xx,xx], [yy,yy])
            plt.draw()

    def sgd_epoch(self, training_data, epochs, epidx, mini_batch_size, learning_rate, lmbda, drop_prob):
        t1 = time.time()

        self.weights_dropout = [ np.ones((y,x)) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
        for w_drop in self.weights_dropout:
            y = w_drop.shape[0]
            w_drop[np.random.random_integers(0,y-1,drop_prob*y),:]=0
        self.biases_dropout = [ np.ones((y,1)) for y in self.sizes[1:]]
        for b_drop in self.biases_dropout:
            y = b_drop.shape[0]
            b_drop[np.random.random_integers(0,y-1,drop_prob*y)]=0

        mini_batches = self.select_mini_batches(training_data, mini_batch_size)
        n = len(training_data)
        counter = 0
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
            pu.printi("[{:3.1f}%] {} T={}/{}. t={}/{}. Epoch={}/{}. Train={}/{}.".format(total_progress*100, pu.bar(progress), pu.to_ht(total_duration), pu.to_ht(total_estimate), pu.to_ht(ep_duration), pu.to_ht(ep_estimate), epidx+1, epochs, counter, n))

    def evaluate(self, test_data):
        matches = 0
        d = self.sizes[-1]
        conf_matrix = np.zeros((d,d))
        for (x,y) in test_data:
            z = int(np.argmax(self.feedforward(x)))
            if y == z:
                matches += 1
            conf_matrix[y][z] += 1
        # matches = sum([int(np.argmax(self.feedforward(x)) == y) for (x, y) in test_data])
        return matches * 1.0 / len(test_data), conf_matrix

    def select_mini_batches(self, training_data, mini_batch_size):
        random.shuffle(training_data)
        return [training_data[k:k+mini_batch_size] for k in xrange(0,len(training_data),mini_batch_size)]

    def update_mini_batch(self, mini_batch, learning_rate, lmbda, n):
        nabla_w, nabla_b = self.gradient(mini_batch)
        m1 = 0.5
        m2 = 1 - m1
        eps = learning_rate/len(mini_batch)
        self.weights = [ w + (w_v - learning_rate*(lmbda/n)*w)*w_drop for w, w_v, w_drop in zip(self.weights, self.weights_v, self.weights_dropout) ]
        self.biases = [ b + b_v*b_drop for b, b_v, b_drop in zip(self.biases, self.biases_v, self.biases_dropout)]
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

