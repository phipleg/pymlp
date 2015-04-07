import gzip
import cPickle
import numpy as np

def load(wordlimit=10000):
    path = "data/reuters.pkl.gz"
    f = gzip.open(path, "rb")
    X,Y = cPickle.load(f)
    f.close()
    n = len(X)
    split = 0.1
    k = int(split * n)
    training_inputs = [vectorized_sequence(x, wordlimit) for x in X[0:n-2*k]]
    training_results = [vectorized_label(y) for y in Y[0:n-2*k]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [vectorized_sequence(x, wordlimit) for x in X[n-2*k:n-k]]
    validation_data = zip(validation_inputs, Y[n-2*k:n-k])
    test_inputs = [vectorized_sequence(x, wordlimit) for x in X[n-2*k:n-k]]
    test_data = zip(test_inputs, Y[n-2*k:n-k])
    return (training_data, validation_data, test_data)

def vectorized_label(label):
    e = np.zeros((46, 1))
    e[label] = 1.0
    return e

def vectorized_sequence(seq, limit):
    e = np.zeros((limit,1))
    for i in seq:
        if i < limit:
            e[i] = 1
    return e

if __name__ == "__main__":
    training_data, validation_data, test_data = load()
    print "Reuters news wire corpus. training={}, validation={}, test={}".format(len(training_data), len(validation_data), len(test_data))
    