import mlp
import mnist

def main():
    training_data, validation_data, test_data = mnist.load()
    net = mlp.MLP([784,36,10])
    epochs = 500
    mini_batch_size = 10
    learning_rate = 0.5
    lmbda = 5.0
    drop_prob = 0.5
    net.sgd(training_data, epochs, mini_batch_size, test_data, learning_rate, lmbda, drop_prob)

if __name__ == "__main__": main()
