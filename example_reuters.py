import mlp
import reuters

def main():
    training_data, validation_data, test_data = reuters.load(10000)
    net = mlp.MLP([10000,80,46])
    epochs = 200
    learning_rate = 1.0
    lmbda = 1.0
    mini_batch_size = 16
    net.sgd(training_data, epochs, mini_batch_size, test_data, learning_rate, lmbda)

if __name__ == "__main__": main()
