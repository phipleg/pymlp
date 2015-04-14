import mlp
import reuters

def main():
    words = 900
    training_data, validation_data, test_data = reuters.load(words)
    net = mlp.MLP([words,80,46])
    epochs = 200
    mini_batch_size = 16
    learning_rate = 0.5
    lmbda = 1.0
    drop_prob = 0.5
    net.sgd(training_data, epochs, mini_batch_size, test_data, learning_rate, lmbda, drop_prob)

if __name__ == "__main__": main()
