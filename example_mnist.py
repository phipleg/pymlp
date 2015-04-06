import mlp
import mnist

def main():
    tr_d, va_d, te_d = mnist.load()
    net = mlp.MLP([784,30,10])
    net.sgd(tr_d,300,10,te_d, 0.5, 5.0)

if __name__ == "__main__": main()
