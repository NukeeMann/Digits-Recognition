import lasagne as lsg
import theano
import theano.tensor as T
import numpy as np
import os


# Class describing our neural networkpip install Lasagne==0.1
class NeuralNetwork:
    # alpha
    alpha = None
    # Hidden layer size
    layer_size = None
    # Size of the image
    pixels = None
    # Number of labels
    num_labels = None
    # Weights
    weights_1 = None
    weights_2 = None

    def __init__(self, num_of_nodes, alpha_val=2, img_size=784, number_of_labels=10):
        self.alpha = alpha_val
        self.layer_size = num_of_nodes
        self.pixels = img_size
        self.num_labels = number_of_labels
        self.weights_1 = 0.02 * np.random.random((self.pixels, self.layer_size)) - 0.01
        self.weights_2 = 0.2 * np.random.random((self.layer_size, self.num_labels)) - 0.1

    # Function to download and load MNIST sources
    @staticmethod
    def load_database():
        # Download from website
        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading ", filename)
            import urllib.request as urllib
            urllib.urlretrieve(source + filename, filename)

        import gzip

        # Load mnist images to vector of matrix 28x28
        def load_mnist_images(filename):
            if not os.path.exists(filename):
                download(filename)
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
                # -1 <- Declare size of vector as big as number of images, 1 <- Monochromatic, 28x28 size of matrix
                data = data.reshape(-1, 28, 28)
                # Convert binary to float from 0 to 1
            return data

        # load mnist labels of images
        def load_mnist_labels(filename):
            if not os.path.exists(filename):
                download(filename)
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            return data

        # Training data
        X_train = load_mnist_images('train-images-idx3-ubyte.gz')
        Y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')

        # Testing data
        X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
        Y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

        one_hot_labels = np.zeros((len(Y_train), 10))
        for i, j in enumerate(Y_train):
            one_hot_labels[i][j] = 1
        Y_train = one_hot_labels
        X_test = X_test.reshape(len(X_test), 784) / 255
        Y_test_2 = np.zeros((len(Y_test), 10))
        for i, j in enumerate(Y_test):
            Y_test_2[i][j] = 1
        Y_test = Y_test_2

        X_train, Y_train = (X_train[0:10000].reshape(10000, 784) / 255, Y_train[0:10000])

        return X_train, Y_train, X_test, Y_test

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh2deriv(output):
        return 1 - (output ** 2)

    @staticmethod
    def softmax(x):
        tmp_e = np.exp(x)
        return tmp_e / np.sum(tmp_e, axis=1, keepdims=True)

    # Neural network training
    def train_nn(self, X_train, Y_train, X_test, Y_test, epos_num, batch_size):
        from timeit import default_timer as timer

        time_start = timer()
        for j in range(epos_num):
            # Training
            correct_cnt = 0
            for i in range(int(len(X_train) / batch_size)):
                batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))

                layer_0 = X_train[batch_start:batch_end]

                layer_1 = self.tanh(np.dot(layer_0, self.weights_1))
                layer_1_drop = np.random.randint(2, size=layer_1.shape)
                layer_1 *= layer_1_drop * 2

                layer_2 = self.softmax(np.dot(layer_1, self.weights_2))

                for k in range(batch_size):
                    correct_cnt += int(
                        np.argmax(layer_2[k:k + 1]) == np.argmax(Y_train[batch_start + k:batch_end + k + 1]))

                layer_2_delta = (Y_train[batch_start:batch_end] - layer_2)
                layer_2_delta /= (batch_size * layer_2.shape[0])
                layer_1_delta = layer_2_delta.dot(self.weights_2.T) * self.tanh2deriv(layer_1)

                layer_1_delta *= layer_1_drop

                self.weights_1 += self.alpha * layer_0.T.dot(layer_1_delta)
                self.weights_2 += self.alpha * layer_1.T.dot(layer_2_delta)

            test_correct_cnt = 0
            for i in range(len(X_test)):
                layer_0 = X_test[i:i + 1]
                layer_1 = self.tanh(np.dot(layer_0, self.weights_1))
                layer_2 = np.dot(layer_1, self.weights_2)
                test_correct_cnt += int(np.argmax(layer_2) == np.argmax(Y_test[i:i + 1]))

            if j % 10 == 0:
                print("I: " + str(j) + " Test-acc: " + str(test_correct_cnt / float(len(X_test))) + " Train-acc: "
                      + str(correct_cnt / float(len(X_train))))

        time_end = timer()
        return time_end, time_start

    # Predict network
    def predict(self, img):
        layer_0 = img.reshape(1, 784)
        layer_1 = self.tanh(np.dot(layer_0, self.weights_1))
        layer_2 = np.dot(layer_1, self.weights_2)
        return layer_2
