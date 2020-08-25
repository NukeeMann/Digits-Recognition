import lasagne as lsg
import theano
import theano.tensor as T
import numpy as np
import os


# Class describing our neural networkpip install Lasagne==0.1
class NeuralNetwork:
    # Placeholders for training data
    # Empty 4 dimensional array of images
    input_var = T.tensor4('inputs')
    # One dimensional vector of labels
    target_var = T.ivector('targets')
    # Number of nodes
    num_of_nodes = 80
    # Training function
    training_fn = None
    # Neural network
    network = None
    # Accuracy function
    acc_fn = None
    # Prediction function
    pr_fn = None

    def __init__(self, num_of_nodes, learning_rate=0.1, momentum=0.9):
        self.num_of_nodes = num_of_nodes
        # We build our neural network
        self.network = self.build_neural_network(self.input_var, num_of_nodes)
        # Define training function
        self.training_fn = self.train_function(learning_rate, momentum)
        # Define accuracy function
        self.acc_fn = self.nn_accuracy()
        # Define prediction function
        self.pr_fn = self.prediction_fn()

    # Our neural network with 2 hidden layers that takes vector of images as an input
    @staticmethod
    def build_neural_network(input_variables=input_var, num_of_nodes=21):
        # Input layer that takes input data. One index in vector determines one matrix of image
        layer_input = lsg.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_variables)
        # We drop 50% of wages to avoid overfitting
        layer_input_drop = lsg.layers.DropoutLayer(layer_input, p=0.50)

        # First hidden layer. 800 nodes. Takes Input layer as a input. rectify function.
        layer_hidden_1 = lsg.layers.DenseLayer(layer_input_drop,
                                               # Number of nodes
                                               num_units=num_of_nodes,
                                               # Activation function - Tanh (S-shape from -1 to 1)
                                               nonlinearity=lsg.nonlinearities.tanh,
                                               # Wages for sigmoid. Glorot/Xavier initialization with Uniform distribution
                                               W=lsg.init.GlorotUniform(gain=1.0))
        layer_hidden_1_drop = lsg.layers.DropoutLayer(layer_hidden_1, p=0.50)

        layer_hidden_2 = lsg.layers.DenseLayer(layer_hidden_1_drop,
                                               num_units=num_of_nodes,
                                               nonlinearity=lsg.nonlinearities.tanh,
                                               W=lsg.init.GlorotUniform(gain=1.0))
        layer_hidden_2_drop = lsg.layers.DropoutLayer(layer_hidden_2, p=0.50)

        layer_out = lsg.layers.DenseLayer(layer_hidden_2_drop,
                                          num_units=10,
                                          # Softmax activation function
                                          nonlinearity=lsg.nonlinearities.softmax,
                                          W=lsg.init.GlorotUniform(gain=1.0))

        return layer_out

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
                data = data.reshape(-1, 1, 28, 28)
                # Convert binary to float from 0 to 1
            return data / np.float32(256)

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

        return X_train, Y_train, X_test, Y_test

    # Declaring error and training functions
    def train_function(self, learning_rate_val=0.1, momentum_val=0.9):
        # Compute an error function
        # Get prediction for current training set
        prediction = lsg.layers.get_output(self.network)
        # Computes the categorical cross-entropy between predictions and targets
        loss = lsg.objectives.categorical_crossentropy(prediction, self.target_var)
        # Aggregate it into scalar
        loss = loss.mean()

        # Getting all params
        params = lsg.layers.get_all_params(self.network, trainable=True)
        # Setting Nesterov accelerated gradient descending update function
        updates = lsg.updates.nesterov_momentum(loss, params, learning_rate=learning_rate_val, momentum=momentum_val)

        # Declaring updating function
        train_fn = theano.function([self.input_var, self.target_var], loss, updates=updates)

        return train_fn

    # Training neural network
    def nn_training(self, x_train, y_train, X_test, Y_test, epos):
        from timeit import default_timer as timer

        time_start = timer()
        # Neural Network learning process
        for step in range(epos):
            train_err = self.training_fn(x_train, y_train)
            if step % 10 == 0:
                print("Current step is " + str(step))
                print("Train-error: " + str(train_err) + "  Train-acc: " + str(self.acc_fn(X_test, Y_test) * 100) + "%\n")
        time_end = timer()

        return time_end, time_start

    # Function to calculate accuracy of neural network
    def nn_accuracy(self):
        # Get prediction
        test_prediction = lsg.layers.get_output(self.network, deterministic=True)
        # Function to count accuracy of our network
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.target_var), dtype=theano.config.floatX)

        # Assign function
        return theano.function([self.input_var, self.target_var], test_acc)

    # Function to predict the number by neural network
    def prediction_fn(self):
        prediction = lsg.layers.get_output(self.network)
        return theano.function([self.input_var], prediction)