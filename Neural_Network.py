import tensorflow as tf
import numpy as np
import os


# Class describing our neural networkpip install Lasagne==0.1
class NeuralNetwork:
    # Placeholders for training data
    # Number of nodes
    number_of_layers = 3
    # Neural network
    network = None
    X_train, Y_train, X_test, Y_test = (None, None, None, None)

    def __init__(self, number_of_layers, learning_rate=0.1):
        self.number_of_layers = number_of_layers
        # Loading database
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.load_database()
        # We build our neural network
        self.network = self.build_cnn_model(number_of_layers) #self.build_fc_model(number_of_layers)
        # Define training function
        self.nn_compile(learning_rate)

    # Our neural network with 2 hidden layers that takes vector of images as an input
    def build_fc_model(self, number_of_layers=3):
        model = [tf.keras.layers.Flatten()]
        for i in range(number_of_layers):
            model.append(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.append(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        fc_model = tf.keras.Sequential(model)
        return fc_model

    def build_cnn_model(self, number_of_layers=3):
        cnn_model = tf.keras.Sequential([

            # TODO: Define the first convolutional layer
            tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), activation=tf.nn.relu),

            # TODO: Define the first max pooling layer
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

            # TODO: Define the second convolutional layer
            tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), activation=tf.nn.relu),

            # TODO: Define the second max pooling layer
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),

            # TODO: Define the last Dense layer to output the classification
            # probabilities. Pay attention to the activation needed a probability
            # output
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        cnn_model.predict(self.X_test[[0]])
        print(cnn_model.summary())
        return cnn_model

    # Function to download and load MNIST sources
    @staticmethod
    def load_database():
        mnist = tf.keras.datasets.mnist
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        X_train = (np.expand_dims(X_train, axis=-1) / 255.).astype(np.float32)
        Y_train = (Y_train).astype(np.int64)
        X_test = (np.expand_dims(X_test, axis=-1) / 255.).astype(np.float32)
        Y_test = (Y_test).astype(np.int64)

        return X_train, Y_train, X_test, Y_test

    # Declaring error and training functions
    def nn_compile(self, learning_rate=1e-1):
        self.network.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])

    # Training neural network
    def nn_training(self, batch_size, epos):
        from timeit import default_timer as timer

        time_start = timer()
        # Neural Network learning process
        self.network.fit(self.X_train, self.Y_train, batch_size=batch_size, epochs=epos)
        time_end = timer()
        print(self.network.summary())
        return time_end, time_start

    # Function to calculate accuracy of neural network
    def acc_fn(self):
        test_loss, test_acc = self.network.evaluate(self.X_test, self.Y_test)

        print('Test accuracy:', test_acc)

        # Assign function
        return test_acc

    # Function to predict the number by neural network
    def pr_fn(self, image):
        predictions = self.network.predict(image)
        print(predictions)
        return predictions