import numpy as np
import h5py
import matplotlib.pyplot as plt


class neuralNetwork:
    def __init__(self, num_iterations, layer_dims, learning_rate=0.0075, print_cost=True):
        self.cost_history = []
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.layer_dims = layer_dims
        self.print_cost = print_cost
        self.parameters = {}

    @staticmethod
    def sigmoid(Z):
        activation = 1 / (1 + np.exp(-Z))
        cache = Z
        return activation, cache

    @staticmethod
    def relu(Z):
        activation = np.maximum(0, Z)
        assert (activation.shape == Z.shape)
        cache = Z
        return activation, cache

    @staticmethod
    def linear_forward(activation, W, b):
        Z = np.dot(W, activation) + b
        assert (Z.shape == (W.shape[0], activation.shape[1]))
        cache = (activation, W, b)
        return Z, cache

    def initialize_parameters_deep(self):
        layer_dims = self.layer_dims
        np.random.seed(1)
        parameters = {}
        num_layers = len(layer_dims)  # number of layers in the network
        # loop over each layer
        for layer in range(1, num_layers):
            parameters['W' + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) / np.sqrt(
                layer_dims[layer - 1])  # *0.01
            parameters['b' + str(layer)] = np.zeros((layer_dims[layer], 1))
            # maker sure the shapes are correct
            assert (parameters['W' + str(layer)].shape == (layer_dims[layer], layer_dims[layer - 1]))
            assert (parameters['b' + str(layer)].shape == (layer_dims[layer], 1))
        return parameters

    def linear_activation_forward(self, activation_prev, W, b, activation):
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.linear_forward(activation_prev, W, b)  # This "linear_cache" contains (A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)  # This "activation_cache" contains "Z"
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.linear_forward(activation_prev, W, b)  # This "linear_cache" contains (A_prev, W, b)
            A, activation_cache = self.relu(Z)  # This "activation_cache" contains "Z"

        assert (A.shape == (W.shape[0], activation_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X, parameters):
        caches = []
        activation = X
        num_layers = len(parameters) // 2  # number of layers in the neural network
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, num_layers):
            activation_prev = activation
            activation, cache = self.linear_activation_forward(activation_prev,
                                                               parameters['W' + str(l)],
                                                               parameters['b' + str(l)], "relu")
            caches.append(cache)
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        activation_layer, cache = self.linear_activation_forward(activation,
                                                                 parameters['W' + str(num_layers)],
                                                                 parameters['b' + str(num_layers)], "sigmoid")
        caches.append(cache)
        assert (activation_layer.shape == (1, X.shape[1]))
        return activation_layer, caches

    @staticmethod
    def compute_cost(activation_layer, Y):
        m = Y.shape[1]
        # Compute loss from aL and y.
        cost = (-1 / m) * (np.dot(Y, np.log(activation_layer).T) + np.dot((1 - Y), np.log(1 - activation_layer).T))
        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert (cost.shape == ())

        return cost

    @staticmethod
    def relu_backward(drv_activation, cache):
        Z = cache
        dZ = np.array(drv_activation, copy=True)  # just converting dz to a correct object.
        # When z <= 0, you should set dz to 0 as well.
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
        return dZ

    @staticmethod
    def sigmoid_backward(drv_activation, cache):
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = drv_activation * s * (1 - s)
        assert (dZ.shape == Z.shape)
        return dZ

    @staticmethod
    def linear_backward(dZ, cache):
        # Here cache is "linear_cache" containing (A_prev, W, b) coming from the forward
        # propagation in the current layer
        activation_prev, W, b = cache
        m = activation_prev.shape[1]

        dW = (1 / m) * np.dot(dZ, activation_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        drv_activation_prev = np.dot(W.T, dZ)
        assert (drv_activation_prev.shape == activation_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return drv_activation_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        # if activation is selected as relu
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
        # if activation is sigmoid
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
        # call the linear backward method
        drv_activation_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return drv_activation_prev, dW, db

    def L_model_backward(self, activation_last_layer, Y, caches):
        grads = {}
        num_layers = len(caches)  # the number of layers
        m = activation_last_layer.shape[1]
        Y = Y.reshape(activation_last_layer.shape)  # after this line, Y is the same shape as AL
        # Initializing the backpropagation
        dAL = - (np.divide(Y, activation_last_layer) - np.divide(1 - Y, 1 - activation_last_layer))

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache".
        # Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = caches[num_layers - 1]  # Last Layer
        grads["dA" + str(num_layers - 1)], grads["dW" + str(num_layers)], grads["db" + str(num_layers)]= \
            self.linear_activation_backward(dAL, current_cache, "sigmoid")

        # Loop from l=L-2 to l=0
        for layer in reversed(range(num_layers - 1)):
            current_cache = caches[layer]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(layer + 1)],
                                                                             current_cache,
                                                                             activation="relu")
            grads["dA" + str(layer)] = dA_prev_temp
            grads["dW" + str(layer + 1)] = dW_temp
            grads["db" + str(layer + 1)] = db_temp
        return grads

    def update_parameters(self, parameters, grads):
        num_layers = len(parameters) // 2  # number of layers in the neural network
        # Update rule for each parameter. Use a for loop.
        for layer in range(num_layers):
            parameters["W" + str(layer + 1)] = parameters["W" + str(layer + 1)] - self.learning_rate * grads["dW" + str(layer + 1)]
            parameters["b" + str(layer + 1)] = parameters["b" + str(layer + 1)] - self.learning_rate * grads["db" + str(layer + 1)]
        return parameters

    def fit(self, X, Y):
        np.random.seed(1)
        # Parameters initialization. (≈ 1 line of code)
        parameters = self.initialize_parameters_deep()
        # Loop (gradient descent)
        for i in range(0, self.num_iterations):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            activation_layer, caches = self.L_model_forward(X, parameters)
            # Compute cost.
            cost = self.compute_cost(activation_layer, Y)
            # Backward propagation.
            grads = self.L_model_backward(activation_layer, Y, caches)
            # Update parameters.
            parameters = self.update_parameters(parameters, grads)

            # Print the cost every 100 training example
            if self.print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if self.print_cost and i % 100 == 0:
                self.cost_history.append(cost)

        self.parameters = parameters
        return parameters

    def evaluate(self, X, y):
        num_samples = X.shape[1]
        p = np.zeros((1, num_samples))
        # Forward propagation
        probabilities, caches = self.L_model_forward(X, self.parameters)
        # convert probabilities to 0/1 predictions
        for i in range(0, probabilities.shape[1]):
            if probabilities[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
        accuracy = np.sum((p == y) / num_samples)
        print("Accuracy: " + str(accuracy))
        return accuracy

    def predict(self, X):
        if len(X.shape)==1:
            num_samples=1
            X = X.reshape(len(X), 1)
        else:
            num_samples = X.shape[1]
        p = np.zeros((1, num_samples))
        # Forward propagation
        probabilities, caches = self.L_model_forward(X, self.parameters)
        # convert probas to 0/1 predictions
        for i in range(0, probabilities.shape[1]):
            if probabilities[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
        # p = np.where(p == 1, 'cat', 'dog')
        # print results
        print ("predictions: " + str(p))
        return p

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_history)), np.squeeze(self.cost_history))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()


"""*************************** Cats and Dogs dataset ************************************************************"""
class Data:
    def load_data(self):
        train_dataset = h5py.File('dataset/train_catvnoncat.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

        test_dataset = h5py.File('dataset/test_catvnoncat.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

        classes = np.array(test_dataset["list_classes"][:])  # the list of classes

        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    @staticmethod
    def data_info():
        print("Number of training examples: " + str(m_train))
        print("Number of testing examples: " + str(m_test))
        print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        print("train_x_orig shape: " + str(train_x_orig.shape))
        print("train_y shape: " + str(train_y.shape))
        print("test_x_orig shape: " + str(test_x_orig.shape))
        print("test_y shape: " + str(test_y.shape))


"""********************************* Model on cats and dogs dataset ***********************************************"""
# cats and dogs example
data_set = Data()
train_x_orig, train_y, test_x_orig, test_y, classes = data_set.load_data()

# Reshape the training and test examples. The "-1" makes reshape flatten the remaining dimensions
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255
test_x = test_x_flatten/255

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

# define the model
model = neuralNetwork(num_iterations=2500, layer_dims=[12288, 20, 7, 5, 1])
parameters = model.fit(train_x, train_y)
model.plot_cost()

model.evaluate(train_x, train_y)
model.predict(train_x[:, 1:10])
model.predict(train_x[:, 1])


