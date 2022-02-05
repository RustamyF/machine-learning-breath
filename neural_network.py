# Package imports
import numpy as np
import matplotlib.pyplot as plt

class dataset:
    def load_planar_dataset(self):
        np.random.seed(1)
        m = 400  # number of examples
        N = int(m / 2)  # number of points per class
        D = 2  # dimensionality
        X = np.zeros((m, D))  # data matrix where each row is a single example
        Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
        a = 4  # maximum ray of the flower

        for j in range(2):
            ix = range(N * j, N * (j + 1))
            t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
            r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            Y[ix] = j

        X = X.T
        Y = Y.T

        return X, Y
data=dataset()
X, Y = data.load_planar_dataset()

# Visualize the data:
# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
# plt.show()

class NN:
    def __init__(self):
        self.cost_history=[]
        self.parameters={}

    def sigmoid(self, x):
        s = 1/(1+np.exp(-x))
        return s
    def initialize_params(self, X, Y, n_h):
        n_x = X.shape[0]
        n_y = Y.shape[0]
        # Initialize parameters
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))
        params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return params

    def forward(self, params):
        # load the parameters
        W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)
        cache={"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return cache

    def cost(self, cache, Y):
        # load A2 from the cache
        A2 = cache['A2']
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost"
        m = Y.shape[1]  # number of example
        # Compute the cross-entropy cost
        logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
        cost = (-1 / m) * np.sum(logprobs)
        return cost
    def back_prop(self, Y, cache, params):
        A2, A1, W2 = cache['A2'], cache['A1'], params['W2']
        m = Y.shape[1]  # number of example
        # Backward propagation: calculate dW1, db1, dW2, db2.
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * (np.sum(dZ2, axis=1, keepdims=True))
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = (1 / m) * (np.dot(dZ1, X.T))
        db1 = (1 / m) * (np.sum(dZ1, axis=1, keepdims=True))
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads
    def update_param(self, params, grads, learning_rate):
        W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
        dW1, db1, dW2, db2 = grads["dW1"], grads["db1"], grads["dW2"], grads["db2"]

        # Update rule for each parameter
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return params
    def fit(self, X, Y, n_h, learning_rate, num_iterations, print_cost=True):
        params = self.initialize_params(X, Y, n_h)
        for i in range(0, num_iterations):
            cache = self.forward(params)
            cost=self.cost(cache, Y)
            self.cost_history.append(cost)
            grads = self.back_prop(Y, cache, params)
            params = self.update_param(params, grads, learning_rate)
            # If print_cost=True, Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
        self.parameters=params
    def plot_cost(self):
        plt.plot(np.arange(len(model.cost_history)), model.cost_history)
        plt.grid()
        plt.xlabel('iteration number')
        plt.ylabel('cost')
        plt.show()


model=NN()
model.fit(X, Y, n_h=4, learning_rate=1.2, num_iterations=10000, print_cost=True)
model.plot_cost()

class NeuralNetwork:
    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s
    def nn_model(self, X, Y, n_h, learning_rate, num_iterations=10000, print_cost=False):
        n_x = X.shape[0]
        n_y = Y.shape[0]

        # Initialize parameters
        W1 = np.random.randn(n_h,n_x) * 0.01
        b1 = np.zeros((n_h,1))
        W2 = np.random.randn(n_y,n_h) * 0.01
        b2 = np.zeros((n_y,1))

        # Loop (gradient descent)
        for i in range(0, num_iterations):
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache"
            Z1 = np.dot(W1, X) + b1
            A1 = np.tanh(Z1)
            Z2 = np.dot(W2, A1) + b2
            A2 = self.sigmoid(Z2)

            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost"
            m = Y.shape[1]  # number of example
            # Compute the cross-entropy cost
            logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
            cost = (-1 / m) * np.sum(logprobs)
            # cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect.

            # Backward propagation: calculate dW1, db1, dW2, db2.
            dZ2 = A2 - Y
            dW2 = (1 / m) * np.dot(dZ2, A1.T)
            db2 = (1 / m) * (np.sum(dZ2, axis=1, keepdims=True))
            dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
            dW1 = (1 / m) * (np.dot(dZ1, X.T))
            db1 = (1 / m) * (np.sum(dZ1, axis=1, keepdims=True))

            # Update rule for each parameter
            W1 = W1 - learning_rate * dW1
            b1 = b1 - learning_rate * db1
            W2 = W2 - learning_rate * dW2
            b2 = b2 - learning_rate * db2

            # If print_cost=True, Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        # Returns parameters learnt by the model. They can then be used to predict output
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return parameters

# parameters = nn_model(X, Y, 4, 1.2,num_iterations=10000, print_cost=True)


