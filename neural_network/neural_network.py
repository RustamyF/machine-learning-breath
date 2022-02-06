# Package imports
import numpy as np
import matplotlib.pyplot as plt
import h5py

class TowLayerNN:
    def __init__(self, n_h=4, learning_rate=1.2, num_iteration=10000, print_cost=True):
        self.cost_history=[]
        self.parameters={}
        self.n_h = n_h
        self.learning_rate = learning_rate
        self.num_iterations = num_iteration
        self.print_cost = print_cost

    @staticmethod
    def sigmoid(x):
        s = 1/(1+np.exp(-x))
        return s

    def initialize_params(self, X, Y):
        n_h = self.n_h
        n_x = X.shape[0]
        n_y = Y.shape[0]
        # Initialize parameters
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))
        params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return params

    def forward(self, params, X):
        # load the parameters
        W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)
        cache={"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return cache

    @staticmethod
    def cost(cache, Y):
        # load A2 from the cache
        A2 = cache['A2']
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost"
        m = Y.shape[1]  # number of example
        # Compute the cross-entropy cost
        logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
        cost = (-1 / m) * np.sum(logprobs)
        return cost

    @staticmethod
    def back_prop(Y, cache, params, X):
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

    @staticmethod
    def update_param(params, grads, learning_rate):
        W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
        dW1, db1, dW2, db2 = grads["dW1"], grads["db1"], grads["dW2"], grads["db2"]

        # Update rule for each parameter
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return params

    def fit(self, X, Y):
        params = self.initialize_params(X, Y)
        for i in range(0, self.num_iterations):
            cache = self.forward(params, X)
            cost=self.cost(cache, Y)
            self.cost_history.append(cost)
            grads = self.back_prop(Y, cache, params, X)
            params = self.update_param(params, grads, self.learning_rate)
            # If print_cost=True, Print the cost every 1000 iterations
            if self.print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
        self.parameters = params

    @staticmethod
    def plot_cost():
        plt.plot(np.arange(len(model.cost_history)), model.cost_history)
        plt.grid()
        plt.xlabel('iteration number')
        plt.ylabel('cost')
        plt.show()


"""*************************** Cats and Dogs dataset ************************************************************"""
# cats and dogs data
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

"""**********************************Flower dataset *************************************************************"""
class Flowerdataset:
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
model=TowLayerNN(n_h=7, learning_rate=0.0075, num_iteration=2500, print_cost=True)
model.fit(train_x, train_y)
# model.plot_cost()


"""********************************* Model on flower dataset *****************************************************"""
# Flower dataset
data=Flowerdataset()
X, Y = data.load_planar_dataset()
print ("train_x's shape: " + str(X.shape))
model=TowLayerNN(n_h=7, learning_rate=1.1, num_iteration=10000, print_cost=True)
model.fit(X, Y)
# Visualize the data:
# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
# plt.show()











