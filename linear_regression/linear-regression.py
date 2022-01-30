# import wget
# wget.download("https://raw.githubusercontent.com/suraggupta/coursera-machine-learning-solutions-python/master/Exercise1/Data/ex1data1.txt")
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, iteration=500):
        self.learning_rate = learning_rate
        self.cost_history = []
        self.iteration = iteration

    def plot_data(self, X,y):
        plt.plot(X,y, '*')
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.grid()
        plt.show()

    def reshape_features(self, X):
        if len(X.shape) == 1:
            m=len(X)
            X = np.stack((np.ones(m), X), axis=1)
        else:
            m=X.shape[0]
            X = np.concatenate([np.ones((m, 1)), X], axis=1)
        return X
    def normalize_features(self, X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        X_norm = (X - mu) / sigma

        return X_norm, mu, sigma

    def normal_equation_fit(self, X, y):
        X = self.reshape_features(X)
        theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        return theta

    def gradient_descent_fit(self, X, y):
        X = self.reshape_features(X)
        m = y.size
        theta = np.zeros(X.shape[1])
        for i in range(self.iteration):
            h_theta = np.dot(X, theta)
            cost = (1 / (2 * m)) * (np.sum(np.square(h_theta - y)))
            self.cost_history.append(cost)
            theta = theta - (self.learning_rate / m) * (h_theta - y).dot(X)
        return theta

    def predict(self, x_test, theta):
        if len(x_test.shape) == 1:
            x_test = x_test.reshape((1, len(x_test)))
        x_test=np.concatenate([np.ones((x_test.shape[0], 1)), x_test], axis=1)
        prediction=np.dot(x_test,theta)
        return prediction

    def plot_cost(self):
        x_values = np.arange(self.iteration)
        y_values = self.cost_history
        plt.plot(x_values, y_values, label='cost')
        plt.xlabel('iteration #')
        plt.ylabel('cost')
        plt.legend()
        plt.grid()
        plt.show()



# load univariete dataset
print("************************* univariate data ************************")
data = np.loadtxt('ex1data1.txt', delimiter=",")
X = data[:, 0]
y = data[:, 1]
# 1. initialize linear regression class
linear_reg = LinearRegression()
# 2. fit the the normalized training data
theta = linear_reg.normal_equation_fit(X, y)
print('theta using normal equation method', theta)
linear_reg = LinearRegression(learning_rate=0.01, iteration=1000)
theta = linear_reg.gradient_descent_fit(X, y)
print('theta using gradient descent method', theta)
# 3. Predict on one sample
# print('prediction for one sample', linear_reg.predict([X[4]], theta))



# load multivariate dataset
print("************************* multivariate data ************************")
data = np.loadtxt('ex1data2.txt', delimiter=",")
X = data[:, 0:2]
y = data[:, 2]

# 1. initialize linear regression class
linear_reg = LinearRegression(learning_rate=0.1, iteration=500)
# 2. normalize the input features
X, mu, sigma= linear_reg.normalize_features(X)
# 3. fit the the normalized training data
theta = linear_reg.gradient_descent_fit(X, y)
print('theta using gradient descent method', theta)
theta = linear_reg.normal_equation_fit(X, y)
print('theta using normal equation method', theta)
# 4. Predict
X_test = np.array([1650, 3])
# normalize the prediction input
X_test = (X_test - mu) / sigma
print('prediction for one sample', linear_reg.predict(X_test, theta))


