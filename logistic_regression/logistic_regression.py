# import wget
# wget.download("https://raw.githubusercontent.com/suraggupta/coursera-machine-learning-solutions-python/master/Exercise2/Data/ex2data1.txt")
# wget.download("https://raw.githubusercontent.com/suraggupta/coursera-machine-learning-solutions-python/master/Exercise2/Data/ex2data2.txt")
import numpy as np
import matplotlib.pyplot as plt
class LogisticRegression:
    def __init__(self, learning_rate=0.001, iteration=500):
        self.learning_rate = learning_rate
        self.cost_history = []
        self.iteration = iteration

    def plot_data(self, X,y):
            pos = y == 1;
            neg = y == 0
            plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
            plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
            plt.xlabel('X1 values')
            plt.ylabel('X2 values')
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


    def sigmoid(self, z):
        sig = 1/(1+np.exp(-z))
        return sig


    def cost(self, X,y):
        theta = np.zeros(X.shape[1])
        h_theta = self.sigmoid(X.dot(theta.T))
        m=y.size
        cost = (1 / m) * np.sum(-y.dot(np.log(h_theta)) - (1 - y).dot(np.log(1 - h_theta)))
        grad=(1/ m) * (h_theta - y).dot(X)
        return cost, grad


    def gradient_descent_fit(self, X, y):
        X = self.reshape_features(X)
        m = y.size
        theta = np.zeros(X.shape[1])
        for i in range(self.iteration):
            h_theta = self.sigmoid(X.dot(theta.T))
            cost = (1 / m) * np.sum(-y.dot(np.log(h_theta)) - (1 - y).dot(np.log(1 - h_theta)))
            self.cost_history.append(cost)
            theta = theta - (self.learning_rate / m) * (h_theta - y).dot(X)
        return theta

    def predict(self, x_test, theta):
        if len(x_test.shape) == 1:
            x_test = x_test.reshape((1, len(x_test)))
        x_test=np.concatenate([np.ones((x_test.shape[0], 1)), x_test], axis=1)
        ht=np.dot(x_test,theta)
        prediction=self.sigmoid(ht)
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





# load multivariate dataset
print("************************* Example ************************")

data = np.loadtxt('ex2data1.txt', delimiter=",")
X = data[:, 0:2]
y = data[:, 2]
# 1. initialize linear regression class
logistic_rg=LogisticRegression(learning_rate=0.1, iteration=5000)
# 2. normalize the input features
X, mu, sigma= logistic_rg.normalize_features(X)
# 3. fit the the normalized training data
theta=logistic_rg.gradient_descent_fit(X,y)
print(theta)
# 4. Predict
X_test=np.array([45,85])
# normalize the prediction input
X_test = (X_test - mu) / sigma
print(logistic_rg.predict(X_test, theta))

