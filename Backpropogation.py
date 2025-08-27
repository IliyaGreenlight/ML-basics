import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def load_data(file_name):
    data = np.loadtxt(file_name)
    return data[:, 0], data[:, 1]


def split_data(x, y, test_size=0.2):
    return train_test_split(x, y, test_size=test_size, random_state=42)


class NeuralNetwork:
    def __init__(self, layers, activation):
        self.layers = layers
        self.activation = activation
        self.weights = [np.random.randn(layers[i], layers[i+1]) * 0.1 for i in range(len(layers) - 1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers) - 1)]

    def activate(self, x):
        if self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "relu":
            return np.maximum(0, x)

    def activate_derivative(self, x):
        if self.activation == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation == "sigmoid":
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif self.activation == "relu":
            return (x > 0).astype(float)

    def forward(self, x):
        self.a = []
        self.z = []
        for w, b in zip(self.weights, self.biases):
            z = x @ w + b
            self.z.append(z)
            x = self.activate(z)
            self.a.append(x)
        return x

    def backward(self, x, y, lr):
        m = x.shape[0]
        y = y.reshape(-1, 1)

        output = self.forward(x)

        dz = output - y

        for i in reversed(range(len(self.weights))):
            dw = self.a[i-1].T @ dz / m if i > 0 else x.T @ dz / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            dz = dz @ self.weights[i].T * self.activate_derivative(self.z[i-1]) if i > 0 else None

            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

    def train(self, x, y, epochs, lr, batch_size=None):
        for epoch in range(epochs):
            if batch_size:
                indices = np.random.permutation(len(x))
                x = x[indices]
                y = y[indices]
                for i in range(0, len(x), batch_size):
                    self.backward(x[i:i+batch_size], y[i:i+batch_size], lr)
            else:
                self.backward(x, y, lr)

    def predict(self, x):
        return self.forward(x)

    def evaluate(self, x, y):
        predictions = self.predict(x)
        mse = mean_squared_error(y, predictions)
        return mse

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate neural networks on dataset.")
    parser.add_argument("file_name", type=str, help="Name of the dataset file (e.g., daneXX.txt)")
    args = parser.parse_args()

    x, y = load_data(args.file_name)
    x = x.reshape(-1, 1)

    x_train, x_test, y_train, y_test = split_data(x, y)

    nn_tanh = NeuralNetwork([1, 10, 10, 1], activation="tanh")
    nn_tanh.train(x_train, y_train, epochs=1000, lr=0.01)
    mse_tanh = nn_tanh.evaluate(x_test, y_test)
    print("Tanh Network MSE:", mse_tanh)

    nn_relu = NeuralNetwork([1, 10, 10, 1], activation="relu")
    nn_relu.train(x_train, y_train, epochs=1000, lr=0.01)
    mse_relu = nn_relu.evaluate(x_test, y_test)
    print("ReLU Network MSE:", mse_relu)

    nn_sgd = NeuralNetwork([1, 10, 10, 1], activation="tanh")
    nn_sgd.train(x_train, y_train, epochs=1000, lr=0.01, batch_size=1)
    mse_sgd = nn_sgd.evaluate(x_test, y_test)
    print("SGD Tanh Network MSE:", mse_sgd)

if __name__ == "__main__":
    main()
