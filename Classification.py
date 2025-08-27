import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, input_dim, learning_rate=0.01, epochs=100):
        self.weights = np.zeros((input_dim + 1,))  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, X):
        linear_output = np.dot(X, self.weights[1:]) + self.weights[0]
        return np.where(linear_output > 0, 1, 0)

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                update = self.learning_rate * (target - prediction)
                self.weights[1:] += update * xi
                self.weights[0] += update


class MultiClassPerceptron:
    def __init__(self, n_classes, input_dim, learning_rate=0.01, epochs=100):
        self.perceptrons = [Perceptron(input_dim, learning_rate, epochs) for _ in range(n_classes)]

    def train(self, X, y):
        for i, perceptron in enumerate(self.perceptrons):
            binary_target = (y == i).astype(int)
            perceptron.train(X, binary_target)

    def predict(self, X):
        predictions = np.array([perceptron.predict(X) for perceptron in self.perceptrons]).T
        return np.argmax(predictions, axis=1)


class LogisticRegression:
    def __init__(self, input_dim, learning_rate=0.01, epochs=100):
        self.weights = np.zeros((input_dim + 1, 3))  # Multi-class weights, 3 classes for Iris dataset
        self.learning_rate = learning_rate
        self.epochs = epochs

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def train(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
        for _ in range(self.epochs):
            logits = np.dot(X, self.weights)
            predictions = self.softmax(logits)
            gradient = np.dot(X.T, (predictions - y)) / X.shape[0]
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        logits = np.dot(X, self.weights)
        probabilities = self.softmax(logits)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        logits = np.dot(X, self.weights)
        return self.softmax(logits)


def one_hot_encode(y, n_classes):
    one_hot = np.zeros((y.size, n_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot


# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encoding for logistic regression
y_train_onehot = one_hot_encode(y_train, n_classes=3)

# Train Perceptron-based multi-class classifier
perceptron_classifier = MultiClassPerceptron(n_classes=3, input_dim=X_train.shape[1])
perceptron_classifier.train(X_train, y_train)
perceptron_predictions = perceptron_classifier.predict(X_test)
print("Perceptron Accuracy:", np.mean(perceptron_predictions == y_test))

# Train Logistic Regression multi-class classifier
logistic_regression = LogisticRegression(input_dim=X_train.shape[1])
logistic_regression.train(X_train, y_train_onehot)
logistic_predictions = logistic_regression.predict(X_test)
logistic_probabilities = logistic_regression.predict_probabilities(X_test)
print("Logistic Regression Accuracy:", np.mean(logistic_predictions == y_test))

# Print probabilities for the first sample in the test set
print("Probabilities for the first test sample:", logistic_probabilities[0])
