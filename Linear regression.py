import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def load_data(file_name):
    data = np.loadtxt(file_name)
    return data[:, 0], data[:, 1]

def linear_model(x, w):
    return w[0] + w[1] * x


def train_linear_model(x_train, y_train, epochs=1000, lr=0.01):
    w = np.random.randn(2)
    for _ in range(epochs):
        predictions = linear_model(x_train, w)
        errors = y_train - predictions

        w[0] += lr * errors.mean()
        w[1] += lr * (errors * x_train).mean()
    return w


def polynomial_model(x, w):
    return w[0] + w[1] * x + w[2] * x**2


def train_polynomial_model(x_train, y_train, epochs=1000, lr=0.01):
    w = np.random.randn(3)
    for _ in range(epochs):
        predictions = polynomial_model(x_train, w)
        errors = y_train - predictions

        w[0] += lr * errors.mean()
        w[1] += lr * (errors * x_train).mean()
        w[2] += lr * (errors * x_train**2).mean()
    return w


def evaluate_model(x, y, model, w):
    predictions = model(x, w)
    mse = mean_squared_error(y, predictions)
    return mse


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models on dataset.")
    parser.add_argument("file_name", type=str, help="Name of the dataset file (e.g., daneXX.txt)")
    args = parser.parse_args()

    x, y = load_data(args.file_name)


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    w1 = train_linear_model(x_train, y_train)
    mse1_train = evaluate_model(x_train, y_train, linear_model, w1)
    mse1_test = evaluate_model(x_test, y_test, linear_model, w1)
    print("Model 1 (Linear):")
    print(f"  Weights: {w1}")
    print(f"  Train MSE: {mse1_train:.4f}")
    print(f"  Test MSE: {mse1_test:.4f}")


    w2 = train_polynomial_model(x_train, y_train)
    mse2_train = evaluate_model(x_train, y_train, polynomial_model, w2)
    mse2_test = evaluate_model(x_test, y_test, polynomial_model, w2)
    print("\nModel 2 (Polynomial):")
    print(f"  Weights: {w2}")
    print(f"  Train MSE: {mse2_train:.4f}")
    print(f"  Test MSE: {mse2_test:.4f}")


    print("\nComparison:")
    print(f"  Model 1 Test MSE: {mse1_test:.4f}")
    print(f"  Model 2 Test MSE: {mse2_test:.4f}")
    print("  Model 2 performs better" if mse2_test < mse1_test else "  Model 1 performs better")

if __name__ == "__main__":
    main()
