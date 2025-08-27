import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def prepare_network(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_training_history(history, output_file=None):
    plt.figure(figsize=(12, 5))


    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')


    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()


def evaluate_model(model, x_test, y_test, class_names, confusion_file=None):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))


    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    if confusion_file:
        plt.savefig(confusion_file)
    else:
        plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10 dataset.")
    parser.add_argument('-o', '--output_model', default='cnn_model.h5', help='Output model file.')
    parser.add_argument('-hist', '--history_image', help='File to save training history plot.')
    parser.add_argument('-c', '--confusion_matrix', help='File to save confusion matrix plot.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    x_train, y_train, x_test, y_test = load_cifar10()
    class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    model = prepare_network(x_train.shape[1:], len(class_names))
    history = model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2)
    model.save(args.output_model)
    plot_training_history(history, args.history_image)
    evaluate_model(model, x_test, y_test, class_names, args.confusion_matrix)


if __name__ == '__main__':
    main()
