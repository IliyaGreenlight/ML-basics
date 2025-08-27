import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train_flatten = x_train.reshape((x_train.shape[0], -1))
x_test_flatten = x_test.reshape((x_test.shape[0], -1))

input_dim = x_train_flatten.shape[1]
encoding_dim = 64

input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

autoencoder.fit(x_train_flatten, x_train_flatten, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test_flatten, x_test_flatten))

encoder = Model(input_layer, encoded)

encoded_data = encoder.predict(x_train_flatten)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(encoded_data, y_train)
y_pred = knn.predict(encoded_data)

report = classification_report(y_train, y_pred, output_dict=False)

umap_reducer = umap.UMAP(n_components=2, random_state=42)
encoded_2d = umap_reducer.fit_transform(encoded_data)

plt.figure(figsize=(10, 8))
for i in range(10):
    idxs = y_train == i
    plt.scatter(encoded_2d[idxs, 0], encoded_2d[idxs, 1], label=str(i), s=2)
plt.title("UMAP Visualization of Encoded MNIST Data")
plt.legend()
plt.show()

report
