import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)


# ==================================================

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True)

def normalize(arr):
    return (arr - np.min(arr))/(np.max(arr) - np.min(arr))

k = 500000
noise = np.round(np.random.normal(0, 0.5, (k, 28, 28)))
print(noise)
for x in x_test:
    baseline = np.max(model.predict(np.expand_dims(x, axis=0)))
    noise_inp = x + noise
    out = np.max(model.predict(noise_inp), axis=1)
    results_baseline = out - baseline
    results = np.matmul(results_baseline, np.reshape(noise, [noise.shape[0], 784]))
    std = np.std(results)
    mean = np.mean(results)

    pos_results = np.clip(results-mean, 0, mean + std)
    neg_results = np.abs(np.clip(results+mean, -(mean + std), 0))

    _, axs = plt.subplots(1, 4, figsize=(12, 12))

    axs[0].imshow(x)
    axs[1].imshow(np.reshape(results, (28, 28)))
    axs[2].imshow(np.reshape(pos_results, (28, 28)))
    axs[3].imshow(np.reshape(neg_results, (28, 28)))
    plt.show()

