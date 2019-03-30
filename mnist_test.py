import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D((5, 5), 32, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(strides=2),
    tf.keras.layers.Conv2D((5, 5), 64),
    tf.keras.layers.MaxPool2D(strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(np.expand_dims(x_train, axis=3), y_train, epochs=5)


# ==================================================
from occlusion_interpretability import OcclusionInterpretability
import matplotlib.pyplot as plt

for t in x_test:
    img = np.expand_dims(t, axis=2)
    img_orig = np.expand_dims(img, axis=0)
    max_index = np.argmax(model.predict(img_orig))
    print(max_index)

    oc = OcclusionInterpretability(model)
    outputs = oc.convolution_occlusion(img, batch=1)
    img_sensitivity = outputs[:, :, max_index]
    img_sens_reshape = np.reshape(img_sensitivity, (28, 28))

    _, axs = plt.subplots(1, 2, figsize=(12, 12))
    axs[0].imshow(np.reshape(img, (28, 28)))
    axs[1].imshow(img_sens_reshape)
    plt.show()

'''
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
'''
