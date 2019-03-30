import tensorflow as tf
import numpy as np
import cv2 as cv
import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
from occlusion_interpretability import OcclusionInterpretability

model = ResNet50(weights='imagenet')


img_path = 'staffi.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# ==================================================
oc = OcclusionInterpretability(model)
img = np.squeeze(x, axis=0)
outputs = oc.convolution_occlusion(img, (12, 12))
print(outputs.shape)
'''
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True)

def normalize(arr):
    return (arr - np.min(arr))/(np.max(arr) - np.min(arr))

k = 100
noise = np.random.normal(0, 1, (k, 244, 244, 3))
for x in x_test:
    baseline = np.max(model.predict(np.expand_dims(x, axis=0)))
    noise_results = []
    results = []
    noise_inp = x + noise
    out = np.max(model.predict(noise_inp), axis=1)
    results_baseline = out - baseline
    results = np.matmul(results_baseline, np.reshape(noise, [noise.shape[0], noise.shape[1] * noise.shape[2] * noise.shape[3]]))
    std = np.std(results)
    results = np.clip(results, -std, std)
    _, axs = plt.subplots(1, 2, figsize=(12, 12))
    axs[0].imshow(x)
    axs[1].imshow(np.reshape(results, (28, 28)))
    plt.show()
'''
