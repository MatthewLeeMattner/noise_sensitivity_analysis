import numpy as np
import matplotlib.pyplot as plt

inp = np.array([0.2, 2.1, 3.5, -1, -10])
weight_matrix = np.array([0, 1, 2, 1, 0])

def predict(inp, weight):
    return np.sum(inp * weight)

def normalize(arr):
    return (arr - np.min(arr))/(np.max(arr) - np.min(arr))

baseline = predict(inp, weight_matrix)
k = 100
noise_results = []
results = []
for i in range(k):
    noise = np.random.randn(inp.shape[0])
    noise_inp = inp + noise
    out = predict(noise_inp, weight_matrix)
    noise_results.append(noise)
    results.append(out)

results = np.array(results)
noise_results = np.array(noise_results)

print("Noise Results: ", noise_results.shape)
print("Results: ", results.shape)
#print("Deltas: ", results-baseline)

results_baseline = np.array(results - baseline)

noise_results_abs = np.abs(noise_results)
#print(noise_results_abs)

results = np.matmul(results_baseline, noise_results_abs)
print(normalize(results))
