import numpy as np

np.random.seed(411)

def unit_uniform_point(dimensions):
    return np.random.uniform(0, 1, dimensions)

def squared_euc_dist(vec1, vec2):
    return ((vec1 - vec2) ** 2).sum()

def sample_distances(samples, dimensions):
    return np.array([
        squared_euc_dist(unit_uniform_point(dimensions), unit_uniform_point(dimensions))
        for i in range(samples)
    ])

def distance_statistics(samples, dimensions):
    distances = sample_distances(samples, dimensions)
    return np.mean(distances), np.var(distances)

print("Dimension, Expectation, Variance")
for i in range(1, 1000):
    print(i, *distance_statistics(10000, i))