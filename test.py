import numpy as np
import matplotlib.pyplot as plt

# Setup
rng = np.random.RandomState(0)  # Seed RNG for replicability
n = 100  # Number of samples to draw

# Generate data
x = rng.normal(size=n)  # Sample 1: X ~ N(0, 1)
y = rng.standard_t(df=5, size=n)  # Sample 2: Y ~ t(5)
line = np.linspace(-3, 3, 100)

# Quantile-quantile plot
plt.figure()
plt.plot(line, line, color='r', label='y = x')
plt.scatter(np.sort(x), np.sort(y))
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
plt.close()