import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv("xy_data.csv")

x = data["x"].values
y = data["y"].values


# Calculate error
def error(params):

    theta, M, X = params

    theta = np.radians(theta)

    # Change coordinates
    xx = x - X
    yy = y - 42

    # Rotate coordinates
    t = xx * np.cos(theta) + yy * np.sin(theta)

    v = -xx * np.sin(theta) + yy * np.cos(theta)

    # Expected value
    predicted = np.exp(M * np.abs(t)) * np.sin(0.3 * t)

    return v - predicted


# Starting values
initial = [25, 0, 50]

# Allowed ranges
lower = [0, -0.05, 0]
upper = [50, 0.05, 100]

# Find the best values
result = least_squares(
    error,
    initial,
    bounds=(lower, upper)
)

theta, M, X = result.x

print("Final Answer")
print("------------")
print("Theta =", theta)
print("M     =", M)
print("X     =", X)


# Plot the result
theta = np.radians(theta)

t = np.linspace(6, 60, 1000)

x_curve = (
    t * np.cos(theta)
    - np.exp(M * np.abs(t))
    * np.sin(0.3 * t)
    * np.sin(theta)
    + X
)

y_curve = (
    42
    + t * np.sin(theta)
    + np.exp(M * np.abs(t))
    * np.sin(0.3 * t)
    * np.cos(theta)
)

plt.scatter(x, y, s=5, label="CSV data")
plt.plot(x_curve, y_curve, label="Fitted curve")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()

plt.show()