import numpy as np
import matplotlib.pyplot as plt

x_1, x_2 = 0.05, 0.1   # Input
y_1, y_2 = 0.01, 0.99  # Output
b_1, b_2 = 0.35, 0.6   # Bias
w = [0.5]*8            # Initial value for weights. Note that Python starts arrays with 0, so w[0] is w_1
#w = [0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.55]
n = 0.5                # Learning rate


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_diff(x):
    e = np.exp(-x)
    return e/(1 + e)**2

errors = []
counter = 1
for epoch in range(8000):

    ### Forward Feeding ###
    # Step 1:
    h_1 = w[0] * x_1 + w[2] * x_2 + b_1
    h_2 = w[1] * x_1 + w[3] * x_2 + b_1
    # Step 1.5:
    out_h_1 = sigmoid(h_1)
    out_h_2 = sigmoid(h_2)
    # Step 2:
    y_tilde_1 = w[4] * out_h_1 + w[6] * out_h_2 + b_2
    y_tilde_2 = w[5] * out_h_1 + w[7] * out_h_2 + b_2

    out_y_tilde_1 = sigmoid(y_tilde_1)
    out_y_tilde_2 = sigmoid(y_tilde_2)

    # Step 3:
    error = 0.5 * ((y_1 - out_y_tilde_1)**2 + (y_2 - out_y_tilde_2)**2)
    print('The error after ', counter, ' epochs: ', error)
    errors.append(error)  # For the visualization

    ### Back Propagation ###
    part_1 = -(y_1 - out_y_tilde_1) * sigmoid_diff(y_tilde_1)
    part_2 = -(y_2 - out_y_tilde_2) * sigmoid_diff(y_tilde_2)
    w[0] -= n * (part_1 * w[4] * sigmoid_diff(h_1) * x_1 + part_2 * w[5] * sigmoid_diff(h_1) * x_1)
    w[1] -= n * (part_1 * w[6] * sigmoid_diff(h_2) * x_1 + part_2 * w[7] * sigmoid_diff(h_2) * x_1)
    w[2] -= n * (part_1 * w[4] * sigmoid_diff(h_1) * x_2 + part_2 * w[5] * sigmoid_diff(h_1) * x_2)
    w[3] -= n * (part_1 * w[6] * sigmoid_diff(h_2) * x_2 + part_2 * w[7] * sigmoid_diff(h_2) * x_2)
    w[4] -= n * (part_1 * out_h_1)
    w[5] -= n * (part_2 * out_h_1)
    w[6] -= n * (part_1 * out_h_2)
    w[7] -= n * (part_2 * out_h_2)
    b_1 -= n * (part_1 * w[4] * sigmoid_diff(h_1) + part_2 * w[5] * sigmoid_diff(h_1))
    b_2 -= n * (part_1 + part_2)

    counter += 1

plt.figure()
plt.plot(errors)
plt.title("Error")
plt.show()


