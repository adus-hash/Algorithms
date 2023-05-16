import random
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(y):
    return y * (1 - y)


inp = np.array([[[0, 0]],
                [[0, 1]],
                [[1, 0]],
                [[1, 1]]])


des_output = [[0, 1, 1, 0]]


w_ih = np.random.uniform(size=(25, len(inp[0][0])))

bias_hidden = np.random.uniform(size=(1, len(w_ih)))

w_ho = np.random.uniform(size=(1, len(w_ih)))

bias_output = np.random.uniform(size=(1, len(w_ho)))



r = -0.12

for i in range(30000):
    index_sample = random.randint(0, len(inp) - 1)

    # Forward propagation
    hidden = sigmoid(np.dot(w_ih, inp[index_sample].T) + bias_hidden.T)
    output = sigmoid(np.dot(w_ho, hidden) + bias_output)

    # Calculating error
    output_error = output - np.array(des_output).T[index_sample]
    hidden_error = np.dot(w_ho.T, output_error)

    # Back propagation
    output_gradient = output_error * sigmoid_derivative(output)
    w_ho += r * np.dot(output_gradient, hidden.T)

    bias_output += r * output_gradient

    hidden_gradient = hidden_error * sigmoid_derivative(hidden)
    w_ih += r * np.dot(hidden_gradient, inp[index_sample])

    bias_hidden += r * np.array(hidden_gradient).T




test = np.array([ [[0, 0]],
                  [[0, 1]],
                  [[1, 0]],
                  [[1, 1]] ])
for i in test:
    h1_test = sigmoid(np.dot(w_ih, np.array(i).T) + bias_hidden.T)      # testing neural newtwork
    output_test = sigmoid(np.dot(w_ho, h1_test) + bias_output)

    print(output_test, i)