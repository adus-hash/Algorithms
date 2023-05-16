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


w_ih = np.random.uniform(size=(25, len(inp[0][0]))) #np.random.normal(0.0, pow(2, -0.5), (2, len(inp[0][0])))  # #np.random.randn(3, len(inp[0][0]))
# w_ih = np.array([[1.5, 1.5],
#                 [.5, .5],])

bias_hidden = np.random.uniform(size=(1, len(w_ih))) # np.zeros((1, len(w_ih))) #
# bias_hidden = np.array([[1, 1.0]])

w_ho = np.random.uniform(size=(1, len(w_ih))) # np.random.randn(1, len(w_ih))
# w_ho = np.array([[2, 1.5]])

bias_output = np.random.uniform(size=(1, len(w_ho))) # np.zeros((1, len(w_ho))) #
# bias_output = np.array([[1.0]])



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

    #print(bias_hidden)
    bias_hidden += r * np.array(hidden_gradient).T




test = np.array([ [[0, 0]],
                  [[0, 1]],
                  [[1, 0]],
                  [[1, 1]] ])
for i in test:
    h1_test = sigmoid(np.dot(w_ih, np.array(i).T) + bias_hidden.T)      # testing neural newtwork
    output_test = sigmoid(np.dot(w_ho, h1_test) + bias_output)

    print(output_test, i)

#print()
#print(w_ih, bias_hidden)
#print(w_ho, bias_output)
