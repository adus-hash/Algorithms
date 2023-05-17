# Neural Network for solving XOR problem

## How Neural network works
Imagine we have dataset x = 5, y = 10 and we are looking for a linear function that fits to our dataset that f(x) = y, f(x) = a*x, so we are looking for an 'a' first we randomly set a = 3 than we substitude 'a' to our f(x) and result is 15, than we calculate error that is how wrong is our model, we calculate it, output - excpected_output, so 15 -10 is error = 5, now we know we need to adjust our 'a' that next time we calculate f(x) we get the lowest possible error, after we find best 'a' than we can use our model to calculate for any 'x' and we get 'y' based on our dataset and trained model.
Real NN works slightly different.

Imagine one neuron with two inputs, our output will be calculated as followed in the image below, it's like our linear function, we are looking for best Weights and Biases so if we substitute out inputs (dataset) to our model, our model must output same values as in dataset (or at least get closer to them). For tuning our Weights and Biases we use Backpropagation algorithm it uses partial derivatives with respect to our Weights and Biases to adjust than as needed to find minimum error for our model. We use activation function to make model non linear because if we just use linear adctivation function we get linear output, so we need to use non linear activation function to get desired output and second thing is partial derivative of linear function (ax + b) is just 'a'.

![NN](https://github.com/adus-hash/Algorithms/assets/66412479/5f497606-cd01-4b40-ab8e-52f8a535af00)

Such network is very flexible and vey capable. We can use it to classify or predict. In our project we use it to solve XOR problem, as we can see from graph below this problem can't be solved using linear function, so if we substitute (0, 0) or (1, 1), our model should output 0 and for (0, 1) or (1, 0), model should output 1.

![XOR-Problem-768x433](https://github.com/adus-hash/Algorithms/assets/66412479/9b6a880a-a3f2-4e04-a21d-d063e357a18c)

## Code

We start by importing random and numpy as we will use random to randomly initialize our weights and biases and numpy to work effectively with arrays. Then we define our activation function in our case it's sigmoid (for any x values it returns values between 0 and 1) and sigmoid derivative for backpropagation.

```Python
import random
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(y):
    return y * (1 - y)
```

Here we define our dataset and randomly initialize our weights and biases for hidden layer and for output layer, 'r' in our case is learning rate (how quickly should our model learn, bigger number doesn't mean better, faster learning) and it's value is 0.12.
```Python
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
```

Our model will train for 30 000 cycles, cycle means calculate output, calculate error of model than based on error tune weights and biases and this cycle repeats for 30 000 times. First we randomly choose a sample from dataset. We use np.dot() therefore matrice multiplication to calculate output for hidden layer and this output will be input for output layer.
```Python
for i in range(30000):
    index_sample = random.randint(0, len(inp) - 1)

    # Forward propagation
    hidden = sigmoid(np.dot(w_ih, inp[index_sample].T) + bias_hidden.T)
    output = sigmoid(np.dot(w_ho, hidden) + bias_output)
```

Here we calculate error for output and hidden layer. Gradient is simply which direction we sould tune our weights and is calculated as output_error times the sigmoid_derivative(output), after that we start tuning weights and biases by multipling learning rate and output_gradient and adding it to weight and biases, repeat the same steps for hidden layer to tune hidden layers weights and biases. That's all for training our model.
```Python
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
```

Here we test our model simply by substituting dataset to it. If our model is well trained, it should outputs the same values sa in des_output variable.
```Python
test = np.array([ [[0, 0]],
                  [[0, 1]],
                  [[1, 0]],
                  [[1, 1]] ])
for i in test:
    h1_test = sigmoid(np.dot(w_ih, np.array(i).T) + bias_hidden.T)      # testing neural newtwork
    output_test = sigmoid(np.dot(w_ho, h1_test) + bias_output)

    print(output_test, i)
```
