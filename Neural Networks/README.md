# Neural Network for solving XOR problem

## How Neural network works
Imagine we have dataset x = 5, y = 10 and we are looking for a linear function that fits to our dataset that f(x) = y, f(x) = a*x, so we are looking for an 'a' first we randomly set a = 3 than we substitude 'a' to our f(x) and result is 15, than we calculate error that is how wrong is our model, we calculate it, output - excpected_output, so 15 -10 is error = 5, now we know we need to adjust our 'a' that next time we calculate f(x) we get the lowest possible error, after we find best 'a' than we can use our model to calculate for any 'x' and we get 'y' based on our dataset and trained model.
Real NN works slightly different.

Imagine one neuron with two inputs, our output will be calculated as followed in the image below, it's like our linear function, we are looking for best Weights and Biases so if we substitute out inputs (dataset) to our model, our model must output same values as in dataset (or at least get closer to them). For tuning our Weights and Biases we use Backpropagation algorithm it uses partial derivatives with respect to our Weights and Biases to adjust than as needed to find minimum error for our model. We use activation function to make model non linear because if we just use linear adctivation function we get linear output, so we need to use non linear activation function to get desired output and second thing is partial derivative of linear function (ax + b) is just 'a'

![NN](https://github.com/adus-hash/Algorithms/assets/66412479/5f497606-cd01-4b40-ab8e-52f8a535af00)

Such network is very flexible and vey capable. We can use it to classify or predict. In our project we use it to solve XOR problem, as we can from graph below this problem can't be solved using linear function

![XOR-Problem-768x433](https://github.com/adus-hash/Algorithms/assets/66412479/9b6a880a-a3f2-4e04-a21d-d063e357a18c)
