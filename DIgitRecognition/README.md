# DIgitRecognition

## Text Recognition using Neural Networks in C++

### Introduction

The objective of this project is to develop a neural network-based 
model for text recognition using the C++ programming language without using any library. The 
project focuses on recognizing handwritten digits from the MNIST 
dataset, employing a neural network architecture with two layers. The 
neural network is trained using the backpropagation algorithm.
Implementation Details

### Neural Network Architecture
The neural network consists of one layers - a hidden layer with 16 
neurons and an output layer with 10 neurons. The activation function 
for the hidden layer is the Leaky ReLU, and the output layer employs 
the softmax activation function for multiclass classification. The weights 
of the neural network are initialized with random values.

### Training Process
The training process involves reading the MNIST training dataset, which 
includes images of handwritten digits and their corresponding labels. 
The backpropagation algorithm is utilized to update the weights of the 
neural network iteratively. The Leaky ReLU activation function is 
employed for the hidden layer, and the softmax function is used for the 
output layer to calculate the loss. The stochastic gradient descent (SGD) 
optimization algorithm is utilized for weight updates.

### Forward Propagation:
Input Pass: The process begins by feeding the input data into the neural 
network through the input layer.
Weighted Sum and Activation: In each neuron of the hidden layers, the 
input values are multiplied by weights, and the results are summed. 
This sum is then passed through an activation function, introducing 
non-linearity to the model.
Output Prediction: The computed values are propagated through 
subsequent layers until the final layer (output layer) produces 
predictions or classifications.
Loss Computation: The predictions are compared with the actual target 
values, and a loss (error) is calculated to quantify the disparity.
Loss Function is -summation of Ai*LogOi    (where Ai is actual output and Oi is predicted output)

### Backpropagation
Gradient Descent Initialization: The optimization process starts by 
initializing the weights and biases. Typically, random values are 
assigned initially.
Backward Pass (Backpropagation): The gradient of the loss with respect 
to the weights and biases is calculated by applying the chain rule of 
calculus backward through the network.
Weight Update: The weights and biases are adjusted in the opposite 
direction of the gradient, aiming to minimize the loss. This is typically 
done using an optimization algorithm like gradient descent.
Iterative Process: Steps 2 and 3 are repeated iteratively over the entire 
training dataset until the model converges to a state where the loss is 
minimized.
The backpropagation algorithm is employed to compute the gradients 
of the loss with respect to the weights. The gradients are then used to 
update the weights in the direction that minimizes the loss. The Leaky 
ReLU derivative is utilized for the hidden layer, and the softmax 
derivative is used for the output layer. The learning rate is set to 0.1, 
and a batch size of 300 is used during training.

## Output 
![Screenshot 2023-12-17 023018](https://github.com/Anish2915/DIgitRecognition/assets/137883198/97d65915-96dc-488f-8581-da2a9723476b)



