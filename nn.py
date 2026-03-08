import numpy as np


class Network():
    def __init__(self, input=1, output=1, **kwargs):
        hidden_layers = list(kwargs.values())
        self.input = input
        self.output = output
        self.weights = []
        self.biases = []
        layer_sizes = [input] + hidden_layers + [output]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.rand(layer_sizes[i], layer_sizes[i + 1]) *  np.sqrt(2/layer_sizes[i]))
            self.biases.append(np.random.rand(layer_sizes[i + 1] ) *  np.sqrt(2/layer_sizes[i]))
            
    def relu_activation(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def MSE_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def binary_cross_entropy_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def _backward_pass(self, y_true, y_pred, x_data):
        """Compute gradients using backpropagation."""
        # Compute output layer error (derivative of MSE loss)
        output_error = 2 * (y_pred - y_true) / len(x_data)
        
        # Backpropagate through layers
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        delta = output_error
        
        # Backward pass through each layer
        for i in reversed(range(len(self.weights))):
            # Compute gradients
            if i == 0:
                weight_gradients[i] = np.dot(x_data.T, delta)
            else:
                weight_gradients[i] = np.dot(self.p_array[i - 1].T, delta)
            
            bias_gradients[i] = np.sum(delta, axis=0)
            
            # Backpropagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta = delta * self.relu_derivative(self.p_array[i - 1])
        
        return weight_gradients, bias_gradients

    def gradient_descent(self, y_true, y_pred, x_data, learning_rate):
        """Update weights and biases using gradient descent."""
        weight_gradients, bias_gradients = self._backward_pass(y_true, y_pred, x_data)
        
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]

    def stochastic_gradient_descent(self, x_data, y_true, learning_rate):
        """Update weights and biases using stochastic gradient descent (one sample at a time)."""
        for i in range(len(x_data)):
            sample_x = x_data[i:i+1]
            sample_y = y_true[i:i+1]
            
            pred = self.forward(sample_x)
            self.gradient_descent(sample_y, pred, sample_x, learning_rate)

    def batch_gradient_descent(self, x_data, y_true, learning_rate, batch_size):
        """Update weights and biases using batch gradient descent."""
        for i in range(0, len(x_data), batch_size):
            batch_x = x_data[i:i + batch_size]
            batch_y = y_true[i:i + batch_size]
            
            pred = self.forward(batch_x)
            self.gradient_descent(batch_y, pred, batch_x, learning_rate)

    def Adam_optimizer(self, y_true, y_pred, x_data, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Update weights and biases using Adam optimizer."""
        if not hasattr(self, 't'):
            # Initialize timestep and moment estimates on first call
            self.t = 0
            self.m_weights = [np.zeros_like(w) for w in self.weights]
            self.v_weights = [np.zeros_like(w) for w in self.weights]
            self.m_biases = [np.zeros_like(b) for b in self.biases]
            self.v_biases = [np.zeros_like(b) for b in self.biases]
        
        self.t += 1
        
        # Compute gradients
        weight_gradients, bias_gradients = self._backward_pass(y_true, y_pred, x_data)
        
        # Update parameters using Adam
        for i in range(len(self.weights)):
            # Update biased first moment estimate
            self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * weight_gradients[i]
            self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * bias_gradients[i]
            
            # Update biased second raw moment estimate
            self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * (weight_gradients[i] ** 2)
            self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * (bias_gradients[i] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_weights_hat = self.m_weights[i] / (1 - beta1 ** self.t)
            m_biases_hat = self.m_biases[i] / (1 - beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_weights_hat = self.v_weights[i] / (1 - beta2 ** self.t)
            v_biases_hat = self.v_biases[i] / (1 - beta2 ** self.t)
            
            # Update parameters
            self.weights[i] -= learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + epsilon)
            self.biases[i] -= learning_rate * m_biases_hat / (np.sqrt(v_biases_hat) + epsilon)

    def _apply_activation(self, x, activation):
        """Apply activation function to input."""
        if activation == "relu":
            return self.relu_activation(x)
        elif activation == "relu_d":
            return self.relu_derivative(x)
        else:
            return x

    def forward(self, x, activation='relu'):
        """Forward propagation through all layers."""
        self.p_array = []
        output = x
        
        # Propagate through all layers
        for i, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            output = np.dot(output, weights) + biases
            
            # Apply activation function to all layers except the last one
            if i < len(self.weights) - 1:
                output = self._apply_activation(output, activation)
            
            self.p_array.append(output)
        
        return self.p_array[-1]  # Return the final output


if __name__ == "__main__":
    #print("Creating a network with 3 input neurons, 2 output neurons, and hidden layers of sizes 10, 10, and 5.")
    net = Network(2, 1, hidden1=5)

    # dataset
    x = np.array([
        np.array([1,0]),
        np.array([0,1]),
        np.array([0,0]),
        np.array([1,1])
        ])
    y = np.array([[1], [1], [1], [0]])

    # train the model
    for epoch in range(1000):
        pred_y = net.forward(x)
        #loss = net.MSE_loss(y, pred_y)
        loss = net.binary_cross_entropy_loss(y, pred_y)
        #net.gradient_descent(y, pred_y, x_data=x, learning_rate=0.05)
        net.Adam_optimizer(y, pred_y, x_data=x, learning_rate=0.05)

    # test the model
    pred_y = net.forward(x, activation="relu_d")
    print(pred_y)


    """
    print("Weights:")
    for w in net.weights:
        print(w)
    print("\nBiases:")
    for b in net.biases:
        print(b)"""