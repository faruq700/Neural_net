import numpy as np

def correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    c = np.sum(x_mean * y_mean) / np.sqrt(np.sum(x_mean ** 2)) * np.sqrt(np.sum(y_mean ** 2) + 1e-8)
    return c

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class correlationNN():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden = []
        self.hidden_weight = []

        self.output_weight = np.random.randn(self.input_size, self.output_size) * 0.1

    def forward(self, X):
        self.hidden_output = []
        H = X
        output = 0
        for w in self.hidden_weight:
            h_out = sigmoid(np.dot(H, w))
            self.hidden_output.append(h_out)
            H = np.hstack((H, h_out))
            output = sigmoid(np.dot(H, self.output_weight))
        return output
        

    def train_output(self, X, y, lr = 0.01, epochs = 500):
        for _ in range(epochs):
            output = self.forward(X)
            error = y - output
            grad = error * sigmoid_derivative(output)

            H = X
            for h in self.hidden_output:
                H = np.hstack(H, h)
            self.output_weight += lr * np.dot(H.T, grad) / X.shape[0]

    
    def add_hidden_neuron(self, X, y, candidates=10, lr=0.1, epochs=200):
        residual = y - self.forward(X)
        best_corr = -1
        best_w = None
        for _ in range(candidates):
            w = np.random.randn(X.shape[1] , 1) * 0.1
            for _ in range(epochs):
                H = X
                for h in self.hidden_output:
                    if h.ndim == 1:
                        h = h.reshape(-1, 1)
                    H = np.hstack((H, h))
                h_out = sigmoid(np.dot(H, w))
                corr = abs(correlation(h_out.flatten(), residual.flatten()))
                if corr > best_corr:
                    best_corr = corr
                    best_w = w
            self.hidden_weight.append(best_w)


if __name__ == "__main__":
    X = np.array([[1,1], [1,0], [0,1], [0,0]])
    Y = np.array([[0], [1], [1], [0]])

    model = correlationNN(2, 1)
    model.train_output(X, Y, epochs = 200)

    for _ in range(2):
        model.add_hidden_neuron(X, Y)
        model.train_output(X, Y, epochs=200)

    pred = model.forward(X)
    print(pred)