import numpy as np
from torch.nn import Sigmoid
from nn import Network

class New_Network():
    def __init__(self, input, output, **kwargs):
        self.input_output_weight = np.random.rand(input, output)
        print(self.input_output_weight)
        self.hidden_layers = list(kwargs.values())
        self.nn = Network()

    def forward(self, x):
        # calculate the input to output forward
        x = np.dot(x, self.input_output_weight)
        x = self.nn.sigmoid(x)
        return x



if __name__ == "__main__":
    net = New_Network(2, 1, hidden1=5)
   # data = np.array([[1,0], [0,1], [1,1], [0,0]])
    data = np.array([
        np.array([1,0]).reshape(1,2),
        np.array([0,1]).reshape(1,2),
        np.array([0,0]).reshape(1,2),
        np.array([1,1]).reshape(1,2)
        ])
    label = np.array([[1], [1], [1], [0]])
    pred = []
    for i in data:
        #print(i.shape)
        y = net.forward(i)
        y = (y >= 0.5).astype(int)
        pred.append(y)


    print(list(pred))