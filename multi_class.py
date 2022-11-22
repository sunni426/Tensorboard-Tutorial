import torch
import torch.nn as nn

# multiclass problem
class NeuralNet2(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet2,self).__init()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.reLU()
        self.linear2 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
loss = nn.CrossEntropyLoss() # (applies Softmax)