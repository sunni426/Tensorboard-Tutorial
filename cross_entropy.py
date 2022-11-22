import torch
import torch.nn as nn
import numpy as np

def cross_entropy(actual,predicted):
    loss = -np.sum(actual*np.log(predicted)) # neg log likelihood loss
    return loss # / float(predicted.shape[0])

# y must be one-hot encoded
Y = np.array([1,0,0])

# y_pred has probabilities
Y_pred_good = np.array([0.7,0.2,0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])

l1 = cross_entropy(Y,Y_pred_good)
l2 = cross_entropy(Y,Y_pred_bad)

print(f'Loss 1 numpy: {l1:.4f}')
print(f'Loss 2 numpy: {l2:.4f}')



# implementing in pytorch:
loss = nn.CrossEntropyLoss()

# Y = torch.tensor([0]) # only put class labels
Y = torch.tensor([2,0,1])# multiple samples

# nsamples x classes = 1x3, an array
Y_pred_good = torch.tensor([[0.1,1.0,2.1],[2.0,1.0,0.1],[2.0,3.0,0.1]])
Y_pred_bad = torch.tensor([[2.5,2.0,0.3],[0.1,2.0,0.3],[0.5,2.0,0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

__, predictions1 = torch.max(Y_pred_good,1)
__, predictions2 = torch.max(Y_pred_bad,1)

print(predictions1)
print(predictions2)