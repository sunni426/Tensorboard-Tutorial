# Pytorch: Linear Regression Model
# training pipeline:

# 1) Design model (input, output size, forward pass)
# 2) Construct loss and otimizer
# 3) Training loop
#     - forward pass: compute prediction & loss 
#     - backward pass: gradients
#     - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets # binary classification dataset
import matplotlib.pyplot as plt

# 0) prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)

# convert to tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# reshape y. now is all in 1 row, we want column vector --> put each  value in one row
y = y.view(y.shape[0],1)

n_samples, n_features = X.shape


# 1) Design model
# in linear regression case, is just one layer
input_size = n_features
output_size = 1
model = nn.Linear(input_size,output_size)

# 2) Construct loss & optimizer
criterion = nn.MSELoss() # built in loss function from pytorch.
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate) # arguments: parameters to optimize

# 3) Training loop
n_epochs = 100
for epoch in range(n_epochs):

    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted,y) # takes in actual labels & predicted values
    
    # backward pass
    loss.backward()

    # update
    optimizer.step() # update the weights

    # gradient zeroing (empty the .grad attribute)
    optimizer.zero_grad()

    if((epoch+1)%10 ==0 ):
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# plot
predicted = model(X).detach() # detach from tensor to prevent tracking in computational graph
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted,'b')
plt.show()