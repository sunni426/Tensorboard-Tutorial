# 1) Design model
# 2) Compute loss & gradient
# 3) Training loop: 
#   forward pass (compute prediction & loss) 
#   backward pass (gradients) 
#   update weights

import torch
import numpy as np
import torch.nn as nn
from sklearn import datasets # binary classification dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # separation of test & training data


# data preparation
bc = datasets.load_breast_cancer() # a binary classification problem
X, y = bc.data, bc.target

n_samples, n_features = X.shape
# print(n_samples, n_features)

# split data:
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1234)

# scale our features
sc = StandardScaler() # for our data to have 0 mean & unit variance, always recommended for logistic regression
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# after scaling, convert to tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# reshape our y tensors
y_train = y_train.view(y_train.shape[0],1) # into column vector
y_test = y_test.view(y_test.shape[0],1)

# 1) model: linear combo of weights & bias. w/ linear regression, add a sigmoid function
# f = wx+b, sigmoid at end
class LogisticRegression(nn.Module): # derived from nn.Module

    def __init__(self, n_input_features):
        super(LogisticRegression,self).__init__()
        self.Linear = nn.Linear(n_input_features,1) # we just want 1 class label in the end

    def forward(self,x):
        y_predicted = torch.sigmoid(self.Linear(x))
        return y_predicted
    
model = LogisticRegression(n_features)


# 2) loss & gradient
learning_rate = 0.01
loss = nn.BCELoss() # binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

# 3) training loop
n_epoch = 100
for epoch in range(n_epoch):
    # forward pass and loss
    y_predicted = model(X_train)
    l = loss(y_predicted, y_train)

    # backward pass
    l.backward()

    # update weights
    optimizer.step()

    # empty gradients
    optimizer.zero_grad()

    if (epoch+1)%10 ==0:
        print(f'epoch: {epoch+1}, loss = {l.item():.4f}')

# evaluate model: should not be part of our computation graph
with torch.no_grad():
    y_predicted = model(X_test)
    # classify to class label, 0 or 1
    y_predicted_class = y_predicted.round()
    accuracy = y_predicted_class.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {accuracy:.4f}')