import torch 
import torch.nn as nn # contains neural network functions

X = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)
X_test = torch.tensor([5],dtype=torch.float32)

n_samples, n_features = X.shape

# model prediction: forward pass. define method
inputsize = n_features
outputsize = n_features

class LinearRegression(nn.Module):

    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        # define our layers
        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self,x):
        return self.lin(x)

# model = nn.Linear(inputsize,outputsize) # this is only one layer, lets say we want to implement a custom model:

model = LinearRegression(inputsize, outputsize)

# start training
learning_rate = 0.05
n_iters = 10

# loss: MSE (characteristic of linear regression)
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')


for epoch in range(n_iters):
    # prediction: forward pass
    y_pred = model(X)

    # loss
    l = loss(Y,y_pred)

    # gradients
    l.backward() # automatic calculation of dl/dw

    optimizer.step()
    optimizer.zero_grad()

    if epoch%1==0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')
