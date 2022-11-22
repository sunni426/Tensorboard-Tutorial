import torch 

X = torch.tensor([1,2,3,4],dtype=torch.float32)
Y = torch.tensor([2,4,6,8],dtype=torch.float32)
w = torch.tensor(0.0,dtype=torch.float32, requires_grad = True)

# f = w*x

# f = 2*x

# model prediction: forward pass. define method
def forward(x):
    return w*x

# loss: MSE (characteristic of linear regression)
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# start training
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # prediction: forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y,y_pred)

    # gradients
    l.backward() # automatic calculation of dl/dw

    # update weights: don't involve this in gradient tracking, so wrap expression!
    with torch.no_grad():
        w -= learning_rate*w.grad # update in the negative direction of gradient!
    
    # zero out gradients: with underscore before (), in place operation
    w.grad.zero_()

    if epoch%1==0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction before training: f(5) = {forward(5):.3f}')
