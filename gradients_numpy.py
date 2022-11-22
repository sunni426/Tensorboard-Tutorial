import numpy as np

# f = w*x

# f = 2*x
X = np.array([1,2,3,4],dtype=np.float32)
Y = np.array([2,4,6,8],dtype=np.float32)

w = 0.0

# model prediction: forward pass. define method
def forward(x):
    return w*x

# loss: MSE (characteristic of linear regression)
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()


# manually calcuate gradient
# J = MSE = 1/N*(w*x-y)**2
# gradient w.r.t weight (parameter): dJ/dw = 1/N*2*x*(w*x-y)
def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# start training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    # prediction: forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y,y_pred)

    # gradients
    dw = gradient(X,Y,y_pred)

    # update weights
    w -= learning_rate*dw # update in the negative direction of gradient!
    
    if epoch%1==0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

    print(f'Prediction before training: f(5) = {forward(5):.3f}')