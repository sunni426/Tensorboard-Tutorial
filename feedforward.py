# MNIST
# DataLoader, Transformations
# Multilayer Neural Net, activation function
# Loss and Optimizer
# Training Loop (batch training)
# Model Evaluation
# GPU Support

import torch
import torch.nn as nn
import numpy as np
import torchvision # for datasets
import torchvision.transforms as transforms
import sys
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import tensorboard_tutorial as tb
writer = SummaryWriter('runs/mnist')

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 784 # 28x28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# import MNIST
train_dataset = torchvision.datasets.MNIST(root='./data',train=True,
    transform=transforms.ToTensor(),download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',train=False,
    transform=transforms.ToTensor(),download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    batch_size=batch_size, shuffle=True) # shuffle True: good for training

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    batch_size=batch_size, shuffle=False)

# let's look at one batch
examples = iter(train_loader) # an iter object
example_data, example_targets = examples.next()
samples, labels = examples.next()
print(samples.shape, labels.shape)
# will print out torch.Size([100, 1, 28, 28]) torch.Size([100]) 
# sample.shape: 100 is batch_size, only 1 channel (color channel)
# labels.shape: have have 100 --> for each class label we have 1 value

# plot
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.show()

# add images to tensorboard: create a grid, then call image method
# img_grid = torchvision.utils.make_grid(example_data)
# writer.add_image('mnist_images',img_grid) # first arg: label
# writer.close() # makes sure output is flushed
# sys.exit()


# now, we have our data loaded. so we want to set up a fully connected 
# neural layer with 1 hidden layer

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes): # num_classes = output size
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self,x): # now, apply all the layers above
        out = self.l1(x);
        out = self.relu(out)
        out = self.l2(out)
        # don't apply softmax here bc will be using cross entropy loss later
        return out
model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


for n_iter in range(50):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('New/train', np.random.random(), n_iter)
writer.close()

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    # loop over all batches
    for i, (images, labels) in enumerate(train_loader): # enumerator function gives us actual index, and then the data (images, labels)
        # 100x1x28x28
        # input size is 784, so 100, 784
        # reshape our tensor
        images = images.reshape(-1,28*28).to(device) # push to gpu if available
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # backwards
        optimizer.zero_grad()
        loss.backward() # back propagation
        optimizer.step()


        # if(i+1)%10==0:
        #     avg_loss = running_loss/10
        #     # print(f'avg loss: {avg_loss:.3f}')
        #     print(f'x val: {epoch+(i/len(train_loader))}')
        #     writer.add_scalar('loss',avg_loss,epoch+(i/len(train_loader))) # maybe: 2nd param: step = i/len(train_loader) ?
        #     running_loss = 0.0

        if (i+1)%100==0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
            

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        # need to reshape again
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index (class label -> what we're interested in)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0] # number of samples in current batch, should be 100
        n_correct += (predictions==labels).sum().item()
    
    accuracy = 100.0*n_correct/n_samples

    print(f'accuracy = {accuracy}')

