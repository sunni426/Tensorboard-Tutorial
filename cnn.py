import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0,1]
# we transform them to tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

train_dataset = torchvision.datasets.CIFAR10(root="./data",train=True,
                                    download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data',train=False,
                                    download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False)

classes = ('plane','car','bird','cat',
            'deer','dog','frog','horse','ship','truck')



# implement conv net
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        # input channel size: 3, because our image set has 3 color channels.
        # output channel size: 6
        # kernel size is 5 (5 x 5)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2) # kernel size of 2 (2 x 2), stride of 2 (after each operation, shift by 2)
        self.conv2 = nn.Conv2d(6, 16, 5) # input channel must equal output of last output channel size
        self.fc1 = nn.Linear(16*5*5,120) # fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # last fcl, output size must be 10, the # of class labels
        # for these fully connected layers, the FIRST INPUT and LAST OUTPUT matter 
        # first input is 16*5*5 --> resulting 3D dimensions after feature learning, before classification part (linear portion)
        # last output is 10 -> # of layers

    def forward(self,x):
        # first convolution & pooling layer
        x = self.pool(F.relu(self.conv1(x))) # --> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x))) # --> n, 16, 5, 5

        # now, pass to 1st fully connected layer
        # first, flatten
        x = x.view(-1, 16*5*5) # first param (-1) : number of batches. --> n, 400
        x = F.relu(self.fc1(x)) # also apply activation function!! --> n, 120
        x = F.relu(self.fc2(x)) # --> n, 84
        x = self.fc3(x) # no activation function in the end --> n, 10
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs,labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label==pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')