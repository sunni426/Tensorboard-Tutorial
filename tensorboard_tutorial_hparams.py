# this: visualizing hyperparameters in tensorboard

from prometheus_client import Summary
import torch
import torchvision
import torch.nn as nn # ALL neural network modules, nn.Linear, nn.Conv2d, BatchNorm, loss functions
import torch.optim as optim # for all optimization algorithms, SGD, Adam, etc
import torch.nn.functional as F # all functions that don't have any parameters
import torchvision.datasets as datasets # has standard datasets we can import in a nice way
import torchvision.transforms as transforms # transformations we can perform on our dataset
from torch.utils.data import DataLoader # gives easier dataset management & creates mini batches
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard

# simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16*7*7, num_classes)
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x
    
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
# learning_rate = 0.001
in_channels = 1
num_classes = 10
# batch_size = 64
num_epochs = 1 #5

# load data
train_dataset = datasets.MNIST(root='dataset/', train=True,
                            transform=transforms.ToTensor(), download=True)
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# initialize network
# model = CNN(in_channels=in_channels, num_classes=num_classes)
# model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(f'runs/MNIST/tryingout_tensorboard') # go to runs folder, specify where we will write all datafiels that tensorboard will use.

# hyperparameter search --> fit
batch_sizes = [2, 64, 1024]
learning_rates = [0.1, 0.01, 0.001, 0.0001] # better method: do in log space and randomize

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        step = 0
        # initialize network
        model = CNN(in_channels=in_channels, num_classes=num_classes) # to reset parameters for model
        model.to(device)
        model.train() # not sure about this
        writer = SummaryWriter(f'runs/MNIST/MiniBatchSize {batch_size} LR {learning_rate}') # because we're plotting all 3 of these on the same plot
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)

        for epoch in range(num_epochs):
            losses = []
            accuracies = []

            for batch_idx, (data, targets) in enumerate(train_loader):
                # get data to cuda if possible
                data = data.to(device=device)
                targets = targets.to(device=device)

                # forward
                scores = model(data)
                loss = criterion(scores, targets)
                losses.append(loss.item())

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # calculate 'running' training accuracy
                _, predictions = scores.max(1)
                num_correct = (predictions == targets).sum()
                running_train_acc = float(num_correct)/float(data.shape[0])
                accuracies.append(running_train_acc)

                # for every batch, write to tensorboard
                writer.add_scalar('Training loss', loss, global_step=step)
                writer.add_scalar('Training Accuracy', running_train_acc, global_step=step)
                step +=1 # added 1 data point --> move one step
                # perhaps for better visualization of hyperparameters:
            writer.add_hparams({'lr': learning_rate, 'bsize': batch_size},
                                {'accuracy': sum(accuracies)/len(accuracies),
                                'loss': sum(losses)/len(losses)})
                
                

        print(f'Mean Loss this epoch was {sum(losses)/len(losses)}')