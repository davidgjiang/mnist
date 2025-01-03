import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)                               # [1,28,28] -> [6,28,28]
        x = F.sigmoid(x)                             
        x = F.max_pool2d(x, kernel_size=2, stride=2)    # [6,28,28] -> [6,14,14]
        x = self.conv2(x)                               # [6,14,14] -> [16,10,10]
        x = F.sigmoid(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)    # [16,10,10] -> [16,5,5]
        x = torch.flatten(x, start_dim=1)               # [16,5,5] -> [400]
        x = self.fc1(x)                                 # [400] -> [120]
        x = F.sigmoid(x)
        x = self.fc2(x)                                 # [120] -> [84]
        x = F.sigmoid(x)
        x = self.fc3(x)                                 # [84] -> [10]
        return x

def train(model, train_loader, val_loader, criterion, optimizer, device):
    running_loss_train = 0.0
    running_loss_val = 0.0

    model.train()
    for batch_i, (train_inputs, train_labels) in enumerate(train_loader):
        train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
        optimizer.zero_grad() 
        train_outputs = model(train_inputs)                    # forward pass
        train_loss = criterion(train_outputs, train_labels)    # compute loss and gradients
        train_loss.backward()                                  # back propagation
        optimizer.step()                                       # update weights
        running_loss_train += train_loss.item()
    
    model.eval()
    with torch.no_grad(): # disables gradient calculation
        for batch_j, (val_inputs, val_labels) in enumerate(val_loader):
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
            running_loss_val += val_loss.item()

    avg_loss_train = running_loss_train / (batch_i+1)
    avg_loss_val = running_loss_val / (batch_j+1)
    print('    LOSS train {} valid {}'.format(avg_loss_train, avg_loss_val))
    
    return avg_loss_train, avg_loss_val

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
    
    accuracy = correct / total
    print('Accuracy on test set: {:.4f}'.format(accuracy))
    
    return accuracy