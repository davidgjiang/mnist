import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from mnist_tools import *

def main():
    print('MNIST Training')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transformation)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transformation)

    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    while True:
        try:
            batch_size = int(input("Enter batch size: "))
            if batch_size <= 0:
                raise ValueError("Batch size must be a positive integer.")
            break 
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a positive integer.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = LeNet()
    model.to(device)

    while True:
        try:
            epochs = int(input("Enter number of epochs: "))
            if epochs <= 0:
                raise ValueError("Number of epochs must be a positive integer.")
            break 
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a positive integer.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    avg_train_losses = []
    avg_val_losses = []

    print('Initialising training...')
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        avg_tloss, avg_vloss = train(model, train_loader, val_loader, criterion, optimizer, device)
        avg_train_losses.append(avg_tloss)
        avg_val_losses.append(avg_vloss)
    print('Training complete!')

    accuracy = test(model, test_loader, device)
    print('Accuracy of model on test set: {:.4f}%'.format(accuracy))

    save = input('Do you want to save the model? (y/n): ')
    if save == 'y':
        torch.save(model.state_dict(), 'mnist_model.pth')
        print('Model saved as mnist_model.pth')

if __name__ == '__main__':
    main()
