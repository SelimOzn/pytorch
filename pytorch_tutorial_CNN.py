import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 4
batch_size = 4
learning_rate = 0.001

transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root="./data", download=True, train=True, transform=transformer)

test_dataset = torchvision.datasets.CIFAR10(root="./data", download=True, train=False, transform=transformer)

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

dataiter = iter(train_loader)
images, labels = next(dataiter)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)

        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print("Finished training.")
"""PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)"""

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_correct_class = [0 for i in range(10)]
    n_samples_class = [0 for i in range(10)]

    for (images, labels) in test_loader:

        images = images.to(device)
        labels = labels.to(device)
        prediction = model(images)
        _, prediction_label = torch.max(prediction, 1)

        n_samples += labels.size(0)
        n_correct += (prediction_label == labels).sum().item()

        for i in range(batch_size):
            if prediction_label[i] == labels[i]:
                n_correct_class[labels[i]] += 1

            n_samples_class[labels[i]] += 1

    acc = (n_correct / n_samples) * 100
    print(f'Accuracy of the network: {acc} %')

for i in range(10):
    acc = (n_correct_class[i] / n_samples_class[i]) * 100
    print(f'Accuracy of {classes[i]}: {acc} %')












