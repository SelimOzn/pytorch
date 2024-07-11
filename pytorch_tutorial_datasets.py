#batch_size = numbr of training samples in one forward & backward pass
#number of iterations = number of passes, each pass using [batch_size] #of samples
#e.g. 100 samples, batch_size=20 --> 100/20 = 5 iterations for 1 epoch

import  torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self):
        #data loading
        xy = np.loadtxt("wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        #dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        #len(dataset)
        return self.n_samples


dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(features, labels)

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

dataiter = iter(dataloader)
data = next(dataiter)
features, labels = data
print(features, labels)

#training loop
epochs = 2
n_samples = len(dataset)
n_iterations = math.ceil(n_samples/4)
print(n_samples, n_iterations)

for epoch in range(epochs):
    for i, (features, labels) in enumerate(dataloader):
        if (i+1) % 5 == 0:
            print(f"epoch {epoch+1}/{epochs}, step {i+1}/{n_iterations}, inputs {features.shape}")


"""torchvision.datasets.MNIST()
torchvision.datasets.cifar()
torchvision.datasets.coco()"""


#Transformlar

class WineDataset_T(Dataset):

    def __init__(self, transform=None):
        #data loading
        xy = np.loadtxt("wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = (xy[:, 1:])
        self.y = (xy[:, [0]])
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        #dataset[0]
        samples = self.x[index], self.y[index]

        if self.transform:
            samples = self.transform(samples)

        return samples
    def __len__(self):
        #len(dataset)
        return self.n_samples




#Custom transformers
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

dataset = WineDataset_T(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features)

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset_T(transform=composed)

first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features)
