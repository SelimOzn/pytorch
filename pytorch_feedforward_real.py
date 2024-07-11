import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

writer = SummaryWriter("runs/mnist2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

#hyperparamater define
input_size = 784 #Images are 28x28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.01

#MNIST load

training_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                              transform=transforms.ToTensor(),
                                              download=True)

test_dataset = torchvision.datasets.MNIST(root="./data", train=False,
                                          transform=transforms.ToTensor())

train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                         shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape) #Samples.shape = 100, 1, 28, 28
assert(torch.Size([100, ]) == labels.shape) #True

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap="gray")
#plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image("mnist_iamges", img_grid)
writer.close()
#sys.exit()
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, samples.view(-1, 28*28).to(device))
writer.close()
#sys.exit()
#Training loop
n_total_steps = len(train_loader)

running_loss = 0.0
running_correct = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)
        #forward
        predictions = model(images)
        a = predictions.get_device()
        loss = criterion(predictions, labels)
        #backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss+= loss.item()

        _, predicted = torch.max(predictions.data, 1)
        running_correct += (predicted == labels).sum().item()

        if (i+1) % 100 == 0:
            print(f"Epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}")
            writer.add_scalar("training loss", running_loss/100, epoch * n_total_steps + i)
            writer.add_scalar("accuracy", running_correct/100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct = 0

class_labels = []
class_preds = []
# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        values, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        class_probs_batch = [F.softmax(output, dim=0) for output in outputs]
        deneme = F.softmax(outputs, dim=0)[:, 9]
        class_preds.append(class_probs_batch)
        class_labels.append(labels)

    # 10000, 10, and 10000, 1
    # stack concatenates tensors along a new dimension
    # cat concatenates tensors in the given dimension
    deneme222 = [torch.cat(batch, axis=-1) for batch in class_preds] #torch.cat sadece var olan dim'de ekleme yaparken 100 tane 10 scalar tensor içeren bir listeyi alıp
    #Tek dimde birleştirip 1000 uzunluğunda tensor yapıyor.
    deneme33 = [torch.stack(batch) for batch in class_preds]#torch.stack var olmayan yeni bir dim'de ekleme yapıyor. 100 tane 10 scalar tensor içeren
    # bir listeyi alıp (100, 10) olan 2d-Tensor'a çeviyor.
    class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    class_labels = torch.cat(class_labels)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

    ############## TENSORBOARD ########################
    classes = range(10)
    for i in classes:
        labels_i = class_labels == i
        preds_i = class_preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()




