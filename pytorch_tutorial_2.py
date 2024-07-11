import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#Prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1) #Dobule dtype def olarak.

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(y_numpy.astype(np.float32)) #Def olarak 1 row.

y = Y.view(Y.shape[0], 1)
y_2 = Y.view(-1, 1)

print(y.shape, y_2.shape, sep="\n")

n_samples, n_features = X.shape

#Define model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

#loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#training loop
num_epochs = 100
for epoch in range(num_epochs):
    #Forward pass and loss
    prediction = model(X)
    loss = criterion(prediction, y)

    #Backward pass
    loss.backward()

    #Update weights
    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f"epoch: {epoch+1}, loss = {loss.item():.4f}")

#plot
predicted = model(X).detach().numpy() #Bu işlemi computational graph'dan çıkarır. Böylelikle backward() fonk.u bu işlemi gradient
# hesaplamalarında hesaplamaz. Requires_gradient attribute = false olur.
plt.plot(X_numpy, y_numpy, "ro")
plt.plot(X_numpy, predicted, "b")
plt.show()



