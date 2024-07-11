import torch
import numpy as np
import torch.nn as nn

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0]) Normalize etmek için


x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print(outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)

#Cross-Entropy, output layer activation function'ı softmax olan nn'lerde loss hesaplama için kullanılır.
#Target label One-Hot Encoded olmalı. Prediction her class için olasılık veriyor olmalı.

y = np.array([1,0,0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(y, Y_pred_good)
l2 = cross_entropy(y, Y_pred_bad)
print(f"Loss1: {l1:.4f}")
print(f"Loss2: {l2:.4f}")

#Pytorch

loss = nn.CrossEntropyLoss() #nn.LogSoftmax + nn.NLLLoss uygular. Son layerda softmax yapma yani. Y_real One-Hot Encoded olmamalı. Sadece doğru class label olmalı.
#Y_pred de raw score. No softmax
#Bu fonk otomatik softmax uygular önce. Sonra loss hesaplar.

# 1 Sample
Y = torch.tensor([0])  #Her sample için target class. Sadece gerçek class.
#nsamples x nclasses = 1X3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]]) #Her sample için tüm classlara ait çıktı değeri olmalı
# Bu yüzden 2d tensor kullanılır

loss_2 = loss(Y_pred_good, Y)
loss_3 = loss(Y_pred_bad, Y)

print(loss_2.item())
print(loss_3.item())

_, predictions_1 = torch.max(Y_pred_good, 1)
_, predictions_2 = torch.max(Y_pred_bad, 1)

print(predictions_1)
print(predictions_2)


#3 samples
Y = torch.tensor([2,0,1])

Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

loss_2 = loss(Y_pred_good, Y)
loss_3 = loss(Y_pred_bad, Y)

print(loss_2.item())
print(loss_3.item())

_, predictions_1 = torch.max(Y_pred_good, 1)
_, predictions_2 = torch.max(Y_pred_bad, 1)

print(predictions_1)
print(predictions_2)

loss_func_2 = nn.BCELoss() #Output binary size, son layerda sigmoid activation yapma. Bu foonk otomatik sigmoid uygular önce.
