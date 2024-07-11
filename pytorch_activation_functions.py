import torch
import torch.nn as nn
import torch.nn.functional as F
#First option for using activation functions
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


#Second option for using activation functions

class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.hidden1(x))
        out_2 = torch.sigmoid(self.hidden2(out))
        return out_2

#Tüm activation functionlar torch.relu gibi bulunmayabilir. Ama hepsi torch.nn.functional.relu olarak bulunur. ÖNEMLİ!!!