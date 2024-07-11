import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

#model = Model(n_input_features=6)

FILE = "model.pth" #Pratikte .pth yapmak yaygın. Pytorch kısaltması
#torch.save(model, FILE)    #Tensor, model, parametre dict'i arg olabilir. Python'ın pickle'ını kullanır ve serialize ederek save eder.

#Bu kaydetmek için lazy yol.
model = torch.load(FILE)
model.eval()

for param in model.parameters():
    print(param)


FILE2 = "state_dict.pth"
model = Model(n_input_features=6)
torch.save(model.state_dict(), FILE2)

for param in model.parameters():
    print(param)

loaded_model = Model(n_input_features=6) #Parametreleri kaydetmek için boş bir model oluşturmamız lazım.
loaded_model.load_state_dict(torch.load(FILE2))#Düz path veremezsin argüman olarak. Dictionary'nin kendisini vermelisin
loaded_model.eval()
for param in loaded_model.parameters():
    print(param) #Weightleri ve biası yazdırır.


learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(loaded_model.state_dict())
print(optimizer.state_dict())

checkpoint = {
    "epoch":90,
    "model_state":model.state_dict(),
    "optim_state":optimizer.state_dict()
}

torch.save(checkpoint, "checkpoint.pth")

loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]

model_2 = Model(n_input_features=6)
optimizer = torch.optim.SGD(model_2.parameters(), lr=0)

model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optim_state"])
#Checkpoint aldık ve yükledik. Training'e bu noktadan devam edebiliriz.

print(optimizer.state_dict())

#Save on GPU, load on CPU
device = torch.device("cuda")
model_2.to(device)
torch.save(model.state_dict(), "cuda_mod.pth")

device = torch.device("cpu")
model = Model(6)
model.load_state_dict(torch.load("cuda_mod.pth", map_location=device))

#Save on GPU, load on GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), "cuda_to_cuda.pth")

model = Model(6)
model.load_state_dict(torch.load("cuda_to_cuda.pth"))
model.to(device)
#Modele input verirken önce hepsi cuda'ya aktarılmalı.

# 3) Save on CPU, Load on GPU
torch.save(model.state_dict(), "cpu_to_cuda.pth")

device = torch.device("cuda")
model = Model(6)
model.load_state_dict(torch.load("cpu_to_cuda.pth", map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
#model.to(torch.device('cuda')) ile modelin parametrelerinin CUDA'ya geçmesini sağla.