import torch
import numpy as np

x = torch.empty(1) #scalar value
x = torch.empty(3) #1d tensor
x = torch.empty(2,3) #2d tensor
print(x)
y = torch.rand(2,3)
print(y)
z = torch.zeros(2,3)
print(z)
a = torch.ones(2,3)
print(a)
print(a.dtype)
b = torch.ones(2, 3, dtype=torch.double)
c = torch.ones(2, 3, dtype=int)
d = torch.ones(2, 3, dtype=torch.int)
print(b.dtype, c.dtype, d.dtype)
print(d.size())

x_1 = torch.tensor([2,5.3,11,15])
print(x_1, x_1.type, x_1.dtype)
f = torch.rand(2,3)
g = torch.rand(2,3)

print(f)
print(f+g) #Element-wise addition
print(torch.add(f,g)) #same as above
print(f.add_(g)) #in-place addition. In pytorch, trailing underscore in a function means in-place operation.
print(f) #f changed
print(g)
print(torch.sub(f,g))
print(f * g) #Element-wise
print(torch.mul(f, g))
print(torch.div(f, g)) #Element-wise
print(f.mul_(g)) #In-place

h = torch.rand(5,3)
print(h)
print(h[:,1]) #Slicing
print(h[1,:])
print(h[1,1], type(h[1,1]))
print(h[1,1].item(), type(h[1,1].item()))

j = torch.rand(4,4)
print(j)
print(j.view(16)) #Reshape
print(j.view(-1,8)) #Automatically determine -1 dimension's size.
#print(j.view(-1,10)) invalid for input of size 16
print(j.view(2,-1,2), j.view(2,-1,2).size())

k = torch.rand(5)
h = torch.rand(4,2)
print(type(k), type(h))
h = k.numpy()
print(type(h))
print(k, h)
print(k.add_(1), h) #While tensor is in cpu, k and h have same memory address. Therefore, when one is changed,
# the other is automatically changed.
m = np.ones(5)
print(m, type(m))
n = torch.from_numpy(m)
#n = torch.from_numpy(m, dtype=torch.float) Dtype cannot be defined.
print(n, type(n))
m+=1
print(m, n) #Same as above

if torch.cuda.is_available():#Is cuda available?
    device = torch.device("cuda")
    a = torch.ones(5, device=device)#If available, create tensor in GPU.
    b = torch.ones(6) #First create tensor in CPU, then move it to GPU.
    b = b.to(device)
    c = a + b #Working in GPU, therefore might be faster.
    print(type(c))
    #c.numpy() is not working, because Numpy can only handle CPU tensors.
    c = c.to("cpu")
    c.numpy()
    print(type(c))
else:
    print("Cuda is not available")

a = torch.ones(3, requires_grad=True) #Default is false. It means this tensor will be used for optimization and
# the gradient will be calculated.
print(a)

x = torch.randn(3, requires_grad=True)
y = x + 2
print(y)
z = y * y * 2
print(z)
a = z.mean()
print(a)
a.backward() #Jacobian gradient calculation
print(x.grad) #For just leaf Tensor
#print(z.grad) Invalid
#b = a * 83
#b.backward() Cannot be calculate grad second time.
print(x.grad)

b = x + 6
c = b * b * 4
v = torch.tensor([0.1, 1.0, 0.1], dtype=torch.float32)
c.backward(v) #Scalar olmayan bir şeyden gradient hesaplarken argüman olarak vector verilmesi lazım.
print(x.grad)

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    outputs = (weights * 2).sum()
    outputs.backward() #Backward is accumulative. Her train loopunda backward hesaplarken her seferinde eski grad'ın üstüne ekler.
    print(weights.grad)
    weights.grad.zero_()

"""weights = torch.ones(4, requires_grad=True)
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()   Torch optimizerlarında da her iterationda gradı sıfırlamak gerekir."""


x = torch.randn(4, requires_grad=True)
#x = x * x * x + 15 Bu eklenirse a artık leaf Tensor olmadığı için hata verir. Sadece leaf Tensorlar .grad attribute ile erişilebilir.
a = torch.randn(4, requires_grad=True)
y = x * a + 5
z = y * y * y
z = z.sum()
z.backward()
print(x.grad)
print(a.grad)


x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

y_hat = x*w
loss = (y_hat - y) ** 2
print(loss)
loss.backward() #Hem local gradient'ları hesaplar hem de backward propagation yapar.
print(w.grad)

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
X_test = torch.tensor([5], dtype=torch.float32)
#f = w * x
#Real f = 2 * x
n_samples, n_features = X.shape
print(n_samples, n_features)
input_size = n_features
output_size = n_features
model = torch.nn.Linear(input_size, output_size)
#Model prediction
def forward(x):
    return w * x
#loss calculation = MSE()
def loss(y_real, y_pred):
    return ((y_real - y_pred) ** 2).mean()
#gradient descent
#dJ/dw = 1/N * 2 * x * (w*x - y_real)
def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

learning_rate = 0.01
n_iters = 20
loss_auto = torch.nn.MSELoss()
#optimizer = torch.optim.SGD([w], lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(n_iters):
    prediction = model(X)
    #l = loss(Y, prediction)
    l = loss_auto(Y, prediction)
    #grad = gradient(X, Y, prediction)
    l.backward()
    """with torch.no_grad():
        w -= learning_rate*w.grad #Bu operation'ın gradient graph'da bir parça olmaması için no_grad() ile yapıldı. Aksi
        #halde gradient'ın bir parçası olurdu ve sonraki gradient hesaplamalarının bir parçası olarak local gradient'ı alınırdı."""
    optimizer.step()
    #w.grad.zero_()
    optimizer.zero_grad()

    if epoch % 2 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch+1}: w={w[0][0].item():.3f}, Yuvarlama olmadan w={w[0][0].item()}, loss={l:.8f}")

#print(f"Prediction after training: f(5) = {forward(5):.3f}, Yuvarlama olmadan f(5) = {forward(5)}")
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')


class LinearRegression(torch.nn.Module): #Elle yeni model oluşturma. Torch'daki tüm modeller nn.Module'in subclassıdır.
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)



