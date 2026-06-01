import torch
import torch.nn as nn
from torch.nn import functional as F
from data_loader import *

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,64), 
            nn.Tanh(), 
            nn.Linear(64, 3), 
            nn.Softmax(1))
        self.beta = nn.Parameter(torch.tensor([0.5]))
        self.gamma = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        y = self.net(x)
        return y[:,0:1], y[:,1:2], y[:,2:3]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = load_latest_run_data()
x = data[0].to(device)
y1 = data[1].to(device)
y2 = data[2].to(device)
y3 = data[3].to(device)

t_max = len(x)
model = PINN().to(device)
opt = torch.optim.Adam(model.parameters())

for i in range(10000):
    x.requires_grad = True
    
    p1, p2, p3 = model(x)
    
    l1 = F.mse_loss(p2, y2) + F.mse_loss(p3, y3)

    d1 = torch.autograd.grad(p1, x, torch.ones_like(p1), True)[0]
    d2 = torch.autograd.grad(p2, x, torch.ones_like(p2), True)[0]
    d3 = torch.autograd.grad(p3, x, torch.ones_like(p3), True)[0]
    
    r1 = d1 - (-t_max * model.beta * p1 * p2)
    r2 = d2 - (t_max * model.beta * p1 * p2 - t_max * model.gamma * p2)
    r3 = d3 - (t_max * model.gamma * p2)
    
    l2 = torch.mean(r1**2) + torch.mean(r2**2) + torch.mean(r3**2)
    
    loss = l1 + (1e-6 * l2)

    loss.backward()
    opt.step()
    opt.zero_grad()
    
    if i % 1000 == 0:
        print(f"Epoch: {i} | Loss: {loss.item()}")

print(f"Beta: {model.beta.item()}")
print(f"Gamma: {model.gamma.item()}")