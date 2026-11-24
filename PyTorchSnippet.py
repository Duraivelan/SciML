#PyTorch training loop
import torch, torch.nn as nn
model = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(2000):
opt.zero_grad() # clear old gradients
loss = ((model(X) - y) ** 2).mean() # MSE loss
loss.backward() # compute gradients
opt.step() # update weights
