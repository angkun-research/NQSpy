import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import NeuralNetworks as nnets
from ExactGS import obtain_train_data
from utils import total_squared_loss


t1 = 1.0
t2 = 0.5
all_X = []
all_y = []
for L in [3,5,7,9,11]:
    X, y = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=False, Reshape=True, Normalize=False)
    #X, y = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=True, Reshape=True, Normalize=True)
    print(f"L={L}, number of configurations: {len(y)}")
    # filter X, y to keep only |y| > 0
    mask = torch.abs(y) > -1 #1e-6
    print(f"After filtering, number of configurations: {torch.sum(mask).item()}")
    all_X.append(X[mask])
    all_y.append(y[mask]) # take absolute value of y to ignore the

print("max of y", torch.max(torch.cat(all_y)))
print("min of y", torch.min(torch.cat(all_y)))
#print(all_y)

in_channels=4
hidden_dim= 16 #256 #32
kernel_size=2  #16s
stride=2
model = nnets.LocalRule(in_channels=in_channels, hidden_dim=hidden_dim, kernel_size=kernel_size, stride=stride)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters in the neural network: {n_params}")

criterion = total_squared_loss
optimizer = optim.Adam(model.parameters(), lr=0.01)#, weight_decay=1e-3) #lr=0.01

# Training loop (example: 500 epochs)
for epoch in range(4000):
    optimizer.zero_grad()
    #outputs = model(X)
    #loss = criterion(outputs, y)
    loss = 0
    for X, y in zip(all_X, all_y):
        outputs = model(X)
        loss += criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Example: predict coefficients for all configurations
with torch.no_grad():
    predicted_coeffs = model(X)
    #print("True coeffs:", y.numpy())
    #print("Predicted coeffs:", predicted_coeffs.numpy())
    print("total loss:", np.sum((predicted_coeffs.numpy() - y.numpy())**2))
    print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - y.numpy())**2)/len(y))
    predicted_coeffs = predicted_coeffs / torch.norm(predicted_coeffs)
    y = y / torch.norm(y)
    fidelity = torch.abs(torch.dot(predicted_coeffs, y)).item()
    print(f"fidelity: {fidelity:.8f}")


L = 13
Xtest, ytest = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=False, Reshape=True, Normalize=False)
mask = torch.abs(ytest) > -1 
Xtemp = Xtest[mask]
ytemp = ytest[mask] # take absolute value of y to ignore the
with torch.no_grad():
    predicted_coeffs = model(Xtemp)
    predicted_coeffs = predicted_coeffs / torch.norm(predicted_coeffs)
    ytemp = ytemp / torch.norm(ytemp)
    fidelity = torch.abs(torch.dot(predicted_coeffs, ytemp)).item()
    print(f"L={L}, Fidelity: {fidelity}")


#Save checkpoint
folder = "data/"
filename = f"SignRule_t2{t2}_hidden{hidden_dim}.pth" #"data/psi_checkpoint.pth"
save_path = folder + filename
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, save_path)
print(f"Saved checkpoint to {save_path}")

# checkpoint = torch.load(save_path, map_location=device)
# psi.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
# psi.to(device)
# print("Loaded checkpoint.")
