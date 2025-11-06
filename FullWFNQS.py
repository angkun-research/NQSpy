import numpy as np
import torch
import torch.optim as optim

import NeuralNetworks as nnets
from ExactGS import obtain_train_data
from utils import total_squared_loss

L = 11 #11
t1 = 1.0
t2 = 0.5
TBcoeff = True
Normalize = True
X, y = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=TBcoeff, Reshape=True, Normalize=Normalize)

print("max of y", torch.max(y))
print("min of y", torch.min(y))
rescale = 1.0 #1/torch.max(y)
print("rescale factor:", rescale)

in_channels=4
hidden_dim = 32 #32
kernel_size = 2 #16
stride = 2
activation = 'tanh'
holewave = True #False 
# model = nnets.ConvSkipHole(in_channels=in_channels, hidden_dim=hidden_dim, 
#                            kernel_size=kernel_size, stride=stride,
#                            activation=activation, holewave=holewave)
model = nnets.LdepConvSkipHole(L, in_channels=in_channels, hidden_dim=hidden_dim,
                               kernel_size=kernel_size, stride=stride,holewave=holewave)
# model = nnets.LdepConv2d(L, in_channels=in_channels, hidden_dim=hidden_dim,kernel_size=kernel_size)
#print number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

criterion = total_squared_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop (example: 500 epochs)
for epoch in range(5000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y*rescale)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


with torch.no_grad():
    predicted_coeffs = model(X)
    #print("True coeffs:", y.numpy())
    #print("Predicted coeffs:", predicted_coeffs.numpy())
    print("total loss:", np.sum((predicted_coeffs.numpy() - y.numpy())**2))
    print("fidelity:", np.sum((predicted_coeffs.numpy() * y.numpy())))



