import numpy as np
import torch
import torch.optim as optim

import NeuralNetworks as nnets
from ExactGS import obtain_train_data
from utils import total_squared_loss

L = 13 #11
t1 = 1.0
t2 = 0.5
TBcoeff = True
Normalize = True
X, y = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=TBcoeff, Reshape=True, Normalize=Normalize)
print("Shape of X:", X.shape) # 12012, 13, 4
print("max of y", torch.max(y))
print("min of y", torch.min(y))
rescale = 1.0 #1/torch.max(y)
print("rescale factor:", rescale)

in_channels=4
hidden_dim = 64 #64 #32 #32
kernel_size = 2 #3 #16
stride = 2
activation = 'tanh'
holewave = True #False 
#model = nnets.FCNet(L, hidden_dim=hidden_dim)
model = nnets.ConvNet(hidden_dim=hidden_dim, kernel_size=kernel_size)
#model = nnets.VisionTransformer1D()
#model = nnets.LdepConvPlusFC(L, Conv_dim=hidden_dim,kernel_size=kernel_size)

#print number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

criterion = total_squared_loss
optimizer = optim.Adam(model.parameters(), lr=1e-2) #1e-3

epochs = 2000
# Training loop (example: 500 epochs)
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y*rescale)
    loss.backward()
    optimizer.step()
    # if epoch % 100 == 0:
    #     print(f"Epoch {epoch}, Loss: {loss.item()}")
    print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Final loss:", loss.item())

with torch.no_grad():
    predicted_coeffs = model(X)
    #print("True coeffs:", y.numpy())
    #print("Predicted coeffs:", predicted_coeffs.numpy())
    norm_val = torch.norm(predicted_coeffs)
    if norm_val > 0:
        predicted_coeffs = predicted_coeffs / norm_val
    print("total loss:", np.sum((predicted_coeffs.numpy() - y.numpy())**2))
    print("fidelity:", np.sum((predicted_coeffs.numpy() * y.numpy())))



