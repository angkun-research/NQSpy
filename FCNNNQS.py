import numpy as np
import torch
import torch.optim as optim

import NeuralNetworks as nnets
from ExactGS import obtain_train_data
from utils import total_squared_loss, fidelity_loss
#from vmc_utils import FCNet

L = 7
t1 = 1.0
t2 = 0.5
X, y = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=True, Reshape=False, Normalize=True)
print(len(y))
model = nnets.FCNet(L)
#model = FCNet(L)

criterion = total_squared_loss
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-10)

# Training loop (example: 500 epochs)
for epoch in range(1000):
    optimizer.zero_grad() # zero the gradient buffers
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step() # update parameters
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Example: predict coefficients for all configurations
with torch.no_grad():
    predicted_coeffs = model(X)
    #print("True coeffs:", y.numpy())
    #print("Predicted coeffs:", predicted_coeffs.numpy())
    print("total loss:", np.sum((predicted_coeffs.numpy() - y.numpy())**2))
    print("fidelity:", np.sum((predicted_coeffs.numpy() * y.numpy())))