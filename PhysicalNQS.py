import numpy as np
import torch
import torch.optim as optim

import NeuralNetworks as nnets
from ExactGS import obtain_train_data
from utils import total_squared_loss

L = 11
t1 = 1.0
t2 = 0.5
TBcoeff = True
X, y = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=TBcoeff, Reshape=True, Normalize=True)

model = nnets.PhysicsLocalLayer()

criterion = total_squared_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop (example: 500 epochs)
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
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



