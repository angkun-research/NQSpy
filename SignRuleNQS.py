import numpy as np
import torch
import torch.optim as optim

import NeuralNetworks as nnets
from ExactGS import obtain_train_data
from utils import total_squared_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

L = 15 #11
t1 = 1.0
t2 = 0.5
TBcoeff = False #True
Normalize = True
X, y = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=TBcoeff, Reshape=True, Normalize=Normalize)

print("max of y", torch.max(y))
print("min of y", torch.min(y))
rescale = 1.0 #1/torch.max(y)
print("rescale factor:", rescale)

LocalRule = nnets.ConvSkipHole(in_channels=4, hidden_dim=32, kernel_size=2, stride=2, activation='tanh')
optimizer = optim.Adam(LocalRule.parameters(), lr=0.01)#, weight_decay=1e-3) #lr=0.01
#print number of parameters
total_params = sum(p.numel() for p in LocalRule.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

folder = "data/"
filename = f"SignRule_t2{t2}_hidden32.pth"
save_path = folder + filename
checkpoint = torch.load(save_path, map_location=device)
LocalRule.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
LocalRule.to(device)
print("Loaded SignRule.")


psi =  nnets.HoleOnLocalRule(LocalRule, L)

total_params = sum(p.numel() for p in psi.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

criterion = total_squared_loss
optimizer = optim.Adam(psi.parameters(), lr=0.01)

# Training loop (example: 500 epochs)
for epoch in range(500):
    optimizer.zero_grad()
    outputs = psi(X)
    loss = criterion(outputs, y*rescale)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


with torch.no_grad():
    predicted_coeffs = psi(X)
    #print("True coeffs:", y.numpy())
    #print("Predicted coeffs:", predicted_coeffs.numpy())
    print("total loss:", np.sum((predicted_coeffs.numpy() - y.numpy())**2))
    print("fidelity:", np.sum((predicted_coeffs.numpy() * y.numpy())))



