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
for L in [3,5,7,9]:
    X, y = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=False, Reshape=True)
    #X, y = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=True, Reshape=True, Normalize=True)
    print(f"L={L}, number of configurations: {len(y)}")
    # filter X, y to keep only |y| > 0
    mask = torch.abs(y) > -1#1e-6
    all_X.append(X[mask])
    all_y.append(y[mask]) # take absolute value of y to ignore the

print("max of y", torch.max(torch.cat(all_y)))
print("min of y", torch.min(torch.cat(all_y)))
#print(all_y)

in_channels=4
hidden_dim=256 #32
kernel_size=3 #16s
stride=2
activation='tanh'
model = nnets.ConvSkipHole(in_channels=in_channels, hidden_dim=hidden_dim, kernel_size=kernel_size, stride=stride, activation=activation)
#model = nnets.VisionTransformer1D(patch_size=kernel_size, embed_dim=64,mlp_dim=128,activation=activation)

criterion = total_squared_loss
#criterion = nn.CrossEntropyLoss()
#criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01)#, weight_decay=1e-3) #lr=0.01

# fn = f"data/convnet_checkpoint_hid{hidden_dim}_ker{kernel_size}_epoch1999.pth"
# if os.path.exists(fn):
#     print("loading model from", fn)
#     ckpt = torch.load(fn, map_location="cpu")
#     model.load_state_dict(ckpt["model_state_dict"])
#     optimizer.load_state_dict(ckpt["optimizer_state_dict"])
#     start_epoch = ckpt.get("epoch", 0) + 1
#     model.train()


# Training loop (example: 500 epochs)
for epoch in range(2000):
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

# save full training checkpoint (recommended if you want to resume)
# torch.save({
#     "epoch": epoch,
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     "loss": loss.item() if hasattr(loss, "item") else None,
# }, f"data/convnet_checkpoint_hid{hidden_dim}_ker{kernel_size}_epoch{epoch}.pth")

# Example: predict coefficients for all configurations
with torch.no_grad():
    predicted_coeffs = model(X)
    #print("True coeffs:", y.numpy())
    #print("Predicted coeffs:", predicted_coeffs.numpy())
    print("total loss:", np.sum((predicted_coeffs.numpy() - y.numpy())**2))
    print("fidelity:", np.sum((predicted_coeffs.numpy() * y.numpy())))




L = 11
Xtest, ytest = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=False, Reshape=True)
mask = torch.abs(ytest) > 1e-6
Xtest = Xtest[mask]
ytest = ytest[mask] # take absolute value of y to ignore the
with torch.no_grad():
    predicted_coeffs = model(Xtest)
    #print("True coeffs:", ytest.numpy())
    #print("Predicted coeffs:", predicted_coeffs.numpy())
    #print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - ytest.numpy())**2)/len(ytest))
    print("total loss:", np.sum((predicted_coeffs.numpy() - ytest.numpy())**2))
    print("fidelity:", np.sum((predicted_coeffs.numpy() * ytest.numpy())))
    print("size", len(ytest))

# temp = predicted_coeffs - ytest
# for i in range(len(ytest)):
#     if abs(temp[i]) > 0.5:
#         print("index", i, "true", ytest[i].item(), "pred", predicted_coeffs[i].item(), "diff", temp[i].item())