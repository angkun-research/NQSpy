import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import NeuralNetworks as nnets
from ExactGS import obtain_train_data
from utils import total_squared_loss


class Class3States(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=32, kernel_size=2, stride=2):
        super(Class3States, self).__init__()
        self.pad = (kernel_size-1)//2
        self.layer1 = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=self.pad, stride=stride)
        # 1x1 conv to produce sign and value per pair (classes: +1, -1, 0)
        self.classifier = nn.Conv1d(hidden_dim, 2, kernel_size=1)
        # produce 3 logits per pair: [p_updown, p_downup, p_other]
        #self.classifier = nn.Conv1d(hidden_dim, 3, kernel_size=1)
        self.pool = nnets.GeometricPool1d()
        self.finallayer = nn.Linear(hidden_dim, 1)
        #self.finallayer = nn.Linear(1, 1)

    def forward(self, x):
        # remove hole site before convolution
        batch_size, L, C = x.shape
        device = x.device
        # assume exactly one hole per sample encoded as channel 0 == 1
        # get hole indices (shape: (B,))
        hole_idx = torch.argmax(x[:, :, 0], dim=1)
        # build mask: True for positions that are NOT the hole
        idx = torch.arange(L, device=device)
        mask = idx.unsqueeze(0) != hole_idx.unsqueeze(1)  # (B, L)
        # select non-hole positions and reshape -> (B, L-1, C)
        x_nohole = x[mask].view(batch_size, L - 1, C)
        # x shape: (batch, L, 4) -> (batch, 4, L)
        x = x_nohole.permute(0, 2, 1)

        #h = torch.sigmoid(self.layer1(x))
        # h = self.layer1(x)
        # logits = self.classifier(h)  # shape: (batch, 2, L//2)
        # sign_logit = logits[:, 0, :]  # shape: (batch, L//2)
        # value_logit = logits[:, 1, :]  # shape: (batch, L//2)
        # sign = torch.tanh(sign_logit)  # in [-1, 1]
        # value = torch.sigmoid(value_logit)  # in [0, 1]
        # x = sign * value  # shape: (batch, L//2)

        #x = self.pool(x).unsqueeze(-1) # shape: (batch,1)

        x = torch.tanh(self.layer1(x).pow(5)) # .pow(3)
        #x = self.layer1(x).pow(5)
        x = self.pool(x).squeeze(-1)  # shape: (batch, hidden_dim)
        x = self.finallayer(x)
        #return torch.tanh(x).squeeze(-1)  # output in [-1, 1] # tanh behaves worse
        #return torch.sigmoid(x).squeeze(-1)
        #return x.squeeze(-1)

        return x.squeeze(-1)


t1 = 1.0
t2 = 0.5
all_X = []
all_y = []
for L in [3,5,7,9,11]:
    X, y = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=False, Reshape=True)
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
hidden_dim= 32 #256 #32
kernel_size=2  #16s
stride=2
activation= 'tanh'
#model = nnets.ConvSkipHole(in_channels=in_channels, hidden_dim=hidden_dim, kernel_size=kernel_size, stride=stride, activation=activation)
#model = nnets.VisionTransformer1D(patch_size=kernel_size, embed_dim=64,mlp_dim=128,activation=activation)
model = Class3States(in_channels=in_channels, hidden_dim=hidden_dim, kernel_size=kernel_size, stride=stride)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters in the neural network: {n_params}")

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
    print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - y.numpy())**2)/len(y))
    #print("fidelity:", np.sum((predicted_coeffs.numpy() * y.numpy())))
    print("size", len(y))




L = 13
Xtest, ytest = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=False, Reshape=True)
mask = torch.abs(ytest) > 1e-6
Xtemp = Xtest[mask]
ytemp = ytest[mask] # take absolute value of y to ignore the
with torch.no_grad():
    predicted_coeffs = model(Xtemp)
    #print("True coeffs:", ytemp.numpy())
    #print("Predicted coeffs:", predicted_coeffs.numpy())
    print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2)/len(ytemp))
    print("total loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2))
    #print("fidelity:", np.sum((predicted_coeffs.numpy() * ytemp.numpy())))
    print("size", len(ytemp))

mask = torch.abs(ytest) < 0.1 
Xtemp = Xtest[mask]
ytemp = ytest[mask] # take absolute value of y to ignore the
with torch.no_grad():
    predicted_coeffs = model(Xtemp)
    #print("True coeffs:", ytemp.numpy())
    #print("Predicted coeffs:", predicted_coeffs.numpy())
    print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2)/len(ytemp))
    print("total loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2))
    #print("fidelity:", np.sum((predicted_coeffs.numpy() * ytemp.numpy())))
    print("size", len(ytemp))

# temp = predicted_coeffs - ytest
# for i in range(len(ytest)):
#     if abs(temp[i]) > 0.5:
#         print("index", i, "true", ytest[i].item(), "pred", predicted_coeffs[i].item(), "diff", temp[i].item())



#Save checkpoint
# folder = "data/"
# filename = f"SignRule_t2{t2}_hidden{hidden_dim}.pth" #"data/psi_checkpoint.pth"
# save_path = folder + filename
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
# }, save_path)
# print(f"Saved checkpoint to {save_path}")


# checkpoint = torch.load(save_path, map_location=device)
# psi.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
# psi.to(device)
# print("Loaded checkpoint.")


L = 15
Xtest, ytest = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=False, Reshape=True)
mask = torch.abs(ytest) > 1e-6
Xtemp = Xtest[mask]
ytemp = ytest[mask] # take absolute value of y to ignore the
with torch.no_grad():
    predicted_coeffs = model(Xtemp)
    #print("True coeffs:", ytemp.numpy())
    #print("Predicted coeffs:", predicted_coeffs.numpy())
    print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2)/len(ytemp))
    print("total loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2))
    #print("fidelity:", np.sum((predicted_coeffs.numpy() * ytemp.numpy())))
    print("size", len(ytemp))

mask = torch.abs(ytest) < 0.1 
Xtemp = Xtest[mask]
ytemp = ytest[mask] # take absolute value of y to ignore the
with torch.no_grad():
    predicted_coeffs = model(Xtemp)
    #print("True coeffs:", ytemp.numpy())
    #print("Predicted coeffs:", predicted_coeffs.numpy())
    print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2)/len(ytemp))
    print("total loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2))
    #print("fidelity:", np.sum((predicted_coeffs.numpy() * ytemp.numpy())))
    print("size", len(ytemp))


L = 17
Xtest, ytest = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=False, Reshape=True)
mask = torch.abs(ytest) > 1e-6
Xtemp = Xtest[mask]
ytemp = ytest[mask] # take absolute value of y to ignore the
with torch.no_grad():
    predicted_coeffs = model(Xtemp)
    #print("True coeffs:", ytemp.numpy())
    #print("Predicted coeffs:", predicted_coeffs.numpy())
    print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2)/len(ytemp))
    print("total loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2))
    #print("fidelity:", np.sum((predicted_coeffs.numpy() * ytemp.numpy())))
    print("size", len(ytemp))

mask = torch.abs(ytest) < 0.1 
Xtemp = Xtest[mask]
ytemp = ytest[mask] # take absolute value of y to ignore the
with torch.no_grad():
    predicted_coeffs = model(Xtemp)
    #print("True coeffs:", ytemp.numpy())
    #print("Predicted coeffs:", predicted_coeffs.numpy())
    print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2)/len(ytemp))
    print("total loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2))
    #print("fidelity:", np.sum((predicted_coeffs.numpy() * ytemp.numpy())))
    print("size", len(ytemp))