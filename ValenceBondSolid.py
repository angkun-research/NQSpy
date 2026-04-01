import numpy as np
import torch
import torch.optim as optim
import itertools

from ExactGS import build_MB_basis_holes
from ExactGS import basis_to_spinconfig_holes,basis_to_onehot
import NeuralNetworks as nnets
from utils import total_squared_loss

def Build_VBS(L):
    assert L % 2 == 0, "L must be even"
    Npair = L // 2
    vbs_basis = []
    vbs_coeffs = []
    singlet_pairs = [(2*i, 2*i+1) for i in range(Npair)]
    for up_sites in itertools.product(*singlet_pairs):
        sign = 1
        for idx, pair in enumerate(singlet_pairs):
            if up_sites[idx] == pair[0]:
                sign *= 1
            elif up_sites[idx] == pair[1]:
                sign *= -1
        vbs_basis.append(((),up_sites))
        vbs_coeffs.append(sign)
    return vbs_basis, vbs_coeffs

def obtain_train_data_VBS(L, Reshape=False,Normalize=False):
    basis = build_MB_basis_holes(L, nholes=0)  # No holes for VBS
    basis_dict = {state: idx for idx, state in enumerate(basis)}
    vbs_basis, vbs_coeffs = Build_VBS(L)
    vbs_vec = np.zeros(len(basis))
    for config, coeff in zip(vbs_basis, vbs_coeffs):
        idx = basis_dict[config]
        vbs_vec[idx] = coeff

    if Normalize:
        factor = np.sqrt(2)**(L//2)  # Normalization factor for VBS state
        vbs_vec = vbs_vec / factor

    spin_configs = basis_to_spinconfig_holes(basis, L)
    coeffs = np.array(vbs_vec)

    onehot_configs = basis_to_onehot(spin_configs, L)
    if Reshape:
        onehot_configs = onehot_configs.reshape(-1, L, 4)  # for reshaping into (num_states, L, 4)

    X = torch.tensor(onehot_configs, dtype=torch.float32)
    y = torch.tensor(coeffs, dtype=torch.float32)
    return X, y

# all_X = []
# all_y = []
# for L in [2, 4, 6, 8, 10]:
#     X, y = obtain_train_data_VBS(L, Reshape=True)
#     #X, y = obtain_train_data(L, t1=t1, t2=t2, TBcoeff=True, Reshape=True, Normalize=True)
#     print(f"L={L}, number of configurations: {len(y)}")
#     # filter X, y to keep only |y| > 0
#     mask = torch.abs(y) > -1.1 #1e-6
#     print(f"After filtering, number of configurations: {torch.sum(mask).item()}")
#     all_X.append(X[mask])
#     all_y.append(y[mask]) # take absolute value of y to ignore the

L = 14
X, y = obtain_train_data_VBS(L, Reshape=True)
all_X = [X]
all_y = [y] 

print("max of y", torch.max(torch.cat(all_y)))
print("min of y", torch.min(torch.cat(all_y)))

in_channels=4
hidden_dim= 16 #256 #32
kernel_size=2  #16s
stride=2
activation= 'tanh'
#model = nnets.ConvSkipHole(in_channels=in_channels, hidden_dim=hidden_dim, kernel_size=kernel_size, stride=stride, activation=activation,nholes=0)
hidden_dim= 1024
model = nnets.FCNet(L, hidden_dim=hidden_dim)


n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters in the neural network: {n_params}")

criterion = total_squared_loss
optimizer = optim.Adam(model.parameters(), lr=0.01)#, weight_decay=1e-3) #lr=0.01

# Training loop (example: 500 epochs)
for epoch in range(5000):
    optimizer.zero_grad()
    # outputs = model(X)
    # loss = criterion(outputs, y)
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
    #print("fidelity:", np.sum((predicted_coeffs.numpy() * y.numpy())))
    print("size", len(y))

# print(model.layer1.weight.shape)
# print(model.layer1.weight.data) 


# L = 12
# Xtest, ytest = obtain_train_data_VBS(L,Reshape=True)
# mask = torch.abs(ytest) > 1e-6
# Xtemp = Xtest[mask]
# ytemp = ytest[mask] # take absolute value of y to ignore the
# with torch.no_grad():
#     predicted_coeffs = model(Xtemp)
#     print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2)/len(ytemp))
#     print("total loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2))
#     print("size", len(ytemp))

# mask = torch.abs(ytest) < 0.1 
# Xtemp = Xtest[mask]
# ytemp = ytest[mask] # take absolute value of y to ignore the
# with torch.no_grad():
#     predicted_coeffs = model(Xtemp)
#     print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2)/len(ytemp))
#     print("total loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2))
#     print("size", len(ytemp))

# L = 16
# Xtest, ytest = obtain_train_data_VBS(L,Reshape=True)
# mask = torch.abs(ytest) > 1e-6
# Xtemp = Xtest[mask]
# ytemp = ytest[mask] # take absolute value of y to ignore the
# with torch.no_grad():
#     predicted_coeffs = model(Xtemp)
#     print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2)/len(ytemp))
#     print("total loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2))
#     print("size", len(ytemp))

# mask = torch.abs(ytest) < 0.1 
# Xtemp = Xtest[mask]
# ytemp = ytest[mask] # take absolute value of y to ignore the
# with torch.no_grad():
#     predicted_coeffs = model(Xtemp)
#     print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2)/len(ytemp))
#     print("total loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2))
#     print("size", len(ytemp))


# L = 20
# Xtest, ytest = obtain_train_data_VBS(L,Reshape=True)
# mask = torch.abs(ytest) > 1e-6
# Xtemp = Xtest[mask]
# ytemp = ytest[mask] # take absolute value of y to ignore the
# with torch.no_grad():
#     predicted_coeffs = model(Xtemp)
#     print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2)/len(ytemp))
#     print("total loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2))
#     print("size", len(ytemp))

# mask = torch.abs(ytest) < 0.1 
# Xtemp = Xtest[mask]
# ytemp = ytest[mask] # take absolute value of y to ignore the
# with torch.no_grad():
#     predicted_coeffs = model(Xtemp)
#     print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2)/len(ytemp))
#     print("total loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2))
#     print("size", len(ytemp))



