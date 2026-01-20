import torch
import torch.optim as optim
import numpy as np

from vmc_utils import FCNet,PhysicalNN  # or ConvNet, etc.
#from vmc_utils import build_MB_basis
from vmc_utils import state_to_onehot, GlobalSampler
from vmc_utils import local_energy_on_the_fly
from vmc_utils import TightBinding_coeff, find_state_coeff
import random
from tqdm import trange 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Obtain_Sampling(initial_state, n_samples, L, pretrain=False, burnin = False,tb_coeff=None):
    Psis = []
    Elocs = []
    state = initial_state
    new_state = state
    for _ in range(n_samples):
        new_state = GlobalSampler(state, L)
        s_onehot = torch.tensor(state_to_onehot(state, L), dtype=torch.float32, device=device).unsqueeze(0)
        new_s_onehot = torch.tensor(state_to_onehot(new_state, L), dtype=torch.float32, device=device).unsqueeze(0)
        psi_val = psi(s_onehot)
        psi_new_val = psi(new_s_onehot)
        #accept_prob = min(1, float((psi_new_val.abs()**2).item() / (psi_val.abs()**2+1e-12).item()))
        if psi_val.abs() < 1e-16:
            accept_prob = 1.0
        else:
            accept_prob = min(1, float((psi_new_val.abs()**2).item() / (psi_val.abs()**2).item()))
        if random.random() < accept_prob:
            state = new_state
            Psis.append(psi_new_val)
        else:
            Psis.append(psi_val)

        if burnin:
            continue
        #if  not pretrain:
        if not pretrain:
            #E_loc = local_energy_from_adjlist(state, psi, basis, basis_dict, H_ind, H_val, device, L)
            E_loc = local_energy_on_the_fly(state, psi, L, t1, t2, J1=J1, J2=J2, device=device)
            Elocs.append(E_loc)
        else:
            Elocs.append(torch.tensor(find_state_coeff(L, state, tb_coeff), device=device))

    return Psis, Elocs, state

def generate_initial_state(L):
    hole = random.randint(0, L-1) #L // 2 # randomly choose a hole position
    L_list = list(range(L))
    L_remove_hole_list = [x for x in L_list if x != hole] # remove the hole site from the list
    upsites = L_remove_hole_list[::2] # choose half of the remaining sites
    initial_state = (hole, tuple(upsites))
    return initial_state


L = 11
t1 = 1.0
t2 = 0.5
J1 = 0.0
J2 = 0.0 #0.81/100

hidden_dim = 32#128 #32
#psi = FCNet(L,hidden_dim=32).to(device)
psi = PhysicalNN(L, hidden_dim=hidden_dim, kernel_size=2, holewave=True).to(device)
#strides = (1,2,3,4,5)#(2,)
#psi = LdepConvStrides(L, nhole=1, hidden_dim=32,kernel_size=3, strides=strides)
#psi = PhysicsLocalLayer(n_freqs=16)
# print number of parameters
n_params = sum(p.numel() for p in psi.parameters() if p.requires_grad)
print(f"Number of parameters in the neural network: {n_params}")

optimizer = optim.Adam(psi.parameters(), lr=1e-3) #1e-3, 1e-2)

save_path = "data/psi_checkpoint.pth"
checkpoint = torch.load(save_path, map_location=device)
psi.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
psi.to(device)
print("Loaded checkpoint.")

# start VMC fine-tuning
# Initialize state
#basis = build_MB_basis(L)
#initial_state = random.choice(basis)
initial_state = generate_initial_state(L)

n_samples = 1000
epochs = 100
burn_in = 1000

_,_, state = Obtain_Sampling(initial_state, burn_in, L, burnin=True) # burn-in

for step in trange(epochs, desc="VMC Sampling"):
    psis, energies, state = Obtain_Sampling(state, n_samples, L)
    # Optimization step every batch
    E_tensor = torch.stack(energies).squeeze()
    psis_tensor = torch.stack(psis).squeeze()
    E_mean = E_tensor.mean()
    loss = 2*torch.mean((E_tensor.detach() - E_mean.detach()) * psis_tensor/psis_tensor.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Use stochastic reconfiguration update (encapsulated)
    #stochastic_reconfiguration_step(psi, energies, logpsis, optimizer, lr=1e-1, reg=1e-3, device=device, noise_sigma=noise_sigma)
    if step % (epochs//20) == 0:
        #print(f"loss: {loss.item()}")
        #E_tensor = torch.stack(energies).squeeze()
        #E_mean = E_tensor.mean()
        print(f"Step {step}: <E> = {E_mean.item():.6f}")
    # if E_mean.item() < -2.5:
    #     print(step)
    #     break  # early stop

print("VMC finished.")

for k in range(5):
    _, energies, state = Obtain_Sampling(state, n_samples, L)
    E_tensor = torch.stack(energies).squeeze()
    E_mean = E_tensor.mean()
    print(f"Next sample {k}: <E> = {E_mean.item():.6f}")


# Save checkpoint
save_path = "data/psi_checkpoint.pth"
torch.save({
    'model_state_dict': psi.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': step,
}, save_path)
print(f"Saved checkpoint to {save_path}")




