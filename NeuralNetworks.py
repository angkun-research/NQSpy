import torch
import torch.nn as nn

# Neural network definition
# fully connected
class FCNet(nn.Module):
    def __init__(self, L, hidden_dim=32):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(4*L, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)
        #return torch.tanh(x).squeeze(-1)  # tanh behaves worse
        #return 2*torch.sigmoid(x).squeeze(-1) - 1  # output in [0, 1]
    
# Convolutional for arbitrary L
class ConvNet(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=32, kernel_size=3, num_layers=5, dilation_growth=2):
        super(ConvNet, self).__init__()
        self.pad = (kernel_size-1)//2
        self.layer1 = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=self.pad)
        self.layer2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=self.pad)
        self.layer3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=self.pad)
        self.layer4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=self.pad)
        self.layer5 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=self.pad)
        #self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        # layers = []
        # dilation = 1
        # for i in range(num_layers):
        #     layers.append(
        #         nn.Conv1d(
        #             in_channels if i == 0 else hidden_dim,
        #             hidden_dim,
        #             kernel_size,
        #             padding=dilation * self.pad,
        #             dilation=dilation
        #         )
        #     )
        #     dilation *= dilation_growth
        # self.convs = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # or AdaptiveSumPool1d if available
        self.finallayer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, L, 4) -> (batch, 4, L)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x)+x)  # Residual connection
        x = torch.relu(self.layer3(x)+x)  # Residual connection
        x = torch.relu(self.layer4(x))  # Residual connection
        x = torch.relu(self.layer5(x))  # Residual connection
        #x = torch.relu(self.layer2(x))  # Residual connection
        # for conv in self.convs:
        #     residual = x
        #     x = torch.relu(conv(x))
        #     # Match shapes for residual (crop if needed)
        #     if x.shape == residual.shape:
        #         x = x + residual
        x = self.pool(x).squeeze(-1)  # shape: (batch, hidden_dim)
        x = self.finallayer(x)
        #return torch.tanh(x).squeeze(-1)  # output in [-1, 1] # tanh behaves worse
        return torch.sigmoid(x).squeeze(-1)

    
class PhysicsLocalLayer(nn.Module):
    def __init__(self, n_freqs=8, amp_hidden=32):
        super().__init__()
        # learnable continuous frequencies (initialized as integers)
        init_freqs = torch.arange(1, n_freqs + 1, dtype=torch.float32)
        self.freqs = nn.Parameter(init_freqs)              # shape (n_freqs,)
        # small MLP: maps [sin(2pi f s), cos(2pi f s)] -> scalar amplitude
        self.amp_mlp = nn.Sequential(
            nn.Linear(2 * n_freqs, amp_hidden),
            nn.ReLU(),
            nn.Linear(amp_hidden, 1)
        )
        
    def forward(self, x):
        # x: (batch, L, 4) one-hot
        # x: (batch, L, 4) one-hot; channel 0 assumed to mark the hole position
        batch_size, self.L, C = x.shape
        device = x.device
        dtype = x.dtype

        signs = torch.ones(batch_size,device=device, dtype=dtype)
        for b in range(batch_size):
            config = x[b]  # shape: (L, 4)s
            sign = 1
            idx = 0
            while idx < self.L and config[idx, 0] == 0 and config[idx+1, 0] == 0:
                left = config[idx]
                right = config[idx + 1]
                sign *= (left[2]*right[1] - left[1]*right[2])
                idx += 2
            if config[idx, 0] == 1:
                idx += 1
            else:
                left = config[idx]
                right = config[idx + 2]
                sign *= (left[2]*right[1] - left[1]*right[2])
                idx += 3
            while idx < self.L - 1:
                left = config[idx]
                right = config[idx + 1]
                sign *= (left[2]*right[1] - left[1]*right[2])
                idx += 2
            signs[b] = sign

        # --- learnable amplitude from hole position (system-size independent) ---
        # find hole position (assumes exactly one hole, channel 0 is hole indicator)
        hole_pos = torch.argmax(x[:, :, 0], dim=1)          # (B,)
        s = hole_pos.float() / float(self.L)                     # normalized position in [0,1)
        # build Fourier features with learnable freqs
        freqs = self.freqs.unsqueeze(0).to(device=device)  # (1, n_freqs)
        phi = 2.0 * torch.pi * (freqs * s.unsqueeze(1))    # (B, n_freqs)
        feats = torch.cat([torch.sin(phi), torch.cos(phi)], dim=1)  # (B, 2*n_freqs)
        amp = self.amp_mlp(feats).squeeze(-1)               # (B,)  raw amplitude

        # combine physics sign and learned amplitude
        out = signs.to(device=device) * amp
        return out


class PhysicsConvNet(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=32, kernel_size=3, num_layers=5, dilation_growth=2):
        super().__init__()
        layers = []
        dilation = 1
        for i in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size,
                    padding=dilation * (kernel_size // 2),
                    dilation=dilation
                )
            )
            dilation *= dilation_growth
        self.convs = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, L, 4) -> (batch, 4, L)
        x = x.permute(0, 2, 1)
        for conv in self.convs:
            residual = x
            x = torch.relu(conv(x))
            # Match shapes for residual (crop if needed)
            if x.shape == residual.shape:
                x = x + residual
        x = self.pool(x).squeeze(-1)
        x = self.final(x)
        return torch.sigmoid(x).squeeze(-1) #x.squeeze(-1)
    

class ConvTransformer(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=32, kernel_size=3, num_layers=3, nhead=4):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=(kernel_size-1)//2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, L, 4) -> (batch, 4, L)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv_in(x))
        # Transformer expects input shape (batch, L, hidden_dim)
        x = x.permute(0, 2, 1)  # (batch, L, hidden_dim)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)  # back to (batch, hidden_dim, L)
        x = self.pool(x).squeeze(-1)
        x = self.final(x)
        return x.squeeze(-1)
    
class TransformerConv(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=32, num_layers=3, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(in_channels, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, L, 4)
        x = self.embedding(x)  # (batch, L, hidden_dim)
        x = self.transformer(x)  # (batch, L, hidden_dim)
        x = x.permute(0, 2, 1)  # (batch, hidden_dim, L)
        x = self.conv1d(x)
        x = self.pool(x).squeeze(-1)  # (batch, hidden_dim) 
        x = self.final(x)  # (batch, 1)
        return torch.sigmoid(x).squeeze(-1) #x.squeeze(-1)


class MultiKernelConvNet(nn.Module):
    def __init__(self, in_channels, hidden_dim=32, kernel_sizes=(2, 3, 4), num_layers=4):
        super().__init__()
        # Each kernel size gets its own Conv1d layer
        self.input_convs = nn.ModuleList([
            nn.Conv1d(in_channels, hidden_dim, k, padding='same')
            for k in kernel_sizes
        ])
        # After concatenation, total hidden dim is hidden_dim * num_kernels
        total_hidden = hidden_dim * len(kernel_sizes)
        # Stack more Conv1d layers (with kernel size 3 here, can be changed)
        self.hidden_layers = nn.ModuleList([
            nn.Conv1d(total_hidden, total_hidden, 3, padding=1)
            for _ in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.final = nn.Linear(total_hidden, 1)

    def forward(self, x):
        # x: (batch, L, C) -> (batch, C, L)
        x = x.permute(0, 2, 1)
        # Apply all input convolutions and concatenate along channel dim
        feats = [torch.relu(conv(x)) for conv in self.input_convs]
        x = torch.cat(feats, dim=1)  # (batch, hidden_dim * num_kernels, L)
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.pool(x).squeeze(-1)
        x = self.final(x)
        return torch.sigmoid(x).squeeze(-1)