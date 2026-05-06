import torch
import torch.nn as nn
import math
import numpy as np

class GeometricPool1d(nn.Module):
    """Wrapper to signed geometric-mean (product) pooling.

    Modes:
      - 'geom': signed geometric mean (exp(mean(log(|x|)))) * sign(prod(x))
                numerically stable via clamp(eps). Zero entries produce zero output.
    Input expected shape: (B, C, L) -> output shape: (B, C)
    """
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x):
        # x: (B, C, L)
        # geom: signed geometric mean across last dim
        # compute sign of product
        sign = torch.prod(torch.sign(x), dim=-1)   # (B, C)
        # compute mean log of absolute values (stable)
        log_abs = torch.log(torch.clamp(torch.abs(x), min=self.eps))
        #log_abs = torch.log(torch.abs(x) + self.eps)
        mean_log = torch.mean(log_abs, dim=-1)    # (B, C)
        geom = torch.exp(mean_log)                # (B, C)
        return sign * geom

# Neural network definition
# fully connected
class FCNet(nn.Module):
    def __init__(self, L, hidden_dim=32):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(4*L, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # flatten input: (batch, L, 4) -> (batch, 4*L)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.tanh(self.fc1(x).pow(3))
        # x = torch.tanh(self.fc2(x).pow(3))
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
        # self.layer4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=self.pad)
        # self.layer5 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=self.pad)
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
        # x = torch.relu(self.layer4(x))  # Residual connection
        # x = torch.relu(self.layer5(x))  # Residual connection
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

class ConvSkipHole(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=32, kernel_size=2, stride=2, 
                 activation='sigmoid',holewave=False,
                 n_freqs=8, amp_hidden=32,nholes=1):
        super(ConvSkipHole, self).__init__()
        self.pad = (kernel_size-1)//2
        self.activation = activation
        self.layer1 = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=self.pad, stride=stride)
        #self.pool = nn.AdaptiveAvgPool1d(1)  # or AdaptiveSumPool1d if available
        self.pool = GeometricPool1d()
        self.finallayer = nn.Linear(hidden_dim, 1)
        self.nholes = nholes

        self.holewave=holewave
        if holewave:
            # plane-wave amplitude module (learnable frequencies + small MLP)
            init_freqs = torch.arange(1, n_freqs + 1, dtype=torch.float32)
            self.freqs = nn.Parameter(init_freqs)              # (n_freqs,)
            # simple linear readout from Fourier features (no hidden layer)
            #self.amp_mlp = nn.Linear(2 * n_freqs, 1, bias=True)
            # small MLP: maps [sin(2pi f s), cos(2pi f s)] -> scalar amplitude
            self.amp_mlp = nn.Sequential(
                nn.Linear(2 * n_freqs, amp_hidden),
                nn.ReLU(),
                nn.Linear(amp_hidden, 1)
            )

    def forward(self, x):
        # remove hole site before convolution
        batch_size, L, C = x.shape
        device = x.device
        # assume exactly one hole per sample encoded as channel 0 == 1
        hole_mask = (x[:, :, 0] == 1).float()  # (B, L)
        hole_pos = torch.argmax(hole_mask, dim=1)  # (B,) — fixed shape, vmap-safe 
        # Build indices [0,1,...,hole-1, hole+1,...,L-1] per sample
        idx = torch.arange(L - self.nholes, device=device).unsqueeze(0).expand(batch_size, -1)  # (B, L-nhole)
        idx = idx + (idx >= hole_pos.unsqueeze(1)).long()  # shift indices past the hole
        x_nohole = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, C))  # (B, L-nhole, C)
        x = x_nohole.permute(0, 2, 1)  # (B, C, L-nhole)

        if self.activation == 'sigmoid':
            x = torch.relu(self.layer1(x)) # shape: (batch, hidden_dim, L//2)
        elif self.activation == 'tanh':
            x = torch.tanh(self.layer1(x).pow(1))
        x = self.pool(x)#.squeeze(-1)  # shape: (batch, hidden_dim)
        #x = torch.tanh(self.finallayer(x).pow(3))
        x = self.finallayer(x)
        #return torch.tanh(x).squeeze(-1)  # output in [-1, 1] # tanh behaves worse
        #return torch.sigmoid(x).squeeze(-1)
        #return x.squeeze(-1)
        # if self.activation == 'sigmoid':
        #     x = torch.sigmoid(x)
        # if self.activation == 'tanh':
        #     x = torch.tanh(x)
            #x = x
        
        if self.holewave:
            # --- compute learned plane-wave amplitude from hole position ---
            # normalized position in [0,1)
            s = hole_idx.float() / float(L)                     # (B,)
            freqs = self.freqs.unsqueeze(0).to(device=device)   # (1, n_freqs)
            phi = 2.0 * math.pi * (freqs * s.unsqueeze(1))      # (B, n_freqs)
            feats = torch.cat([torch.sin(phi), torch.cos(phi)], dim=1)  # (B, 2*n_freqs)
            amp = self.amp_mlp(feats).squeeze(-1)               # (B,) raw amplitude (can be positive/negative)
            x = x * amp

        return x.squeeze(-1)

class ConvHoleTanh(nn.Module):
    """
    Convolutional net that appends a tanh-domain-wall channel computed from the hole
    position. The encoding is tanh((site_index - hole_pos) / xi) so xi controls the
    width of the domain wall (in sites). Set xi small -> sharper sign change at hole.
    """
    def __init__(self, in_channels=4, hidden_dim=32, kernel_size=2, stride=2, xi=1.0):
        super(ConvHoleTanh, self).__init__()
        self.xi = float(xi)
        self.pad = (kernel_size-1)//2
        self.layer1 = nn.Conv1d(in_channels+1, hidden_dim, kernel_size, padding=self.pad, stride=stride)
        self.pool = nn.AdaptiveAvgPool1d(1)  # or AdaptiveSumPool1d if available
        self.finallayer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # remove hole site before convolution
        batch_size, L, C = x.shape
        device = x.device
        # assume exactly one hole per sample encoded as channel 0 == 1
        # get hole indices (shape: (B,))
        hole_idx = torch.argmax(x[:, :, 0], dim=1)
        idx = torch.arange(L, device=device).unsqueeze(0)   # (1, L)
        rel = (idx - hole_idx.unsqueeze(1)).float()        # (B, L)
        # tanh domain wall encoding: shape (B, L, 1)
        # note: xi in units of sites; smaller xi -> sharper wall
        tanh_enc = torch.tanh(rel / (self.xi + 1e-9)).unsqueeze(-1)
        # concatenate encoding as extra channel -> (B, L, C+1)
        x_enc = torch.cat([x, tanh_enc], dim=2)
        # x shape: (batch, L, 4) -> (batch, 4, L)
        x = x_enc.permute(0, 2, 1)
        x = torch.relu(self.layer1(x)) # shape: (batch, hidden_dim, L//2)
        x = self.pool(x).squeeze(-1)  # shape: (batch, hidden_dim)
        x = self.finallayer(x)
        #return torch.tanh(x).squeeze(-1)  # output in [-1, 1] # tanh behaves worse
        return torch.sigmoid(x).squeeze(-1)
        #return x.squeeze(-1)

class PhysicsLocalLayer(nn.Module):
    def __init__(self, n_freqs=8, amp_hidden=32):
        super().__init__()
        # learnable continuous frequencies (initialized as integers)
        init_freqs = torch.arange(1, n_freqs + 1, dtype=torch.float32)
        self.freqs = nn.Parameter(init_freqs)              # shape (n_freqs,)
        # small MLP: maps [sin(2pi f s), cos(2pi f s)] -> scalar amplitude
        # self.amp_mlp = nn.Sequential(
        #     nn.Linear(2 * n_freqs, amp_hidden),
        #     nn.ReLU(),
        #     nn.Linear(amp_hidden, 1)
        # )
        self.amp_mlp = nn.Linear(2 * n_freqs, 1, bias=True)
        
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
    


class VisionTransformer1D(nn.Module):
    """
    Vision-Transformer style model adapted to 1D sequences.
    - Splits the length into non-overlapping patches (patch_size).
    - Uses a cls token + positional embeddings and a standard TransformerEncoder.
    - Returns a scalar per sample (default sigmoid activation to match other nets).
    This version is independent of the compile-time input length L and computes
    positional encodings at runtime.
    """
    def __init__(self, in_channels=4, patch_size=2, embed_dim=64,
                 num_layers=4, num_heads=4, mlp_dim=128, dropout=0.1,
                 activation='sigmoid'):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.activation = activation

        # patch embedding using Conv1d (stride = kernel = patch_size)
        self.patch_embed = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        patch_in_dim = in_channels * patch_size
        self.patch_mlp = nn.Sequential(
            nn.Linear(patch_in_dim, embed_dim),
            #nn.ReLU(),
            #nn.Linear(embed_dim, embed_dim)
        )
        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        L = 1000  # dummy length to compute number of patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + L, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=mlp_dim, dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # head -> scalar
        self.head = nn.Linear(embed_dim, 1)

        # init
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def sinusoidal_positional_encoding(self, seq_len, device, dtype):
        """Return (1, seq_len, embed_dim) sinusoidal positional encodings.
        We zero the CLS token's positional embedding (position 0)."""
        pe = torch.zeros(seq_len, self.embed_dim, device=device, dtype=dtype)
        position = torch.arange(0, seq_len, device=device, dtype=dtype).unsqueeze(1)  # (seq_len,1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device=device, dtype=dtype) *
                             -(math.log(10000.0) / self.embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_len, embed_dim)
        # zero CLS position (index 0) so cls token doesn't get positional bias by default
        if seq_len > 0:
            pe[:, 0, :] = 0.0
        return pe
    
    def forward(self, x):
        # x: (B, L, C)
        batch_size, L, C = x.shape
        device = x.device
        dtype = x.dtype

        # assume exactly one hole per sample encoded as channel 0 == 1
        # get hole indices (shape: (B,))
        hole_idx = torch.argmax(x[:, :, 0], dim=1)
        # build mask: True for positions that are NOT the hole
        idx = torch.arange(L, device=device)
        mask = idx.unsqueeze(0) != hole_idx.unsqueeze(1)  # (B, L)
        # select non-hole positions and reshape -> (B, L-1, C)
        x_nohole = x[mask].view(batch_size, L - 1, C)

        # to (B, C, L_patched)
        x_c = x_nohole.permute(0, 2, 1)
        patches = self.patch_embed(x_c)                # (B, embed_dim, n_patches)
        n_patches = patches.shape[2]
        patches = patches.permute(0, 2, 1)             # (B, n_patches, embed_dim)
        # # Build non-overlapping patches by flattening patch_size sites into one vector
        # n_patches = (L - 1) // self.patch_size
        # if n_patches > 0:
        #     # truncate remainder sites that don't form a full patch
        #     x_trim = x_nohole[:, : n_patches * self.patch_size, :]                    # (B, n_patches*patch_size, C)
        #     patches = x_trim.view(batch_size, n_patches, self.patch_size * C)         # (B, n_patches, patch_size*C)
        #     # MLP operates on last dim -> (B, n_patches, embed_dim)
        #     patches = self.patch_mlp(patches)

        # prepend cls token
        cls = self.cls_token.expand(batch_size, -1, -1)         # (B,1,embed_dim)
        seq = torch.cat([cls, patches], dim=1)         # (B, 1+n_patches, embed_dim)

        # # add runtime positional embeddings
        # pos = self.sinusoidal_positional_encoding(seq.size(1), device, dtype)  # (1, seq_len, embed_dim)
        # seq = seq + pos
        # add positional embeddings (supports smaller L via slicing)
        #seq = seq + self.pos_embed[:, : seq.size(1), :]

        # transformer -> take cls token
        seq = self.transformer(seq)
        seq = self.norm(seq)
        cls_out = seq[:, 0, :]                          # (B, embed_dim)
        out = self.head(cls_out).squeeze(-1)           # (B,)

        if self.activation == 'sigmoid':
            return torch.sigmoid(out)
        if self.activation == 'tanh':
            return torch.tanh(out)
        return out
    

class LdepConvSkipHole(nn.Module):
    def __init__(self, L, in_channels=4, hidden_dim=32, kernel_size=2, stride=2, 
                 holewave=False,
                 n_freqs=8, amp_hidden=32):
        super(LdepConvSkipHole, self).__init__()
        self.pad = (kernel_size-1)//2
        self.layer1 = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=self.pad, stride=stride)
        self.pool = GeometricPool1d()
        #self.fc1 = nn.Linear(hidden_dim*(L//2)+L, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim+L, hidden_dim)
        self.finallayer = nn.Linear(hidden_dim, 1)

        self.holewave=holewave

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
        x = torch.tanh(self.layer1(x)) # shape: (batch, hidden_dim, L//2)
        #x = x.view(batch_size, -1)  # flatten
        x = self.pool(x).squeeze(-1)  # shape: (batch, hidden_dim)
        # add hole position encoding to flattened vector
        # make hole position one-hot vector
        hole_vec = torch.zeros(batch_size, L, device=device)
        if self.holewave:
            hole_vec[torch.arange(batch_size), hole_idx] = 1.0
        x = torch.cat([x, hole_vec], dim=1)
        x = torch.tanh(self.fc1(x))  # shape: (batch, hidden_dim)
        x = self.finallayer(x)
        #x = torch.tanh(x)
        
        return x.squeeze(-1)
    

class LdepConv2d(nn.Module):
    def __init__(self, L, in_channels=4, hidden_dim=1024, kernel_size=3):
        super(LdepConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size-1) // 2
        dim1 = int(np.sqrt(hidden_dim))
        while hidden_dim % dim1 != 0:
            dim1 -= 1
        dim1p = hidden_dim // dim1
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(L*in_channels, hidden_dim),
            nn.Tanh(),
            nn.Unflatten(1, (1, dim1, dim1p)), # Reshape to (batch_size, channels=1, height=dim1, width=dim1p)
            nn.Conv2d(1, 4, kernel_size=kernel_size, padding=self.padding),  # Conv2d layer
            nn.Tanh(),
            nn.Flatten(),  # Flatten back to 1D
            nn.Linear(dim1 * dim1p * 4, hidden_dim // 2),  # Adjust dimensions for Conv2d output
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)  # 1 for output
        )
        # self.layers = nn.Sequential(
        #     nn.Conv1d(in_channels, 32, kernel_size=kernel_size, padding=self.padding),
        #     nn.Tanh(),
        #     nn.Flatten(),
        #     nn.Linear(32*L, hidden_dim),  # Adjust dimensions for Conv2d output
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, 1)  # 1 for output
        # )

    def forward(self, x):
        # x shape: (batch, L, 4) -> (batch, 4, L)
        x = x.permute(0, 2, 1)
        x = self.layers(x)
        return x.squeeze(-1)
    



class LdepConvHoles(nn.Module):
    def __init__(self, L, nhole=2, in_channels=4, hidden_dim=32, kernel_size=3, stride=2):
        super(LdepConvHoles, self).__init__()
        self.pad = (kernel_size-1)//2
        self.layer1 = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=self.pad, stride=stride)
        self.pool = GeometricPool1d()
        #self.fc1 = nn.Linear(hidden_dim*(L//2)+L, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim+L, hidden_dim)
        self.finallayer = nn.Linear(hidden_dim, 1)
        self.nhole = nhole

    def forward(self, x):
        # remove hole site before convolution
        batch_size, L, C = x.shape
        device = x.device
        # obtain  holes per sample encoded as channel 0 == 1
        # get hole indices (shape: (B,L,4)) holes are marked by channel 0 == 1 
        hole_mask = (x[:, :, 0] == 1)   # (B, L) boolean mask
        # vectorized indices: (N,2) -> (batch_idx, pos)
        idxs = torch.nonzero(hole_mask, as_tuple=False)    # (B*nhole, 2)
        # positions grouped by batch must be exactly nhole per batch
        hole_pos = idxs[:, 1].view(batch_size, self.nhole).to(device)  # (B, nhole)
        # build keep-mask (True = keep non-hole positions): vectorized
        hole_positions_mask = torch.zeros(batch_size, L, dtype=torch.bool, device=device)
        rows = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.nhole)  # (B, nhole)
        hole_positions_mask[rows, hole_pos] = True

        mask = ~hole_positions_mask  # (B, L) True for non-hole sites
        # select non-hole positions and reshape -> (B, L-1, C)
        x_nohole = x[mask].view(batch_size, L - self.nhole, C)
        # x shape: (batch, L, 4) -> (batch, 4, L)
        x = x_nohole.permute(0, 2, 1)
        x = torch.tanh(self.layer1(x).pow(5)) # shape: (batch, hidden_dim, L//2)
        #x = x.view(batch_size, -1)  # flatten
        x = self.pool(x).squeeze(-1)  # shape: (batch, hidden_dim)
        # add hole position encoding to flattened vector
        # make hole position one-hot vector
        hole_vec = torch.zeros(batch_size, L, device=device)
        # set the positions of holes to be 1 in the hole_vec of size L
        hole_vec.scatter_(1, hole_pos, 1.0)
        x = torch.cat([x, hole_vec], dim=1)
        x = torch.tanh(self.fc1(x).pow(3))  # shape: (batch, hidden_dim)
        x = self.finallayer(x)
        #x = torch.tanh(x)
        
        return x.squeeze(-1)
    

class LdepConvStrides(nn.Module):
    def __init__(self, L, nhole=2, in_channels=4, hidden_dim=32, kernel_size=3, strides=(1,2,3,4)):
        super(LdepConvStrides, self).__init__()
        self.L = L
        self.nhole = nhole
        self.pad = (kernel_size-1)//2
        # create parallel conv branches with different strides
        self.strides = tuple(strides)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=self.pad, stride=s)
            for s in self.strides
        ])
        # one pooling module per branch (keeps feature dim = hidden_dim)
        self.pools = nn.ModuleList([GeometricPool1d() for _ in self.strides])
        #self.pools = nn.ModuleList([nn.AdaptiveAvgPool1d(5) for _ in self.strides])

        # fc input: concatenated pooled features from each branch + L-sized hole one-hot
        self.fc1 = nn.Linear(hidden_dim * len(self.strides) + L, hidden_dim)
        #self.fc1 = nn.Linear(hidden_dim * len(self.strides)*5 + L, hidden_dim)
        self.finallayer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, L, C = x.shape
        device = x.device
        dtype = x.dtype

        # obtain holes per sample encoded as channel 0 == 1
        hole_mask = (x[:, :, 0] == 1)   # (B, L) boolean mask
        idxs = torch.nonzero(hole_mask, as_tuple=False)    # (B*nhole, 2)
        hole_pos = idxs[:, 1].view(batch_size, self.nhole).to(device)  # (B, nhole)

        # build mask for non-hole positions
        hole_positions_mask = torch.zeros(batch_size, L, dtype=torch.bool, device=device)
        rows = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.nhole)  # (B, nhole)
        hole_positions_mask[rows, hole_pos] = True
        mask = ~hole_positions_mask  # True for non-hole sites

        # select non-hole positions -> (B, L-nhole, C)
        x_nohole = x[mask].view(batch_size, L - self.nhole, C)
        x_perm = x_nohole.permute(0, 2, 1)  # (B, C, L-nhole)

        # apply each conv branch and pool (each yields (B, hidden_dim))
        feats = []
        for conv, pool in zip(self.convs, self.pools):
            y = torch.tanh(conv(x_perm))
            #y = torch.relu(conv(x_perm))
            y = pool(y).squeeze(-1)  # (B, hidden_dim,5)
            #y = y.flatten(1)
            feats.append(y)
        feats_cat = torch.cat(feats, dim=1)  # (B, hidden_dim * n_branches)
        #print(feats_cat.shape)
        # hole one-hot vector of length L
        hole_vec = torch.zeros(batch_size, L, device=device, dtype=dtype)
        hole_vec.scatter_(1, hole_pos, 1.0)

        # combine branch features + hole encoding
        x_comb = torch.cat([feats_cat, hole_vec], dim=1)  # (B, hidden_dim*n_branches + L)
        x_out = torch.tanh(self.fc1(x_comb))
        #x_out = torch.relu(self.fc1(x_comb))
        x_out = self.finallayer(x_out)

        return x_out.squeeze(-1)
    


class HoleOnLocalRule(nn.Module):
    """
    Wrap a trained LocalRule and add L learnable amplitudes (one per hole position).
    Output = local_rule(x) * amplitude[hole_index]
    """
    def __init__(self, local_rule: nn.Module, L: int, init_val: float = 1.0, hidden_dim: int = 32):
        super(HoleOnLocalRule, self).__init__()
        self.local_rule = local_rule
        self.local_rule.eval()
        for p in self.local_rule.parameters():
            p.requires_grad = False
        self.L = L
        # one scalar amplitude per hole position
        #self.amplitudes = nn.Parameter(torch.full((L,), float(init_val)))
        self.fc = nn.Linear(L, hidden_dim)
        self.finallayer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, L, C), assume channel 0 encodes hole (one-hot)
        # get local sign/amplitude from learned local rule (expects full x)
        with torch.no_grad():
            local_out = self.local_rule(x)  # shape (B,)
        # get hole indices
        #hole_idx = torch.argmax(x[:, :, 0], dim=1)  # shape (B,)
        # index amplitudes per sample
        #amp = self.amplitudes[hole_idx]  # shape (B,)
        #return local_out * amp

        batch_size, L, C = x.shape
        device = x.device
        # hole_mask = (x[:, :, 0] == 1)   # (B, L) boolean mask
        # idxs = torch.nonzero(hole_mask, as_tuple=False)    # (B*nhole, 2)
        # hole_pos = idxs[:, 1].view(batch_size, 1).to(device)  # (B, nhole)
        # hole_vec = torch.zeros(batch_size, L, device=device)
        # hole_vec[torch.arange(batch_size), hole_pos[:,0]] = 1.0

        # obtain holes per sample encoded as channel 0 == 1
        hole_mask = (x[:, :, 0] == 1).float()  # (B, L)
        hole_vec = hole_mask  # already (B, L) with 1.0 at hole position

        yhole = self.fc(hole_vec)
        yhole = torch.tanh(yhole.pow(3))
        yhole = self.finallayer(yhole).squeeze(-1)
        return local_out * yhole
    

class LdepConvPlusFC(nn.Module):
    def __init__(self, L, nhole=1, in_channels=4, Conv_dim=32, kernel_size=3, stride=2, ifFC=False,FC_dim=32):
        super(LdepConvPlusFC, self).__init__()
        self.L = L
        self.nhole = nhole
        self.pad = (kernel_size-1)//2
        self.ifFC = ifFC
        # create parallel conv branches with different strides
        self.conv = nn.Conv1d(in_channels, Conv_dim, kernel_size=kernel_size, padding=self.pad, stride=stride)
        
        # one pooling module per branch (keeps feature dim = hidden_dim)
        self.pool = GeometricPool1d()
        #self.maxpool = nn.AdaptiveMaxPool1d(1)
        #self.pools = nn.ModuleList([nn.AdaptiveAvgPool1d(5) for _ in self.strides])

        # add simple conv branch
        #self.conv_0 = nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=self.pad, stride=1)
        # fc input: concatenated pooled features from each branch + L-sized hole one-hot
        #self.fc_2 = nn.Linear(2*hidden_dim+L, 2*hidden_dim)
        #self.fc_2 = nn.Linear(hidden_dim+32+L, 32+hidden_dim)
        #self.fc_2 = nn.Linear(32+L, 32)
        #self.fc1 = nn.Linear(hidden_dim * len(self.strides)*5 + L, hidden_dim)
        #self.finallayer = nn.Linear(hidden_dim, 1)
        #self.finallayer = nn.Linear(32+hidden_dim, 1)

        if self.ifFC:
            self.fc_squence = nn.Sequential(
                nn.Flatten(),
                nn.Linear(L*in_channels, FC_dim),
                nn.ReLU(),
                nn.Linear(FC_dim, FC_dim),
                nn.ReLU()
            )
            self.fc_2 = nn.Linear(Conv_dim+FC_dim+L, Conv_dim)
            self.finallayer = nn.Linear(Conv_dim, 1)
        else:
            self.fc_2 = nn.Linear(Conv_dim+L, Conv_dim)
            self.finallayer = nn.Linear(Conv_dim, 1)


    def forward(self, x):
        batch_size, L, C = x.shape
        device = x.device
        dtype = x.dtype

        # obtain holes per sample encoded as channel 0 == 1
        hole_mask = (x[:, :, 0] == 1)   # (B, L) boolean mask
        idxs = torch.nonzero(hole_mask, as_tuple=False)    # (B*nhole, 2)
        hole_pos = idxs[:, 1].view(batch_size, self.nhole).to(device)  # (B, nhole)

        # build mask for non-hole positions
        hole_positions_mask = torch.zeros(batch_size, L, dtype=torch.bool, device=device)
        rows = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.nhole)  # (B, nhole)
        hole_positions_mask[rows, hole_pos] = True
        mask = ~hole_positions_mask  # True for non-hole sites

        # select non-hole positions -> (B, L-nhole, C)
        x_nohole = x[mask].view(batch_size, L - self.nhole, C)
        x_perm = x_nohole.permute(0, 2, 1)  # (B, C, L-nhole)

        # apply each conv branch and pool (each yields (B, hidden_dim))
        feats = []
        y1 = torch.tanh(self.conv(x_perm).pow(5))
        y1 = self.pool(y1).squeeze(-1)  # (B, hidden_dim,5)
        feats.append(y1)
        hole_vec = torch.zeros(batch_size, L, device=device)
        hole_vec[torch.arange(batch_size), hole_pos[:,0]] = 1.0
        feats.append(hole_vec)
        if self.ifFC:
            y2 = self.fc_squence(x)
            feats.append(y2)

        # y3 = torch.tanh(self.conv_0(x_perm))
        # y3 = self.maxpool(y3).squeeze(-1)
        # feats.append(y3)

        feats_cat = torch.cat(feats, dim=1)  # (B, hidden_dim * n_branches)
    
        # combine branch features + hole encoding
        #x_out = torch.tanh(self.fc_2(feats_cat))
        #x_out = torch.relu(self.fc1(x_comb))
        #x_out = self.finallayer(x_out)

        x_out = torch.tanh(self.fc_2(feats_cat).pow(3)) # best pow(3)
        #x_out = torch.relu(self.fc_2(feats_cat)) 
        x_out = self.finallayer(x_out)

        return x_out.squeeze(-1)
    


class LocalRule(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=32, kernel_size=2, stride=2, nholes=1):
        super(LocalRule, self).__init__()
        self.nholes = nholes
        self.pad = (kernel_size-1)//2
        self.layer1 = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=self.pad, stride=stride)
        # 1x1 conv to produce sign and value per pair (classes: +1, -1, 0)
        self.classifier = nn.Conv1d(hidden_dim, 2, kernel_size=1)
        # produce 3 logits per pair: [p_updown, p_downup, p_other]
        #self.classifier = nn.Conv1d(hidden_dim, 3, kernel_size=1)
        self.pool = GeometricPool1d()
        self.finallayer = nn.Linear(hidden_dim, 1)
        #self.finallayer = nn.Linear(1, 1)

    def forward(self, x):
        # remove hole site before convolution
        batch_size, L, C = x.shape
        device = x.device
        # assume exactly one hole per sample encoded as channel 0 == 1
        # get hole indices (shape: (B,))
        # hole_idx = torch.argmax(x[:, :, 0], dim=1)
        # # build mask: True for positions that are NOT the hole
        # idx = torch.arange(L, device=device)
        # mask = idx.unsqueeze(0) != hole_idx.unsqueeze(1)  # (B, L)
        # # select non-hole positions and reshape -> (B, L-1, C)
        # x_nohole = x[mask].view(batch_size, L - 1, C)
        # # x shape: (batch, L, 4) -> (batch, 4, L)
        # x = x_nohole.permute(0, 2, 1)

        # assume exactly one hole per sample encoded as channel 0 == 1
        hole_mask = (x[:, :, 0] == 1).float()  # (B, L)
        hole_pos = torch.argmax(hole_mask, dim=1)  # (B,) — fixed shape, vmap-safe 
        # Build indices [0,1,...,hole-1, hole+1,...,L-1] per sample
        idx = torch.arange(L - self.nholes, device=device).unsqueeze(0).expand(batch_size, -1)  # (B, L-nhole)
        idx = idx + (idx >= hole_pos.unsqueeze(1)).long()  # shift indices past the hole
        x_nohole = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, C))  # (B, L-nhole, C)
        x = x_nohole.permute(0, 2, 1)  # (B, C, L-nhole)

        x = torch.tanh(self.layer1(x).pow(5)) # .pow(3)
        x = self.pool(x).squeeze(-1)  # shape: (batch, hidden_dim)
        x = self.finallayer(x)

        return x.squeeze(-1)