import torch


# ----------------------------------------------------------------------
# State conversions
# ----------------------------------------------------------------------

def states_to_spin_tensor(states, L, device):
    """
    Convert Python states to a tensor representation.

    Encoding:
         0 = hole
         1 = up spin
        -1 = down spin

    Input:
        states: list of (hole, up_sites), or an existing tensor (B, L)

    Returns:
        int8 tensor with shape (B, L)
    """
    if torch.is_tensor(states):
        if states.ndim != 2 or states.shape[1] != L:
            raise ValueError(
                f"Tensor states must have shape (batch, {L}), "
                f"received {tuple(states.shape)}"
            )

        return states.to(
            device=device,
            dtype=torch.int8,
        )

    batch_size = len(states)

    spins = torch.full(
        (batch_size, L),
        -1,
        device=device,
        dtype=torch.int8,
    )

    rows = torch.arange(batch_size, device=device)

    holes = torch.as_tensor(
        [state[0] for state in states],
        device=device,
        dtype=torch.long,
    )

    up_sites = torch.as_tensor(
        [state[1] for state in states],
        device=device,
        dtype=torch.long,
    )

    spins[rows, holes] = 0

    up_rows = rows[:, None].expand_as(up_sites)
    spins[up_rows, up_sites] = 1

    return spins


def spin_tensor_to_onehot(states, dtype=torch.float32):
    """
    Convert spin-tensor states to neural-network one-hot inputs.

    Input:
        states: (B, L) or (L,) tensor

    Returns:
        Tensor with shape (B, L, 4)
    """
    if states.ndim == 1:
        states = states.unsqueeze(0)

    empty_channel = torch.zeros_like(
        states,
        dtype=torch.bool,
    )

    onehot = torch.stack(
        (
            states == 0,      # hole
            states == 1,      # up
            states == -1,     # down
            empty_channel,    # unused
        ),
        dim=-1,
    )

    return onehot.to(dtype=dtype)


def spin_tensor_to_states(states):
    """
    Convert GPU spin tensors back to the original Python representation.

    This performs a GPU-to-CPU transfer, so use it only for output,
    diagnostics, or compatibility—not inside the sampling loop.
    """
    single_state = states.ndim == 1

    if single_state:
        states = states.unsqueeze(0)

    states_cpu = states.detach().cpu()
    result = []

    for row in states_cpu:
        hole = int(
            torch.nonzero(
                row == 0,
                as_tuple=False,
            )[0, 0]
        )

        up_sites = tuple(
            torch.nonzero(
                row == 1,
                as_tuple=False,
            ).flatten().tolist()
        )

        result.append((hole, up_sites))

    return result[0] if single_state else result


# ----------------------------------------------------------------------
# Internal GPU helpers
# ----------------------------------------------------------------------

def _swap_sites_gpu(states, left, right):
    """
    Swap one pair of sites in every walker.

    states: (B, L)
    left:   (B,)
    right:  (B,)
    """
    rows = torch.arange(
        states.shape[0],
        device=states.device,
    )

    proposed = states.clone()

    left_values = states[rows, left]
    right_values = states[rows, right]

    proposed[rows, left] = right_values
    proposed[rows, right] = left_values

    return proposed


def _choose_valid_gpu(valid):
    """
    Uniformly select one valid entry from each row of a Boolean mask.

    Returns:
        selected_index: (B,)
        has_valid:      (B,)
    """
    scores = torch.rand(
        valid.shape,
        device=valid.device,
    )

    scores.masked_fill_(~valid, -1.0)

    selected_index = scores.argmax(dim=1)
    has_valid = valid.any(dim=1)

    return selected_index, has_valid


def _swapped_candidates_gpu(states, left, right):
    """
    Construct multiple swapped candidates for every walker.

    states: (B, L)
    left:   (B, K)
    right:  (B, K)

    Returns:
        candidates: (B, K, L)
    """
    batch_size, L = states.shape
    number_of_moves = left.shape[1]

    candidates = (
        states[:, None, :]
        .expand(batch_size, number_of_moves, L)
        .clone()
    )

    batch_indices = (
        torch.arange(batch_size, device=states.device)[:, None]
        .expand_as(left)
    )

    move_indices = (
        torch.arange(number_of_moves, device=states.device)[None, :]
        .expand_as(left)
    )

    left_values = torch.gather(states, 1, left)
    right_values = torch.gather(states, 1, right)

    candidates[
        batch_indices,
        move_indices,
        left,
    ] = right_values

    candidates[
        batch_indices,
        move_indices,
        right,
    ] = left_values

    return candidates


# ----------------------------------------------------------------------
# GPU samplers
# ----------------------------------------------------------------------

def GlobalSampler_gpu(states, L=None):
    """
    Batched GPU equivalent of GlobalSampler.

    A valid move either:
      - swaps the hole with another site, or
      - swaps two opposite spins.

    Args:
        states: int8 tensor with shape (B, L)
        L: optional system size

    Returns:
        proposed states with shape (B, L)
    """
    if states.ndim != 2:
        raise ValueError(
            "states must have shape (n_walkers, L)"
        )

    if L is None:
        L = states.shape[1]

    site1_all, site2_all = torch.triu_indices(
        L,
        L,
        offset=1,
        device=states.device,
    )

    value1 = states[:, site1_all]
    value2 = states[:, site2_all]

    valid = (
        (value1 == 0)
        | (value2 == 0)
        | (value1 * value2 == -1)
    )

    choice, has_valid = _choose_valid_gpu(valid)

    site1 = site1_all[choice]
    site2 = site2_all[choice]

    proposed = _swap_sites_gpu(
        states,
        site1,
        site2,
    )

    # Normally every physical state has valid moves. This fallback keeps
    # the old state if malformed input has no valid move.
    return torch.where(
        has_valid[:, None],
        proposed,
        states,
    )


def BalancedSampler_gpu(states, L=None):
    """
    Batched GPU equivalent of BalancedSampler.

    With probability 1/2:
      - move the hole to a nearest neighbor.

    Otherwise:
      - swap nearest-neighbor opposite spins, or
      - swap opposite spins separated by a hole.
    """
    if states.ndim != 2:
        raise ValueError(
            "states must have shape (n_walkers, L)"
        )

    device = states.device
    batch_size, actual_L = states.shape

    if L is None:
        L = actual_L

    rows = torch.arange(
        batch_size,
        device=device,
    )

    hole = (
        (states == 0)
        .to(torch.long)
        .argmax(dim=1)
    )

    # --------------------------------------------------------------
    # Hole moves
    # --------------------------------------------------------------
    hole_deltas = torch.tensor(
        (-1, 1),
        device=device,
        dtype=torch.long,
    )

    raw_hole_targets = (
        hole[:, None]
        + hole_deltas[None, :]
    )

    valid_hole_moves = (
        (raw_hole_targets >= 0)
        & (raw_hole_targets < L)
    )

    hole_targets = raw_hole_targets.clamp(
        min=0,
        max=L - 1,
    )

    hole_choice, _ = _choose_valid_gpu(
        valid_hole_moves
    )

    selected_hole_target = hole_targets[
        rows,
        hole_choice,
    ]

    # --------------------------------------------------------------
    # Adjacent opposite-spin swaps
    # --------------------------------------------------------------
    adjacent_left = torch.arange(
        L - 1,
        device=device,
    )
    adjacent_right = adjacent_left + 1

    adjacent_valid = (
        states[:, adjacent_left]
        * states[:, adjacent_right]
        == -1
    )

    # --------------------------------------------------------------
    # Opposite spins separated by a hole
    # --------------------------------------------------------------
    across_left = torch.arange(
        max(L - 2, 0),
        device=device,
    )
    across_right = across_left + 2

    if across_left.numel() > 0:
        across_valid = (
            (states[:, across_left + 1] == 0)
            & (
                states[:, across_left]
                * states[:, across_right]
                == -1
            )
        )
    else:
        across_valid = torch.empty(
            (batch_size, 0),
            device=device,
            dtype=torch.bool,
        )

    spin_left_all = torch.cat(
        (adjacent_left, across_left)
    )
    spin_right_all = torch.cat(
        (adjacent_right, across_right)
    )
    spin_valid = torch.cat(
        (adjacent_valid, across_valid),
        dim=1,
    )

    spin_choice, has_spin_move = _choose_valid_gpu(
        spin_valid
    )

    selected_spin_left = spin_left_all[spin_choice]
    selected_spin_right = spin_right_all[spin_choice]

    # Choose the move category independently for every walker.
    choose_hole_move = (
        torch.rand(batch_size, device=device) < 0.5
    )

    # Fall back to a hole move if no valid spin move exists.
    choose_hole_move = (
        choose_hole_move | ~has_spin_move
    )

    selected_left = torch.where(
        choose_hole_move,
        hole,
        selected_spin_left,
    )

    selected_right = torch.where(
        choose_hole_move,
        selected_hole_target,
        selected_spin_right,
    )

    return _swap_sites_gpu(
        states,
        selected_left,
        selected_right,
    )


# ----------------------------------------------------------------------
# Batched local energy
# ----------------------------------------------------------------------

@torch.no_grad()
def local_energy_on_the_fly_gpu(
    states,
    psi,
    L,
    t1,
    t2,
    J1=0.0,
    J2=0.0,
):
    """
    Compute local energies for a batch of GPU walker states.

    Args:
        states:
            Spin tensor with shape (B, L), or one state with shape (L,).
        psi:
            Neural-network wavefunction.
        L:
            System size.
        t1, t2, J1, J2:
            Hamiltonian parameters.

    Returns:
        Tensor of shape (B,), or a scalar for one input state.
    """
    single_state = states.ndim == 1

    if single_state:
        states = states.unsqueeze(0)

    parameter = next(psi.parameters())
    device = parameter.device
    model_dtype = parameter.dtype

    states = states.to(
        device=device,
        dtype=torch.int8,
    )

    batch_size = states.shape[0]

    hole = (
        (states == 0)
        .to(torch.long)
        .argmax(dim=1)
    )

    candidates = []
    coefficient_blocks = []

    diagonal_energy = torch.zeros(
        batch_size,
        device=device,
        dtype=model_dtype,
    )

    # --------------------------------------------------------------
    # NN hopping
    # --------------------------------------------------------------
    nn_deltas = torch.tensor(
        (-1, 1),
        device=device,
        dtype=torch.long,
    )

    nn_target_raw = (
        hole[:, None] + nn_deltas[None, :]
    )

    nn_valid = (
        (nn_target_raw >= 0)
        & (nn_target_raw < L)
    )

    nn_target = nn_target_raw.clamp(
        min=0,
        max=L - 1,
    )

    nn_hole = hole[:, None].expand_as(nn_target)

    candidates.append(
        _swapped_candidates_gpu(
            states,
            nn_hole,
            nn_target,
        )
    )

    coefficient_blocks.append(
        torch.full(
            nn_valid.shape,
            -t1,
            device=device,
            dtype=model_dtype,
        )
        * nn_valid
    )

    # --------------------------------------------------------------
    # NNN hopping
    # --------------------------------------------------------------
    nnn_deltas = torch.tensor(
        (-2, 2),
        device=device,
        dtype=torch.long,
    )

    nnn_target_raw = (
        hole[:, None] + nnn_deltas[None, :]
    )

    nnn_valid = (
        (nnn_target_raw >= 0)
        & (nnn_target_raw < L)
        & ((hole % 2) == 0)[:, None]
    )

    nnn_target = nnn_target_raw.clamp(
        min=0,
        max=L - 1,
    )

    nnn_hole = hole[:, None].expand_as(nnn_target)

    candidates.append(
        _swapped_candidates_gpu(
            states,
            nnn_hole,
            nnn_target,
        )
    )

    # Original coefficient: -t2 * (-1) = +t2.
    coefficient_blocks.append(
        torch.full(
            nnn_valid.shape,
            t2,
            device=device,
            dtype=model_dtype,
        )
        * nnn_valid
    )

    # --------------------------------------------------------------
    # NN spin exchange
    # --------------------------------------------------------------
    if J1 != 0.0:
        nn_left_fixed = torch.arange(
            L - 1,
            device=device,
        )
        nn_right_fixed = nn_left_fixed + 1

        nn_left = nn_left_fixed[None, :].expand(
            batch_size,
            -1,
        )
        nn_right = nn_right_fixed[None, :].expand(
            batch_size,
            -1,
        )

        spin_i = torch.gather(
            states,
            1,
            nn_left,
        )
        spin_j = torch.gather(
            states,
            1,
            nn_right,
        )

        occupied = (
            (spin_i != 0)
            & (spin_j != 0)
        )

        diagonal_energy += (
            (J1 / 4.0)
            * spin_i.to(model_dtype)
            * spin_j.to(model_dtype)
            * occupied
        ).sum(dim=1)

        flip_valid = (
            occupied
            & (spin_i != spin_j)
        )

        candidates.append(
            _swapped_candidates_gpu(
                states,
                nn_left,
                nn_right,
            )
        )

        coefficient_blocks.append(
            torch.full(
                flip_valid.shape,
                J1 / 2.0,
                device=device,
                dtype=model_dtype,
            )
            * flip_valid
        )

    # --------------------------------------------------------------
    # NNN spin exchange
    # --------------------------------------------------------------
    if J2 != 0.0:
        nnn_left_fixed = torch.arange(
            0,
            L - 2,
            2,
            device=device,
        )
        nnn_right_fixed = nnn_left_fixed + 2

        nnn_left = nnn_left_fixed[None, :].expand(
            batch_size,
            -1,
        )
        nnn_right = nnn_right_fixed[None, :].expand(
            batch_size,
            -1,
        )

        spin_i = torch.gather(
            states,
            1,
            nnn_left,
        )
        spin_j = torch.gather(
            states,
            1,
            nnn_right,
        )

        occupied = (
            (spin_i != 0)
            & (spin_j != 0)
        )

        diagonal_energy += (
            (J2 / 4.0)
            * spin_i.to(model_dtype)
            * spin_j.to(model_dtype)
            * occupied
        ).sum(dim=1)

        flip_valid = (
            occupied
            & (spin_i != spin_j)
        )

        candidates.append(
            _swapped_candidates_gpu(
                states,
                nnn_left,
                nnn_right,
            )
        )

        coefficient_blocks.append(
            torch.full(
                flip_valid.shape,
                J2 / 2.0,
                device=device,
                dtype=model_dtype,
            )
            * flip_valid
        )

    # --------------------------------------------------------------
    # Batched neural-network evaluation
    # --------------------------------------------------------------
    all_candidates = torch.cat(
        candidates,
        dim=1,
    )

    coefficients = torch.cat(
        coefficient_blocks,
        dim=1,
    )

    number_of_connections = all_candidates.shape[1]

    current_inputs = spin_tensor_to_onehot(
        states,
        dtype=model_dtype,
    )

    connected_inputs = spin_tensor_to_onehot(
        all_candidates.reshape(-1, L),
        dtype=model_dtype,
    )

    psi_s = psi(current_inputs).reshape(-1)

    psi_connected = psi(
        connected_inputs
    ).reshape(
        batch_size,
        number_of_connections,
    )

    safe_psi_s = torch.where(
        psi_s.abs() < 1e-12,
        torch.full_like(psi_s, 1e-12),
        psi_s,
    )

    E_loc = (
        diagonal_energy.to(dtype=psi_s.dtype)
        + (
            coefficients.to(dtype=psi_s.dtype)
            * psi_connected
            / safe_psi_s[:, None]
        ).sum(dim=1)
    )

    return E_loc[0] if single_state else E_loc


# ----------------------------------------------------------------------
# Optional GPU pretraining coefficients
# ----------------------------------------------------------------------

@torch.no_grad()
def find_state_coeff_gpu(states, tb_coeff):
    """
    Batched GPU equivalent of find_state_coeff().
    """
    single_state = states.ndim == 1

    if single_state:
        states = states.unsqueeze(0)

    batch_size, L = states.shape

    if L % 2 != 1:
        raise ValueError("L must be odd")

    hole = (
        (states == 0)
        .to(torch.long)
        .argmax(dim=1)
    )

    # Remove the hole while preserving site order.
    indices = (
        torch.arange(
            L - 1,
            device=states.device,
        )[None, :]
        .expand(batch_size, -1)
    )

    indices = indices + (
        indices >= hole[:, None]
    ).to(indices.dtype)

    nonhole_spins = torch.gather(
        states,
        1,
        indices,
    ).reshape(
        batch_size,
        -1,
        2,
    )

    first = nonhole_spins[:, :, 0]
    second = nonhole_spins[:, :, 1]

    pair_sign = torch.where(
        (first == 1) & (second == -1),
        torch.ones_like(first),
        torch.where(
            (first == -1) & (second == 1),
            -torch.ones_like(first),
            torch.zeros_like(first),
        ),
    )

    sign = pair_sign.prod(dim=1)

    tb_coeff = torch.as_tensor(
        tb_coeff,
        device=states.device,
    )

    result = (
        sign.to(tb_coeff.dtype)
        * tb_coeff[hole]
    )

    return result[0] if single_state else result


# ----------------------------------------------------------------------
# Fully GPU-resident sampling
# ----------------------------------------------------------------------

def Obtain_Sampling_batch_gpu(
    psi,
    L,
    initial_states,
    n_steps,
    pretrain=False,
    burnin=False,
    tb_coeff=None,
    print_rate=False,
    Sampler=GlobalSampler_gpu,
    t1=1.0,
    t2=0.5,
    J1=0.0,
    J2=0.0,
    device=None,
    evaluation_batch_size=None,
):
    """
    Fully GPU-resident batched Metropolis sampling.

    Args:
        psi:
            Neural-network wavefunction already placed on CPU or CUDA.
        initial_states:
            Original list of Python states for the first call, or an
            int8 tensor returned by a previous call.
        evaluation_batch_size:
            Optional chunk size for final amplitude and local-energy
            evaluation. Use this if all samples do not fit in GPU memory.

    Returns:
        psis:
            Amplitudes with gradients, shape (n_steps*n_walkers,).
        energies:
            Detached local energies with the same shape.
        states:
            Final int8 walker tensor on the model's device.
        sampled_states:
            All retained int8 states on the model's device.
    """
    parameter = next(psi.parameters())
    model_device = parameter.device
    model_dtype = parameter.dtype

    # The model is the authoritative source of device.
    states = states_to_spin_tensor(
        initial_states,
        L,
        device=model_device,
    )

    n_walkers = states.shape[0]
    sampled_blocks = []

    accept_count = torch.zeros(
        (),
        device=model_device,
        dtype=torch.long,
    )

    # Cache current amplitudes. Model parameters remain unchanged during
    # this function.
    with torch.no_grad():
        current_inputs = spin_tensor_to_onehot(
            states,
            dtype=model_dtype,
        )

        psi_vals = psi(current_inputs).reshape(-1)

    for _ in range(n_steps):
        # Sampler processes every walker on the GPU.
        proposed = Sampler(states, L)

        proposed_inputs = spin_tensor_to_onehot(
            proposed,
            dtype=model_dtype,
        )

        with torch.no_grad():
            psi_new_vals = psi(
                proposed_inputs
            ).reshape(-1)

            denominator = psi_vals.abs().square()

            ratio = (
                psi_new_vals.abs().square()
                / (denominator + 1e-12)
            )

            accept_probability = ratio.clamp(max=1.0)

            accept = (
                torch.rand(
                    n_walkers,
                    device=model_device,
                )
                < accept_probability
            )

            # No CPU acceptance mask and no Python walker loop.
            states = torch.where(
                accept[:, None],
                proposed,
                states,
            )

            psi_vals = torch.where(
                accept,
                psi_new_vals,
                psi_vals,
            )

            accept_count += accept.sum()

        if not burnin:
            sampled_blocks.append(states.clone())

    if print_rate and n_steps > 0:
        acceptance_rate = (
            accept_count.float()
            / float(n_steps * n_walkers)
        )

        # Printing intentionally synchronizes once, after sampling.
        print(
            "Acceptance rate (batched GPU): "
            f"{acceptance_rate.item():.4f}"
        )

    if burnin or not sampled_blocks:
        empty_values = torch.empty(
            0,
            device=model_device,
            dtype=model_dtype,
        )

        empty_states = torch.empty(
            (0, L),
            device=model_device,
            dtype=torch.int8,
        )

        return (
            empty_values,
            empty_values,
            states,
            empty_states,
        )

    sampled_states = torch.stack(
        sampled_blocks,
        dim=0,
    ).reshape(-1, L)

    if evaluation_batch_size is None:
        chunk_size = sampled_states.shape[0]
    else:
        if evaluation_batch_size <= 0:
            raise ValueError(
                "evaluation_batch_size must be positive"
            )
        chunk_size = evaluation_batch_size

    # --------------------------------------------------------------
    # Gradient-enabled sample amplitudes
    # --------------------------------------------------------------
    psi_chunks = []

    for start in range(
        0,
        sampled_states.shape[0],
        chunk_size,
    ):
        state_chunk = sampled_states[
            start:start + chunk_size
        ]

        input_chunk = spin_tensor_to_onehot(
            state_chunk,
            dtype=model_dtype,
        )

        # Gradients must be enabled here because psis is used in the loss.
        psi_chunks.append(
            psi(input_chunk).reshape(-1)
        )

    psis = torch.cat(psi_chunks, dim=0)

    # --------------------------------------------------------------
    # Detached energies
    # --------------------------------------------------------------
    if pretrain:
        if tb_coeff is None:
            raise ValueError(
                "tb_coeff is required when pretrain=True"
            )

        energies = find_state_coeff_gpu(
            sampled_states,
            tb_coeff,
        ).to(
            device=model_device,
            dtype=model_dtype,
        )

    else:
        energy_chunks = []

        for start in range(
            0,
            sampled_states.shape[0],
            chunk_size,
        ):
            state_chunk = sampled_states[
                start:start + chunk_size
            ]

            energy_chunks.append(
                local_energy_on_the_fly_gpu(
                    state_chunk,
                    psi,
                    L,
                    t1,
                    t2,
                    J1=J1,
                    J2=J2,
                )
            )

        energies = torch.cat(
            energy_chunks,
            dim=0,
        ).reshape(-1)

    return psis, energies, states, sampled_states