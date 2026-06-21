x# NQSpy

[![arXiv](https://img.shields.io/badge/arXiv-2606.17045-b31b1b.svg)](https://arxiv.org/abs/2606.17045)

Neural Network Quantum Many-body states under python


# NQSpy

Neural-network ansatz and variational Monte Carlo (VMC) workflows for one-hole quantum spin-chain many-body states in Python.

This repository contains:
- Exact-data generators for supervised training.
- Reusable neural ansatz modules (fully connected, convolutional, hole-aware, local-rule variants).
- VMC samplers, local-energy evaluators, and stochastic reconfiguration (SR) optimizers.
- Experiment driver scripts for exact and non-exact Hamiltonians.
- CSV/checkpoint outputs under data.

## What This Repo Does

NQSpy studies doped spin systems (primarily one-hole chains) with state representation:
- State: (hole, up_sites), where hole is an integer site index and up_sites is a sorted tuple.
- One-hot tensor shape: (L, 4), channel order [hole, up, down, empty].

Core conventions are implemented in:
- vmc_utils.py
- ExactGS.py

## Installation

Dependencies are listed in requirements.txt:
- torch >= 2.0
- numpy
- scipy
- tqdm
- matplotlib
- pandas

Install with:
pip install -r requirements.txt

## Project Layout

Primary modules:
- ExactGS.py: Exact basis construction, RVB/tight-binding coefficient generation, one-hot training data, multi-hole basis/Hamiltonian helpers.
- vmc_utils.py: Main VMC runtime library (basis/Hamiltonian builders, samplers, local energies, SR/Jacobians, batched walkers, model classes used by VMC scripts).
- NeuralNetworks.py: Broader model zoo for supervised and architecture experiments.
- utils.py: Loss utilities.

Typical experiment scripts:
- Supervised exact data:
  - ExactNQS.py
  - ConvNQS.py
  - PhaseNQS.py
  - SignRuleNQS.py
- VMC and SR:
  - vmc_pretrain_test.py
  - vmc_sampler_test.py
  - vmc_sr_test.py
  - vmc_stc.py
  - vmc_runs.py
  - vmc_nonexact.py
  - vmc_nonexact_run.py
  - vmc_pretrain_stc.py
  - vmc_signrule.py
  - vmc_fullconfig.py (currently has an early exit in script body)

## Quick Start

Run from repository root:
- python ExactNQS.py
- python ConvNQS.py
- python PhaseNQS.py
- python SignRuleNQS.py
- python vmc_pretrain_test.py
- python vmc_sampler_test.py
- python vmc_sr_test.py
- python vmc_stc.py
- python vmc_nonexact.py
- python vmc_runs.py

Note:
- Most scripts are experiment drivers (not import-safe libraries). They execute immediately on run.
- Several defaults are computationally heavy (large L, many walkers/samples/epochs).

## Core API Summary

### Exact-data API (ExactGS.py)

- build_MB_basis(L): One-hole basis for odd L.
- build_MB_basis_holes(L, nholes=1, Nup=None): Generalized multi-hole basis.
- RVB_state(L): RVB basis/coefficient construction.
- exact_ground_state(L, t1, t2, basis, TBcoeff=True): Tight-binding-weighted RVB vector.
- basis_to_spinconfig(...), basis_to_onehot(...): Encodings.
- obtain_train_data(L, t1=1.0, t2=1.0, TBcoeff=False, Reshape=False, Normalize=False): Main supervised dataset helper.
- build_Hamiltonian_holes(...), build_Hamiltonian_holes_adjlist(...): Multi-hole Hamiltonian builders.

References:
- ExactGS.py

### VMC utilities API (vmc_utils.py)

State/basis/Hamiltonian:
- build_MB_basis
- build_Hamiltonian
- build_Hamiltonian_adjlist
- adjlist_to_csr
- state_to_onehot

Sampling:
- propose_move
- GlobalSampler
- BalancedSampler
- generate_initial_state
- Obtain_Sampling_batch (multi-walker batched network evaluation)

Energies:
- local_energy
- local_energy_from_adjlist
- local_energy_on_the_fly
- local_energy_coeff

Reference coefficients:
- TightBinding_coeff
- find_state_coeff

Stochastic reconfiguration:
- compute_log_psi_jacobian_vmap
- sr_gradient
- sr_update
- sr_update_optimizer

Memory helper:
- cleanup_memory

References:
- vmc_utils.py

### Neural model API (NeuralNetworks.py)

Main reusable classes include:
- FCNet
- ConvNet
- ConvSkipHole
- ConvHoleTanh
- PhysicsLocalLayer
- PhysicsConvNet
- ConvTransformer
- TransformerConv
- MultiKernelConvNet
- VisionTransformer1D
- LdepConvSkipHole
- LdepConv2d
- LdepConvHoles
- LdepConvStrides
- HoleOnLocalRule
- LdepConvPlusFC
- LocalRule

Reference:
- NeuralNetworks.py

## Script Workflows

Common VMC workflow (as implemented across vmc scripts):
1. Set physics parameters L, t1, t2, J1, J2 at top of script.
2. Instantiate ansatz model.
3. Optionally pretrain to tight-binding or exact-derived targets.
4. Burn in Markov walkers.
5. Run sampling + local energy estimation.
6. Update parameters via gradient estimator or SR.
7. Log energies and optionally save CSV/checkpoint artifacts.

Examples:
- Pretrain then VMC fine-tune: vmc_sampler_test.py
- Batched walkers + SR update: vmc_sr_test.py, vmc_stc.py
- Non-exact J1/J2 runs and error CSV output: vmc_nonexact_run.py, vmc_runs.py
- Sign-rule assisted ansatz loading and VMC: vmc_signrule.py

## Outputs

Typical outputs:
- Energy/error CSV logs in data
- Optional PyTorch checkpoints (many save blocks are present but sometimes commented out in scripts)
- Printed diagnostics: acceptance rate, mean local energy, standard error, fidelity (for tractable exact comparisons)

## Notes and Caveats

- Scripts are largely research/experiment style; many have hard-coded hyperparameters and long loops.
- Some scripts duplicate helper functions locally instead of importing a single shared runner.
- vmc_fullconfig.py currently exits early before optimization loop (contains an explicit exit call).
- For large L, prefer on-the-fly local energies and batched samplers to avoid full Hilbert-space construction.

## Citation and Paper

The corresponding manuscript is available at:
- [arXiv:2606.17045](https://arxiv.org/abs/2606.17045)