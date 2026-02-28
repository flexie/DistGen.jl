# DomainDecomposition.jl

Domain-decomposed deep learning in Julia with MPI, automatic differentiation, and GPU support.

DomainDecomposition.jl implements large-scale HPC neural networks using spatial domain decomposition with halo exchanges. All parallel primitives (halo exchange, broadcast, sum-reduce, all-reduce, repartition) are exposed as linear operators with exact adjoints via [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl), enabling seamless integration with Zygote reverse-mode AD.

The package ships a complete Karras-style 3D U-Net (EDM2) for score-based diffusion and flow matching, targeting seismic full-waveform inversion at scales up to 1024^3 with 20+ conditional channels.

## Architecture

The package is organized in three layers:

```
Layer 2 ── Distributed NN layers (Karras U-Net, Conv3d, GroupNorm, ...)
           │ composes from ↓
Layer 1 ── MPI primitives with AD adjoints (halo exchange, repartition, ...)
           │ runs on ↓
Layer 0 ── Device abstraction (CPU / CUDA via KernelAbstractions.jl)
```

### Layer 0: Device Abstraction

GPU-portable array operations via KernelAbstractions.jl. All tensor allocations use `similar(x, ...)` to preserve device type (CPU Array or CUDA CuArray). Helper functions `_zeros_like` / `_ones_like` provide Zygote-compatible, device-aware constant tensor creation.

### Layer 1: MPI Primitives

| Primitive | Forward | Adjoint (backward) |
|---|---|---|
| **Halo Exchange** | Bulk → neighbor ghost regions | Reverse exchange + accumulate into bulk |
| **Broadcast** | Root → all workers | Sum-reduce back to root |
| **Sum-Reduce** | All workers → root (sum) | Broadcast from root |
| **All-Reduce** | Sum across all workers | Self-adjoint (same all-reduce) |
| **Repartition** | Redistribute between topologies | Reverse repartition (swap src ↔ dst) |

All primitives use non-blocking MPI for communication and have `ChainRulesCore.rrule` definitions so Zygote can differentiate through them automatically.

### Layer 2: Distributed NN Layers

- **DistConv3d** — Domain-decomposed 3D convolution: halo exchange → local `NNlib.conv` → optional channel reduce
- **DistGroupNorm** — Group normalization with `AllReduce` for cross-partition statistics
- **DistAdaptiveGroupNorm** — FiLM-style conditioned normalization for time/noise embeddings
- **DistLinear** — Distributed matrix multiply with `AllReduce` aggregation
- **DistInterpolateDown / Up** — Mean-pool or nearest-neighbor upsample with topology transitions
- **DistKarrasEncoder / Decoder** — Magnitude-preserving residual blocks (Karras et al. 2024)
- **DistKarrasUNet3d** — Complete encoder-bottleneck-decoder U-Net with skip connections
- **EDM Preconditioner** — Sigma-dependent input/output scaling (Karras et al. 2022)

Magnitude-preserving operations: `mp_silu`, `mp_add`, `mp_cat`, `pixel_norm`, `normalize_weight`, `mp_fourier_embedding`.

## Installation

```julia
using Pkg
Pkg.develop(path="path/to/DomainDecomposition.jl")
```

### Dependencies

| Package | Purpose |
|---|---|
| MPI.jl | Message Passing Interface |
| ChainRulesCore.jl | AD rule definitions |
| NNlib.jl | Conv, pooling, activation functions |
| KernelAbstractions.jl | GPU-portable array operations |
| Zygote.jl | Reverse-mode automatic differentiation |

### Optional dependencies

| Package | Purpose |
|---|---|
| CUDA.jl | NVIDIA GPU support (weak dependency) |
| ParametricOperators.jl | Parametric linear operators via Kronecker products (extension) |

### MPI setup

DomainDecomposition.jl requires a working MPI installation. Install the Julia MPI wrapper:

```julia
using MPI
MPI.install_mpiexecjl()
```

This installs `mpiexecjl` which ensures the correct MPI library is used. If `mpiexecjl` is not on your PATH, you can launch tests via:

```julia
julia --project -e 'using MPI; mpiexec(exe -> run(`$exe -n 4 julia --project test/runtests.jl`))'
```

## Quick Start

```julia
using MPI
using DomainDecomposition

MPI.Init()
comm = MPI.COMM_WORLD

# Create a 2×2×1 spatial partition over 4 MPI ranks
P = create_cartesian_topology(comm, (2, 2, 1))

# Build a Karras-style U-Net
model = DistKarrasUNet3d(P, 2, 2;    # 2 in channels, 2 out channels
    dim=8, dim_max=16,
    num_downsamples=2,
    num_blocks_per_stage=(1, 1),
    fourier_dim=8)

# Initialize parameters (same seed on all ranks)
using Random
ps = init_dist_karras_unet(MersenneTwister(42), model; T=Float32)

# Each rank builds its local tile of the global 32^3 input
global_sp = (32, 32, 32)
x_global = randn(MersenneTwister(100), Float32, global_sp..., 2, 1)
local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
x_local = x_global[local_ranges..., :, :]

# Forward pass — halo exchanges happen automatically
y = dist_karras_unet_forward(x_local, Float32[0.5], ps, model)

# Gradient via Zygote — adjoints of MPI ops are automatic
using Zygote
loss, grads = Zygote.withgradient(ps) do p
    D_x = edm_precond_forward(x_local, Float32[1.0], p, model)
    sum(D_x .^ 2)
end
```

See `examples/train_karras_unet.jl` for a complete training loop with both EDM and flow matching.

## Testing

The test suite is organized in layers that mirror the source structure. Most tests require MPI (multiple ranks), and GPU tests additionally require NVIDIA GPUs.

### Test files

| File | What it tests | MPI ranks | GPU |
|---|---|---|---|
| `test_tensor_decomposition.jl` | Balanced decomposition, subtensor indices, overlaps | 1+ | No |
| `test_partition.jl` | Cartesian topology, neighbor ranks, sub-partitions | 4 | No |
| `test_halo_exchange.jl` | Ghost region exchange (1D, 2D, 3D), boundary conditions | 2-4 | No |
| `test_broadcast_reduce.jl` | Broadcast, sum-reduce, all-reduce operations | 4 | No |
| `test_repartition.jl` | All-to-all redistribution between partition topologies | 4 | No |
| `test_adjoint.jl` | Dot-product adjoint verification for all MPI primitives | 4 | No |
| `test_mp_ops.jl` | mp_silu, mp_add, mp_cat, pixel_norm, normalize_weight | 1 | No |
| `test_layers.jl` | DistConv3d, DistGroupNorm, DistLinear, skip connections, up/downsample | 4 | No |
| `test_karras_blocks.jl` | Karras encoder/decoder blocks, FiLM conditioning, downsample/upsample | 4 | No |
| `test_karras_unet.jl` | Full U-Net forward pass, EDM preconditioner, time conditioning | 4 | No |
| `test_ad.jl` | Zygote gradients of mp_ops and weight-normed DistConv3d | 4 | No |
| `test_gpu.jl` | All of the above on CuArrays, GPU vs CPU result comparison | 8 | Yes |

### Running CPU tests (4 MPI ranks)

The main test suite runs on 4 MPI ranks and covers all functionality on CPU:

```bash
# Using mpiexecjl (if installed):
mpiexecjl -n 4 julia --project test/runtests.jl

# Using Julia's MPI wrapper (always works):
julia --project -e 'using MPI; mpiexec(exe -> run(`$exe -n 4 julia --project test/runtests.jl`))'
```

Expected output: all test sets pass, including the Zygote AD tests. Typical runtime is ~2 minutes on a modern workstation.

### Running GPU tests (8 MPI ranks, 8 NVIDIA GPUs)

The GPU test suite validates that all operations produce identical results on CuArrays and CPU Arrays. It requires 8 MPI ranks with one GPU per rank (e.g., 8x NVIDIA A100).

```bash
# On a multi-GPU node (8 GPUs):
mpiexecjl -n 8 julia --project test/test_gpu.jl

# Or via Julia's MPI wrapper:
julia --project -e 'using MPI; mpiexec(exe -> run(`$exe -n 8 julia --project test/test_gpu.jl`))'
```

**GPU test structure:**
1. Conditional CUDA import — skips gracefully if CUDA is unavailable
2. Rank-to-GPU binding via `CUDA.device!(rank % ndevices)`
3. MPI primitives on CuArrays: halo exchange, all-reduce, repartition
4. NN layers on CuArrays: mp_ops, DistConv3d (with concat_ones), Karras encoder/decoder, full U-Net, EDM preconditioner
5. Zygote AD on CuArrays: gradients of DistConv3d, mp_silu, pixel_norm, normalize_weight

Each test creates data on CPU with a deterministic RNG, transfers to GPU, runs the operation, transfers back, and compares with the CPU reference result.

**Graceful degradation:**
- On machines without CUDA: prints `"CUDA not available"` and exits cleanly
- With fewer than 8 ranks: prints a message and exits cleanly
- Without Zygote: skips AD tests with an info message

### Running individual test files

Any test file can be run standalone (useful for debugging):

```bash
# Single-process tests (no MPI needed):
julia --project test/test_mp_ops.jl

# MPI tests (need appropriate rank count):
mpiexecjl -n 4 julia --project test/test_halo_exchange.jl
mpiexecjl -n 4 julia --project test/test_karras_unet.jl
mpiexecjl -n 4 julia --project test/test_ad.jl
```

### Running the training example

```bash
mpiexecjl -n 4 julia --project examples/train_karras_unet.jl
```

This runs 10 steps of EDM training followed by 10 steps of flow matching on a small 32^3 domain with 4 MPI ranks, demonstrating the full gradient computation pipeline.

### SLURM job script (multi-node GPU cluster)

```bash
#!/bin/bash
#SBATCH --job-name=dd-gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00

module load julia cuda mpi

srun julia --project test/test_gpu.jl
```

## GPU Compatibility

All source code uses generic `AbstractArray` patterns to work transparently on both CPU and GPU:

| Pattern | CPU | GPU (CuArray) |
|---|---|---|
| `similar(x, sz...)` | Creates `Array` | Creates `CuArray` |
| `copy(x)` | Copies `Array` | Copies `CuArray` |
| `copyto!(dst, src)` | CPU memcpy | GPU memcpy |
| `_zeros_like(x, sz...)` | `fill!(Array(...), 0)` | `fill!(CuArray(...), 0)` |
| `_ones_like(x, sz...)` | `fill!(Array(...), 1)` | `fill!(CuArray(...), 1)` |
| `repeat(x, inner=...)` | CPU repeat | GPU kernel |
| Broadcasting (`.+`, `.*`) | CPU vectorized | GPU kernel |

CUDA.jl is a **weak dependency** — it is never loaded unless the user explicitly imports it. The package works on CPU-only machines without any GPU libraries installed.

### MPI + CUDA interoperability

MPI.jl supports CUDA-aware MPI transports (e.g., NVIDIA NCCL, UCX). When running with a CUDA-aware MPI build, `MPI.Isend` / `MPI.Irecv!` operate directly on device memory without staging through the host. Set these environment variables for optimal GPU-direct communication:

```bash
export UCX_MEMTYPE_CACHE=no
export UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE
```

## Project Structure

```
DomainDecomposition.jl/
├── Project.toml
├── src/
│   ├── DomainDecomposition.jl      # Module definition and exports
│   ├── device.jl                   # Layer 0: device abstraction, _zeros_like/_ones_like
│   ├── partition.jl                # CartesianPartition topology management
│   ├── tensor_decomposition.jl     # Balanced decomposition, index computation
│   ├── halo_exchange.jl            # Ghost region exchange + rrule
│   ├── broadcast.jl                # Broadcast + rrule (adjoint: sum-reduce)
│   ├── sum_reduce.jl               # Sum-reduce + rrule (adjoint: broadcast)
│   ├── all_reduce.jl               # All-reduce + rrule (self-adjoint)
│   ├── repartition.jl              # Topology repartition + rrule
│   ├── adjoint_test.jl             # Dot-product adjoint verification
│   └── layers/
│       ├── layers.jl               # Layer include organizer
│       ├── mp_ops.jl               # Magnitude-preserving operations (Karras 2024)
│       ├── dist_conv.jl            # Distributed 3D convolution
│       ├── dist_norm.jl            # Distributed group norm + adaptive group norm
│       ├── dist_linear.jl          # Distributed linear layer
│       ├── dist_interpolate.jl     # Interpolation-based up/downsampling
│       ├── dist_sample.jl          # Strided conv up/downsampling
│       └── dist_unet.jl            # Karras U-Net, EDM, loss functions, samplers
├── ext/
│   └── ParametricOperatorsExt.jl   # ParametricOperators.jl bridge
├── examples/
│   └── train_karras_unet.jl        # EDM + flow matching training example
└── test/
    ├── runtests.jl                 # Main test runner (CPU, 4 ranks)
    ├── test_gpu.jl                 # GPU test suite (8 ranks, 8 GPUs)
    └── test_*.jl                   # Individual test files (see table above)
```

## References

- Karras et al. (2024), *Analyzing and Improving the Training Dynamics of Diffusion Models* — magnitude-preserving operations, EDM2
- Karras et al. (2022), *Elucidating the Design Space of Diffusion-Based Generative Models* — EDM preconditioning
- Hewett & Grady (2020), *A Linear Algebraic Framework for Domain-Decomposed Machine Learning* ([arXiv:2006.03108](https://arxiv.org/abs/2006.03108)) — adjoint theory for domain-decomposed ML
- [distdl](https://github.com/distdl/distdl) — Python prototype for domain-decomposed PyTorch
- [NVIDIA PhysicsNemo](https://docs.nvidia.com/physicsnemo/25.08/user-guide/domain_parallelism/domain_parallelism.html) — domain parallelism reference

## License

See repository for license information.
