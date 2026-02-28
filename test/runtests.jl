using Test
using DomainDecomposition

# Tests are organized by layer.
# MPI tests require `mpiexecjl -n 4 julia test/runtests.jl` or similar.
# Single-process tests verify logic; multi-process tests verify communication.

include("test_tensor_decomposition.jl")
include("test_partition.jl")
include("test_halo_exchange.jl")
include("test_broadcast_reduce.jl")
include("test_repartition.jl")
include("test_adjoint.jl")
include("test_mp_ops.jl")
include("test_layers.jl")
include("test_karras_blocks.jl")
include("test_karras_unet.jl")
include("test_ad.jl")
