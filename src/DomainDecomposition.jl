module DomainDecomposition

using MPI
using KernelAbstractions
using ChainRulesCore
using NNlib

# Layer 0: Hardware abstraction utilities
include("device.jl")

# Layer 1: Core primitives
include("partition.jl")
include("tensor_decomposition.jl")
include("halo_exchange.jl")
include("broadcast.jl")
include("sum_reduce.jl")
include("all_reduce.jl")
include("repartition.jl")
include("adjoint_test.jl")

# Layer 2: Distributed NN layers (Lux.jl integration)
include("layers/layers.jl")

# Exports — Layer 1 primitives
export CartesianPartition, create_cartesian_topology, create_subpartition,
       create_partition_union, neighbor_ranks, coords_to_rank

export HaloInfo, compute_halo_info, halo_exchange!, halo_exchange

export BroadcastInfo, setup_broadcast, broadcast_op
export SumReduceInfo, setup_sum_reduce, sum_reduce_op
export AllReduceInfo, setup_all_reduce, all_reduce_op, all_reduce_op!
export RepartitionInfo, setup_repartition, repartition_op

export adjoint_test

# Exports — Tensor decomposition utilities
export balanced_decomposition, subtensor_indices, compute_overlaps, local_shape,
       compute_halo_sizes

# Exports — Device utilities
export device_array, device_copy!

# Exports — Layer 2: Magnitude-preserving operations (Karras)
export mp_silu, mp_add, mp_cat, pixel_norm, normalize_weight, mp_fourier_embedding

# Exports — Layer 2: Distributed NN layers
export DistConv3d, dist_conv3d_forward, init_dist_conv3d, pad_for_halo
export DistGroupNorm, DistAdaptiveGroupNorm, dist_groupnorm_forward
export DistDownsample, DistUpsample
export DistInterpolateDown, DistInterpolateUp
export DistLinear, DistSkipConnection
export DistUNet3d, PartitionPlan, plan_partitions
export DistKarrasEncoder, DistKarrasDecoder, DistKarrasUNet3d
export dist_karras_encoder_forward, dist_karras_decoder_forward
export dist_karras_unet_forward, edm_precond_forward
export init_dist_karras_encoder, init_dist_karras_decoder, init_dist_karras_unet
export sinusoidal_embedding
export score_based_diffusion_loss, flow_matching_loss
export langevin_sample, ode_sample

end # module
