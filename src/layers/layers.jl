# Layer 2: Distributed NN Layers
#
# Each layer implements Lux.jl's AbstractLuxLayer interface and is composed
# from Layer 1 primitives (halo exchange, broadcast, reduce, repartition).

include("mp_ops.jl")
include("dist_conv.jl")
include("dist_norm.jl")
include("dist_sample.jl")
include("dist_linear.jl")
include("dist_interpolate.jl")
include("dist_unet.jl")
