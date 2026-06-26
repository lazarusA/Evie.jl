module Kernels

using NNlib

export scaled_dot_attention

function scaled_dot_attention(q, k, v; mask = nothing)
    scale = inv(sqrt(size(q, 1)))
    logits = batched_mul(q, permutedims(k, (2, 1, 3))) .* scale

    if mask !== nothing
        logits = ifelse.(mask, typemin(eltype(logits)), logits)
    end
    α = softmax(logits; dims = 2)

    return batched_mul(α, v)
end

end
