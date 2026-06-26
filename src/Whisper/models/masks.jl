module Masks

export causal_mask

# true = keep, false = block — consistent with NNlib.dot_product_attention
causal_mask(n::Int) = tril(ones(Bool, n, n))

end
