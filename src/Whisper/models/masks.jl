module Masks

export causal_mask

causal_mask(n::Int) = triu(ones(Bool, n, n), 1)

end
