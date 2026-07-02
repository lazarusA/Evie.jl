# TransformerStack — kwargs-forwarding sequential combinator.
# Defined inline here since it's ~15 lines and avoids an external dependency.

struct TransformerStack{L} <: LuxCore.AbstractLuxContainerLayer{(:layers,)}
    layers::L
end

function TransformerStack(layers::Vector)
    names = Tuple(Symbol(:layer_, i) for i in eachindex(layers))
    return TransformerStack(NamedTuple{names}(layers))
end

function (m::TransformerStack)(x, ps, st; kwargs...)
    st_new = st.layers
    for name in keys(m.layers)
        x, st_i = m.layers[name](x, ps.layers[name], st.layers[name]; kwargs...)
        st_new = merge(st_new, (; name => st_i))
    end
    return x, (layers = st_new,)
end
