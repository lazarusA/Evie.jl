export GroupedQueryAttention, KVCache

struct KVCache{K, V}
    k::K   # (head_dim, n_kv_heads, max_seq_len, max_batch)
    v::V
end

function KVCache(head_dim::Int, n_kv_heads::Int, max_seq_len::Int, max_batch::Int)
    return KVCache(
        zeros(Float32, head_dim, n_kv_heads, max_seq_len, max_batch),
        zeros(Float32, head_dim, n_kv_heads, max_seq_len, max_batch),
    )
end

struct GroupedQueryAttention{Q, K, V, O} <: Lux.AbstractLuxLayer
    wq::Q
    wk::K
    wv::V
    wo::O
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
end

function GroupedQueryAttention(dim::Int, n_heads::Int, n_kv_heads::Int)
    head_dim = dim ÷ n_heads
    return GroupedQueryAttention(
        Dense(dim => n_heads * head_dim; use_bias = false),
        Dense(dim => n_kv_heads * head_dim; use_bias = false),
        Dense(dim => n_kv_heads * head_dim; use_bias = false),
        Dense(n_heads * head_dim => dim; use_bias = false),
        n_heads, n_kv_heads, head_dim,
    )
end

function Lux.initialparameters(rng::AbstractRNG, m::GroupedQueryAttention)
    return (
        wq = Lux.initialparameters(rng, m.wq),
        wk = Lux.initialparameters(rng, m.wk),
        wv = Lux.initialparameters(rng, m.wv),
        wo = Lux.initialparameters(rng, m.wo),
    )
end

function Lux.initialstates(rng::AbstractRNG, m::GroupedQueryAttention)
    return (
        wq = Lux.initialstates(rng, m.wq),
        wk = Lux.initialstates(rng, m.wk),
        wv = Lux.initialstates(rng, m.wv),
        wo = Lux.initialstates(rng, m.wo),
    )
end

function repeat_kv(x::AbstractArray{T, 4}, n_rep::Int) where {T}
    n_rep == 1 && return x
    head_dim, n_kv, seq_len, batch = size(x)
    x = reshape(x, head_dim, 1, n_kv, seq_len, batch)
    x = repeat(x, 1, n_rep, 1, 1, 1)
    return reshape(x, head_dim, n_rep * n_kv, seq_len, batch)
end

# Causal mask for a prefill chunk attending over a KV cache that may already
# hold `kv_len - seq_len` past tokens. Shape (kv_len, q_len), true = attend —
# matches NNlib.dot_product_attention's expected mask orientation directly,
# rather than building it via tril/triu and risking a transpose.
function causal_mask_offset(seq_len::Int, kv_len::Int)
    offset = kv_len - seq_len
    mask = falses(kv_len, seq_len)
    for q in 1:seq_len, k in 1:kv_len
        mask[k, q] = k <= offset + q
    end
    return mask
end

# x: (dim, seq_len, batch). cache is mutated in place. start_pos is 1-based.
function (m::GroupedQueryAttention)(x, cache::KVCache, start_pos::Int, cosf, sinf, ps, st)
    dim, seq_len, batch = size(x)
    n_rep = m.n_heads ÷ m.n_kv_heads

    q, st_q = m.wq(x, ps.wq, st.wq)
    k, st_k = m.wk(x, ps.wk, st.wk)
    v, st_v = m.wv(x, ps.wv, st.wv)

    q = reshape(q, m.head_dim, m.n_heads, seq_len, batch)
    k = reshape(k, m.head_dim, m.n_kv_heads, seq_len, batch)
    v = reshape(v, m.head_dim, m.n_kv_heads, seq_len, batch)

    q = apply_rotary_embeddings(q, cosf, sinf)
    k = apply_rotary_embeddings(k, cosf, sinf)

    # Write into cache at [start_pos, start_pos+seq_len-1]
    cache.k[:, :, start_pos:(start_pos + seq_len - 1), 1:batch] .= k
    cache.v[:, :, start_pos:(start_pos + seq_len - 1), 1:batch] .= v

    kv_len = start_pos + seq_len - 1
    k_full = @view cache.k[:, :, 1:kv_len, 1:batch]
    v_full = @view cache.v[:, :, 1:kv_len, 1:batch]

    k_rep = repeat_kv(k_full, n_rep)   # (head_dim, n_heads, kv_len, batch)
    v_rep = repeat_kv(v_full, n_rep)

    # NNlib dot_product_attention wants (feat, seq, ...) with heads folded via nheads kwarg,
    # so flatten the explicit head dim back into a single (dim, seq, batch) layout per head group.
    q2 = reshape(permutedims(q, (1, 3, 2, 4)), m.head_dim, seq_len, :)
    k2 = reshape(permutedims(k_rep, (1, 3, 2, 4)), m.head_dim, kv_len, :)
    v2 = reshape(permutedims(v_rep, (1, 3, 2, 4)), m.head_dim, kv_len, :)

    # Causal mask only needed when seq_len > 1 (prefill); single-token decode needs none
    mask = seq_len > 1 ? causal_mask_offset(seq_len, kv_len) : nothing

    y, _ = dot_product_attention(q2, k2, v2; mask, nheads = 1)  # heads already split via reshape

    y = reshape(y, m.head_dim, seq_len, m.n_heads, batch)
    y = permutedims(y, (1, 3, 2, 4))
    y = reshape(y, m.n_heads * m.head_dim, seq_len, batch)

    out, st_o = m.wo(y, ps.wo, st.wo)
    return out, (wq = st_q, wk = st_k, wv = st_v, wo = st_o)
end
