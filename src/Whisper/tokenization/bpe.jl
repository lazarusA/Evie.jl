module BPE

export encode_bpe

function encode_bpe(ranks, bytes)

    parts = [[b] for b in bytes]

    while true
        best = nothing
        best_rank = typemax(Int)

        for i in 1:(length(parts) - 1)
            pair = [parts[i]..., parts[i + 1]...]
            r = get(ranks, pair, nothing)
            if r !== nothing && r < best_rank
                best = i
                best_rank = r
            end
        end

        best === nothing && break

        merged = [parts[best]..., parts[best + 1]...]
        deleteat!(parts, best + 1)
        parts[best] = merged
    end

    return [ranks[p] for p in parts]
end

end
