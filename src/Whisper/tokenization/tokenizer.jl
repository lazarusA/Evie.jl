module Tokenizer

using ..BPE
using ..Vocabulary

export BPETokenizer

const DEFAULT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

struct BPETokenizer
    vocab::Vocab
    pattern::Regex
end

function BPETokenizer(vocab::Vocab)
    return BPETokenizer(vocab, DEFAULT_PATTERN)
end

function (t::BPETokenizer)(text::String)
    ids = Int[]
    in_special = false
    special_buf = ""

    for m in eachmatch(t.pattern, text)
        # Accumulate special token
        if in_special
            special_buf *= m.match
            if m.match == "|>"
                in_special = false
                push!(ids, t.vocab.special_tokens[special_buf])
                special_buf = ""
            end
            continue
        end

        # Detect start of special token
        if m.match == "<|"
            in_special = true
            special_buf = m.match
            continue
        end

        # Regular BPE encoding
        append!(ids, encode_bpe(t.vocab.mergeable, codeunits(m.match)))
    end

    return ids
end

# Decode a single token id back to string
function decode(t::BPETokenizer, id::Int)
    # Check special tokens first
    for (k, v) in t.vocab.special_tokens
        v == id && return k
    end
    # Otherwise reverse lookup in mergeable ranks
    for (bytes, rank) in t.vocab.mergeable
        rank == id && return String(bytes)
    end
    error("Unknown token id: $id")
end

# Decode a sequence of token ids
function decode(t::BPETokenizer, ids::AbstractVector{Int}; include_specials::Bool = true)
    bytes = UInt8[]
    for id in ids
        found_special = false
        for (k, v) in t.vocab.special_tokens
            if v == id
                found_special = true
                include_specials && append!(bytes, codeunits(k))
                break
            end
        end
        found_special && continue
        for (b, rank) in t.vocab.mergeable
            if rank == id
                append!(bytes, b)
                break
            end
        end
    end
    return String(bytes)
end

end
