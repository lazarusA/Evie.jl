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
function decode(t::BPETokenizer, ids::AbstractVector{<:Integer}; include_specials::Bool = true)
    bytes = UInt8[]
    for id in ids
        v = get(t.vocab.decoder, Int64(id), nothing)
        if isnothing(v)
            include_specials && append!(bytes, codeunits(t.vocab.special_decoder[Int64(id)]))
        else
            append!(bytes, v)
        end
    end
    return String(bytes)
end
function decode(t::BPETokenizer, id::Integer)
    v = get(t.vocab.decoder, Int64(id), nothing)
    return isnothing(v) ? t.vocab.special_decoder[Int64(id)] : String(v)
end

end
