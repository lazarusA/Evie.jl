module Vocabulary
using Base64
using OrderedCollections
include("specials.jl")

export Vocab, load_vocab

struct Vocab
    mergeable::Dict{Vector{UInt8}, Int}
    special_tokens::Dict{String, Int}
    # Reverse lookups
    decoder::Dict{Int, Vector{UInt8}}
    special_decoder::Dict{Int, String}
end

function load_vocab(file::String)
    mergeable = Dict{Vector{UInt8}, Int}(
        (token == "=" ? UInt8[] : Base64.base64decode(token)) => parse(Int, rank)
            for (token, rank) in (split(line) for line in readlines(file))
    )
    n_vocab = length(mergeable)
    special_tokens = Dict{String, Int}(
        zip(SPECIALS, n_vocab:(n_vocab + length(SPECIALS) - 1))
    )
    decoder = Dict{Int, Vector{UInt8}}(v => k for (k, v) in mergeable)
    special_decoder = Dict{Int, String}(v => k for (k, v) in special_tokens)

    return Vocab(mergeable, special_tokens, decoder, special_decoder)
end

end
