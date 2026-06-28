export BPETokenizer, encode_bpe
export Vocab, load_vocab

const LANGUAGES = OrderedDict(
    "en" => "english",
    "de" => "german",
    "fr" => "french",
    "es" => "spanish"
    # add more as needed
)

const SPECIALS = vcat(
    ["<|endoftext|>", "<|startoftranscript|>"],
    ["<|$lang|>" for lang in keys(LANGUAGES)],      # keys = language codes
    ["<|translate|>", "<|transcribe|>"],
    ["<|startoflm|>", "<|startofprev|>"],
    ["<|nospeech|>", "<|notimestamps|>"],
    ["<|$(@sprintf("%.2f", i * 0.02))|>" for i in 0:1500],  # timestamp tokens
)

const DEFAULT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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
