module Tokenizer

using ..BPE

export BPETokenizer

struct BPETokenizer
    vocab
    pattern::Regex
end

function (t::BPETokenizer)(text::String)
    ids = Int[]

    for m in eachmatch(t.pattern, text)
        append!(ids, encode_bpe(t.vocab.mergeable, codeunits(m.match)))
    end

    return ids
end

end
