module Vocabulary

export vocals, load_vocab

struct vocals
    mergeable::Dict{Vector{UInt8}, Int}
    special_tokens::Dict{String, Int}
end

function load_vocab(_file)
    mergeable = Dict{Vector{UInt8}, Int}()
    special = Dict{String, Int}()

    return vocals(mergeable, special)
end

end
