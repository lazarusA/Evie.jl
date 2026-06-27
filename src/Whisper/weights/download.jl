using Downloads

function download_weights(name::String, url::String, cache::String)
    file = joinpath(cache, name * ".pt")
    if !isfile(file)
        @info "Downloading $name weights from $url"
        Downloads.download(url, file)
    else
        @info "Found cached weights at $file"
    end
    return file
end
