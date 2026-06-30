using FFTW
using AbstractFFTs
using Statistics
using Printf

export prep_audio, pad_or_trim, log_mel_spectrogram, prep_audio_chunks

const SAMPLE_RATE = 16_000
const N_FFT = 400
const HOP_LENGTH = 160
const N_MELS = 80
const N_MELS_V3 = 128
const N_FRAMES = 3000
const CHUNK_SIZE = SAMPLE_RATE * 30   # 480_000 samples = 30 seconds

function load_audio(path::String)
    s = FileIO.load(path)
    n_frames = round(Int, nframes(s) * (SAMPLE_RATE / samplerate(s)))  # nframes not length
    sout = SampleBuf(Float32, SAMPLE_RATE, n_frames, nchannels(s))
    write(SampleBufSink(sout), SampleBufSource(s))

    if nchannels(sout) == 1
        return vec(sout.data)
    elseif nchannels(sout) == 2
        sd = sout.data
        return Float32[(sd[i, 1] + sd[i, 2]) / 2.0f0 for i in 1:size(sd, 1)]
    else
        error("Unsupported number of channels: $(nchannels(sout))")
    end
end

# ── 1. Pad or trim to exactly 30 seconds ──────────────────────────────────────
function pad_or_trim(waveform::Vector{Float32}; len::Int = CHUNK_SIZE)
    if length(waveform) >= len
        return waveform[1:len]
    else
        padded = zeros(Float32, len)
        padded[1:length(waveform)] .= waveform
        return padded
    end
end

# ── 2. Hann window
function hanning(n::Int)
    scale = Float32(π / n)
    return Float32[sin(scale * k)^2 for k in 0:(n - 1)]
end

# ── 3. Reflect padding
function pad_reflect(x::Vector{Float32}, padding::Tuple{Int, Int})
    left, right = padding
    return vcat(x[left:-1:1], x, x[end:-1:(end - right + 1)])
end

# ── 4. Split into overlapping frames
function splitframes(x::Vector{Float32}; framelen::Int, hopsize::Int)
    n_frames = (length(x) - framelen) ÷ hopsize + 1
    frames = Matrix{Float32}(undef, framelen, n_frames)
    for i in 1:n_frames
        s = (i - 1) * hopsize + 1
        e = s + framelen - 1
        frames[:, i] = x[s:e]
    end
    return frames   # (framelen, n_frames)
end

# ── 5. STFT
function stft(
        x::Vector{Float32};
        n_fft::Int = N_FFT,
        hopsize::Int = HOP_LENGTH,
        center::Bool = true,
    )
    window = hanning(n_fft)
    n_bins = 1 + n_fft ÷ 2     # 201
    start = 1
    extra = 0

    x_frames_pre = Matrix{Float32}(undef, n_fft, 0)
    x_frames_post = Matrix{Float32}(undef, n_fft, 0)

    if center
        start_k = ceil(Int, (n_fft ÷ 2) / hopsize)
        tail_k = (length(x) + n_fft ÷ 2 - n_fft) ÷ hopsize + 1

        if tail_k <= start_k
            # Tail and head overlap — pad whole signal
            padding = (n_fft ÷ 2, n_fft ÷ 2)
            x = pad_reflect(x, padding)
        else
            # Pad head and tail separately
            start = start_k * hopsize - n_fft ÷ 2 + 1

            # Head frames
            slice = 1:((start_k - 1) * hopsize - n_fft ÷ 2 + n_fft + 1)
            x_pre = pad_reflect(x[slice], (n_fft ÷ 2, 0))
            x_frames_pre = splitframes(x_pre; framelen = n_fft, hopsize)
            x_frames_pre = x_frames_pre[:, 1:start_k]
            extra = size(x_frames_pre, 2)

            # Tail frames
            if (tail_k * hopsize - n_fft ÷ 2 + n_fft) <= (length(x) + n_fft ÷ 2)
                s = tail_k * hopsize - n_fft ÷ 2 + 1
                x_post = pad_reflect(x[s:end], (0, n_fft ÷ 2))
                x_frames_post = splitframes(x_post; framelen = n_fft, hopsize)
                extra += size(x_frames_post, 2)
            end
        end
    end

    x_frames = splitframes(x[start:end]; framelen = n_fft, hopsize)
    n_frames = size(x_frames, 2) + extra

    # Allocate output
    y = Matrix{ComplexF32}(undef, n_bins, n_frames)

    # Process head frames
    if extra > 0
        n_pre = size(x_frames_pre, 2)
        for i in 1:n_pre
            y[:, i] = rfft(x_frames_pre[:, i] .* window)
        end
        # Process tail frames
        n_post = size(x_frames_post, 2)
        if n_post > 0
            for i in 1:n_post
                y[:, n_frames - n_post + i] = rfft(x_frames_post[:, i] .* window)
            end
        end
    end

    # Process middle frames
    y_offset = size(x_frames_pre, 2)
    for i in 1:size(x_frames, 2)
        y[:, i + y_offset] = rfft(x_frames[:, i] .* window)
    end

    return y   # (201, n_frames) complex
end

# ── 6. Mel scale
function hz2mel(ω::Float32)
    fmin = 0.0f0
    fsp = 200.0f0 / 3.0f0
    mels = (ω - fmin) / fsp
    min_log_ω = 1000.0f0
    if ω >= min_log_ω
        min_log_mel = (min_log_ω - fmin) / fsp
        logstep = log(6.4f0) / 27.0f0
        mels = min_log_mel + log(ω / min_log_ω) / logstep
    end
    return mels
end

function mel2hz(mels::Float32)
    fmin = 0.0f0
    fsp = 200.0f0 / 3.0f0
    ω = fmin + fsp * mels
    min_log_ω = 1000.0f0
    min_log_mel = (min_log_ω - fmin) / fsp
    if mels >= min_log_mel
        logstep = log(6.4f0) / 27.0f0
        ω = min_log_ω * exp(logstep * (mels - min_log_mel))
    end
    return ω
end

# ── 7. Mel filterbank
function mel_filterbank(
        sample_rate::Float32;
        n_fft::Int = N_FFT,
        n_mels::Int = N_MELS,
        fmin::Float32 = 0.0f0,
        fmax::Float32 = sample_rate / 2.0f0,
    )
    # FFT frequencies
    fft_ω = Float32.(fftfreq(n_fft, sample_rate)[1:(n_fft ÷ 2 + 1)])
    fft_ω[end] *= -1.0f0   # flip negative Nyquist sign

    # Mel frequency points
    mel_ω = collect(range(hz2mel(fmin), hz2mel(fmax), length = n_mels + 2)) .|> mel2hz
    Δ = mel_ω[2:end] .- mel_ω[1:(end - 1)]

    # Ramp matrix — (n_freqs, n_mels+2)
    ramps = Matrix{Float32}(undef, length(fft_ω), length(mel_ω))
    for i in 1:length(fft_ω), j in 1:length(mel_ω)
        ramps[i, j] = mel_ω[j] - fft_ω[i]
    end

    # Build triangular filters — (n_freqs, n_mels)
    ω = zeros(Float32, n_fft ÷ 2 + 1, n_mels)
    for i in 1:n_mels
        lower = -ramps[:, i] ./ Δ[i]
        upper = ramps[:, i + 2] ./ Δ[i + 1]
        ω[:, i] .= max.(0.0f0, min.(lower, upper))
    end

    # Normalize
    enorm = 2.0f0 ./ (mel_ω[3:end] .- mel_ω[1:(end - 2)])
    ω .*= reshape(enorm, 1, :)

    return transpose(ω)   # (n_mels, n_freqs) = (80, 201)
end

# Cache filterbank — computed once at module load time
const MEL_FILTERS = mel_filterbank(Float32(SAMPLE_RATE); n_mels = N_MELS)
const MEL_FILTERS_V3 = mel_filterbank(Float32(SAMPLE_RATE); n_mels = N_MELS_V3)

# ── 8. Log mel spectrogram
function log_mel_spectrogram(
        waveform::Vector{Float32};
        n_mels::Int = N_MELS,
        n_fft::Int = N_FFT,
        hopsize::Int = HOP_LENGTH,
        debug::Bool = false,
    )
    filters = n_mels == N_MELS_V3 ? MEL_FILTERS_V3 : MEL_FILTERS

    # STFT → complex spectrogram
    freqs = stft(waveform; n_fft, hopsize)

    if debug
        @info "STFT shape: $(size(freqs))"
    end

    # Power spectrogram — drop last frame to get exactly N_FRAMES
    magnitudes = abs.(freqs[:, 1:(end - 1)]) .^ 2   # (201, 3000)

    if debug
        @info "Magnitudes shape: $(size(magnitudes))"
        @info "Magnitudes range: min=$(minimum(magnitudes)) max=$(maximum(magnitudes))"
    end

    # Apply mel filterbank
    mel_spec = filters * magnitudes             # (n_mels, 3000)

    if debug
        @info "Mel spec range: min=$(minimum(mel_spec)) max=$(maximum(mel_spec))"
    end

    # Log compression
    log_spec = log10.(max.(mel_spec, 1.0f-10))

    if debug
        @info "Log mel range before norm: min=$(minimum(log_spec)) max=$(maximum(log_spec))"
    end

    # Whisper normalization
    log_spec = max.(maximum(log_spec) - 8.0f0, log_spec)
    log_spec = (log_spec .+ 4.0f0) ./ 4.0f0

    if debug
        @info "Log mel range after norm: min=$(minimum(log_spec)) max=$(maximum(log_spec))"
    end

    return log_spec   # (n_mels, N_FRAMES)
end

# ── 9. Full pipeline ──────────────────────────────────────────────────────────
function prep_audio(
        waveform::Vector{Float32};
        n_mels::Int = N_MELS,
        debug::Bool = false,
    )
    # 1. Pad or trim to exactly 30 seconds
    waveform = pad_or_trim(waveform)

    # 2. Log mel spectrogram → (n_mels, N_FRAMES)
    log_spec = log_mel_spectrogram(waveform; n_mels, debug)

    # 3. Permute to (time, n_mels, batch=1) for model
    return permutedims(reshape(log_spec, n_mels, N_FRAMES, 1), (2, 1, 3))
    #      → (3000, 80, 1)
end

# ── 10. Chunked pipeline — for audio longer than 30 seconds ──────────────────
function prep_audio_chunks(
        waveform::Vector{Float32};
        n_mels::Int = N_MELS,
        overlap::Int = 0,
        debug::Bool = false,
    )
    chunks = Vector{Array{Float32, 3}}()
    step = CHUNK_SIZE - overlap
    start = 1

    while start <= length(waveform)
        stop = min(start + CHUNK_SIZE - 1, length(waveform))
        chunk = waveform[start:stop]
        push!(chunks, prep_audio(chunk; n_mels, debug))
        start += step
    end

    return chunks   # Vector of (3000, 80, 1) arrays
end
