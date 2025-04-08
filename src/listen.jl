export initFsBuf
export listenEvie
function initFsBuf(N = 1024, fmin = 0Hz, fmax = 10000Hz)
    PortAudioStream(1, 0) do stream
        buf = read(stream, N)
        fs =  domain(fft(buf)[fmin .. fmax])
        return fs, Array(abs.(fft(buf)[fmin .. fmax])), buf
    end
end

# audio_obs, speech_obs, audio_buf, t_seconds, btn_label, model_att
function listenEvie(buf_obs, txt_obs, circ_buf, t_seconds, model_att;
    N = 1024, fmin = 0Hz, fmax = 10000Hz, transcribe_text=false)
    ctx, wparams = loadWhisperModel(model_att)

    PortAudioStream(1, 0) do stream
        done = false
        buf = read(stream, N)
        @sync begin
            @async while !done
                yield()
                if transcribe_text
                    if isfull(circ_buf)
                        txt_obs[] = liveTranscribe(circ_buf, ctx, wparams)
                        empty!(circ_buf)
                    end
                end
            end
            @async while !done
                yield()
                read!(stream, buf)
                buf_obs[] = 3 .+ Array(5*abs.(fft(buf)[fmin .. fmax]))
                append!(circ_buf, Array(buf)[1:end])
            end
            sleep(t_seconds)
            Whisper.whisper_free(ctx)
            GC.gc()
            done = true
        end
    end
    return nothing
end

function pitchHalver(x) # decrease pitch by one octave via FFT
    N = length(x)
    mod(N,2) == 0 || throw("N must be multiple of 2")
    F = fft(x) # original spectrum
    Fnew = [F[1:N÷2]; zeros(N+1); F[(N÷2+2):N]]
    out = 2 * real(ifft(Fnew))[1:N]
    out.samplerate /= 2 # trick!
    return out
end