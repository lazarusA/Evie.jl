export initFsBuf
export listenToMe
function initFsBuf(N = 1024, fmin = 0Hz, fmax = 10000Hz)
    PortAudioStream(1, 0) do stream
        buf = read(stream, N)
        fs =  domain(fft(buf)[fmin..fmax])
        return fs, Array(abs.(fft(buf)[fmin..fmax]))
    end
end
function listenToMe(seconds, buf_obs; N = 1024, fmin = 0Hz, fmax = 10000Hz, transcribe_text=false)
    PortAudioStream(1, 0) do stream
        done = false
        buf = read(stream, N)
        @sync begin
            if transcribe_text
                transcribe_async!(stream, model_att, done)
            end
            @async while !done
                read!(stream, buf)
                buf_obs[] = 3 .+ Array(5*abs.(fft(buf)[fmin..fmax]))
            end
            sleep(seconds)
            done = true
        end
    end
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