using Evie, GLMakie
fs, _buf = initFsBuf()
buf_obs = Observable(_buf)

with_theme(theme_dark()) do
     # backgroundcolor=:ghostwhite
    plt_spectra = plotSpectrogram(buf_obs, fs; marker=:circle,
        colormap=:Hiroshige)
end

listenToMe(10, buf_obs)