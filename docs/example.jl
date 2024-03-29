using Evie, GLMakie
fs, _buf = initFsBuf()
buf_obs = Observable(_buf)
x, y, rot_theta = circleShape(length(fs))

with_theme(theme_dark()) do
    plt_spectra = plotSpectrogram(x, y, rot_theta, buf_obs)
end

listenToMe(25, buf_obs)