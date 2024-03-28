export circleShape
export plotSpectrogram
function plotSpectrogram(x, y, rot_theta, buf_obs)
    n_cut = length(x)÷2
    fig = Figure(; size = (400,400))
    ax = Axis(fig[1, 1]; aspect=1)
    ax_txt = Axis(fig[1,1]; width=180, height=50,
        tellwidth=false, tellheight=false,
        halign=0.5, valign=0.35, backgroundcolor=(:grey45, 0.1))
    scatter!(ax, x, y; color = buf_obs,
        colorrange=(1,100), marker=:rect, colormap= :Bay,
        rotations=rot_theta[end:-1:1] .+ pi/2,
        markersize = @lift(Vec2f.($buf_obs, 6))
        )
    lines!(ax, -0.5..0.5, @lift([$buf_obs[n_cut:-1:1]..., $buf_obs[1:n_cut]...]/length(x));
        color=@lift([$buf_obs[n_cut:-1:1]..., $buf_obs[1:n_cut]...]),
        colorrange=(1,100),
        colormap = :Bay, linewidth=0.5, transparency=true
        )
    hidedecorations!(ax)
    hidedecorations!(ax_txt)
    hidespines!(ax)
    hidespines!(ax_txt)
    limits!(ax, -1.25,1.25,-1.25,1.25)
    fig
end

function circleShape(range_spectrum; h=0, k=0, r=1 )
    θ = range(pi/2, 2pi + pi/2-2π/range_spectrum, range_spectrum)[end:-1:1]
    return h .+ r*sin.(θ), k .+ r*cos.(θ), θ
end