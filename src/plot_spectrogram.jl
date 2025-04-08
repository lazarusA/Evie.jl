export circleShape
export plotSpectrogram
function plotSpectrogram(buf_obs, fs, txt_query, btn_label; colormap=:Bay, marker=:rect)
    x, y, rot_theta = circleShape(length(fs); r=0.8)
    n_cut = length(x)÷2
    # playing = Threads.Atomic{Bool}(false)

    fig = Figure(; size = (400,400), backgroundcolor=:grey8)
    ax = Axis(fig[1, 1]; aspect=DataAspect(), backgroundcolor=:grey8)
    button = Makie.Button(fig[1, 1]; label=btn_label, tellheight=false, tellwidth=false,
        halign=1, valign=1, buttoncolor=(:grey35, 0.15),
        buttoncolor_hover =:grey65, labelcolor=:silver)

    ax_txt_input = Axis(fig[1,1]; width=250, height=50,
        tellwidth=false, tellheight=false,
        halign=0.0, valign=1, backgroundcolor=(:grey25, 0.15))

    ax_txt_output = Axis(fig[1,1]; width=250, height=50,
        tellwidth=false, tellheight=false,
        halign=0.5, valign=0.0, backgroundcolor=(:grey25, 0.15))
    text!(ax_txt_input, txt_query, align=(:center, :center), color=:silver)
    
    scatter!(ax, x, y; color = buf_obs,
        colorrange=(1,100), marker, colormap,
        rotation=rot_theta[end:-1:1] .+ pi/2,
        markersize = @lift(Vec2f.($buf_obs, 6))
        )
    text!(ax_txt_output, txt_query, align=(:center, :center), color=:silver)

    lines!(ax, -0.5..0.5, @lift([$buf_obs[n_cut:-1:1]..., $buf_obs[1:n_cut]...]/(2*length(x)));
        color=@lift([$buf_obs[n_cut:-1:1]..., $buf_obs[1:n_cut]...]),
        colorrange=(1,100),
        colormap, linewidth=0.5, transparency=true
        )
    hidedecorations!(ax)
    hidedecorations!(ax_txt_output)
    hidedecorations!(ax_txt_input)
    hidespines!(ax)
    hidespines!(ax_txt_input)
    hidespines!(ax_txt_output)
    limits!(ax, -1.25,1.25,-1.25,1.25)
    display(fig; title="Evie.jl")
end

function circleShape(range_spectrum; h=0, k=0, r=1 )
    θ = range(pi/2, 2pi + pi/2-2π/range_spectrum, range_spectrum)[end:-1:1]
    return h .+ r*sin.(θ), k .+ r*cos.(θ), θ
end