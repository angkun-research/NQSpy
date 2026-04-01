using Plots
using LaTeXStrings
using Measures

# plot tanh(x), tanh(x^3), and tanh(x^5)
x = -2:0.01:2
y1 = tanh.(x)
y2 = tanh.(x.^3)
y3 = tanh.(x.^5)

plot(ylabel = L"a(x)",ymirror = true,
    xlabel = L"x",
     framestyle = :box,grid=false,legend=:bottomright,
        xtickfontsize=16, ytickfontsize=16,
        xguidefontsize=16, yguidefontsize=16,
        legendfontsize=16,#titlefontsize=12,
        xlim = [-2, 2], ylim = [-1.1, 1.1],
        size=(400, 300)
        )
plot!(x, y1, label = L"\tanh(x)", linewidth=5)
plot!(x, y2, label = L"\tanh(x^3)", linewidth=4)
plot!(x, y3, label = L"\tanh(x^5)", linewidth=4)
# savefig("~/Desktop/Fig1.pdf")