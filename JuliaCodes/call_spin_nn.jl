ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
# call_spin_nn.jl
using PythonCall

const repo_root = normpath(joinpath(@__DIR__, ".."))
const sys = pyimport("sys")
sys.path.insert(0, repo_root)

# Assumes:
# 1. `simple_spin_nn.py` is in the same directory or on PYTHONPATH
# 2. PyTorch is installed in the Python environment used by PythonCall
const pynet = pyimport("simple_spin_nn")

function load_nn(path::String="data/spin_net.pt")
    return pynet.load_model(path)
end

function nn_value(model, spin_config::AbstractVector{<:Integer})
    # Convert Julia vector to a plain Python-friendly vector of Int
    cfg = collect(Int.(spin_config))
    y = pynet.predict_scalar(model, cfg)
    return pyconvert(Float64, y)
end

# Example usage
model = load_nn("data/spin_net.pt")

sigma = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
val = nn_value(model, sigma)

println("spin config = ", sigma)
println("NN scalar output = ", val)

# Sketch of how you would use this in a QMC loop:
function ansatz_amplitude(model, sigma::Vector{Int})
    return nn_value(model, sigma)
end

for _ in 1:3
    trial_sigma = rand(0:1, 10)
    println("trial = ", trial_sigma, "  ansatz = ", ansatz_amplitude(model, trial_sigma))
end