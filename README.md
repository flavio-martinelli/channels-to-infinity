# Flat Channels to Infinity in Neural Loss Landscapes

Code for the NeurIPS 2025 paper *Flat Channels to Infinity in Neural Loss
Landscapes* by Flavio Martinelli, Alexander van Meegen, Berfin Şimşek, Wulfram
Gerstner, and Johanni Brea.

We identify and characterise *channels to infinity* — quasi-flat regions of the
loss landscape, asymptotically parallel to symmetry-induced saddle lines, along
which two neurons' input weights become equal while their output weights
diverge to ±∞. Optimizers like SGD and ADAM enter these channels with high
probability and look like they have found a flat local minimum. At convergence
the two neurons jointly implement a *gated linear unit*.

## Links

- **Paper (arXiv):** https://arxiv.org/abs/2506.14951
- **NeurIPS 2025 poster:** https://neurips.cc/virtual/2025/loc/san-diego/poster/118631
- **Blog post — interactive 2D channel visualisation:** https://flavio-martinelli.github.io/blog/2026/infinity/

## Citation

```bibtex
@inproceedings{martinelli2025channels,
  title     = {Flat Channels to Infinity in Neural Loss Landscapes},
  author    = {Martinelli, Flavio and van Meegen, Alexander and
               {\c{S}}im{\c{s}}ek, Berfin and Gerstner, Wulfram and Brea, Johanni},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

## Repository layout

```
channels-to-infinity/
├── Project.toml, Manifest.toml     # Julia environment (pinned to 1.9.4)
├── scripts/
│   ├── simulations/                # experiment runners
│   │   ├── loss_levels.jl
│   │   ├── scaling.jl
│   │   └── edge_of_stability_training.jl
│   └── data_process/               # plotting & analysis
│       ├── loss_levels_plots.jl
│       ├── loss_levels_plots_appendix.jl
│       ├── scaling_plots.jl
│       ├── edge_of_stability_plotting.jl
│       ├── plots_csv.jl
│       └── plot_gp.jl
├── src/                            # helper modules
│   ├── loadenv.jl                  # ensures the env is instantiated
│   ├── argparsing.jl               # --slot / --nprocs CLI flags
│   ├── distributed.jl              # spawns workers, broadcasts helper_experiment
│   ├── helper_experiment.jl        # run_experiment(exp_name, settings)
│   ├── helper_train.jl             # setup() + train wrapper
│   ├── funs_input.jl               # input-sample generators
│   ├── funs_teachers.jl            # rosenbrock + GP teachers
│   ├── helper_analyse.jl           # result retrieval / DataFrame assembly
│   ├── helper_retrievers.jl        # weight-norm, eigenvalue, closest-pair retrievers
│   ├── helper_plot.jl              # plotting utilities
│   ├── colors.jl                   # color palettes
│   └── plot_formatting_settings.mplstyle
├── paper_data/                     # committed reference inputs used directly by plotting
│   ├── plots_csv/                  #   CSVs for Fig 5 + Fig 6 (a–c)
│   ├── edge_of_stability/          #   initialization.dat for Fig 6 (d)
│   ├── fig1A_saddle_perturbation/  #   .dat files for Fig 2 (c, d)
│   └── scaling/                    #   scaling network used as the seed for Fig 3
└── data/                           # auto-created at runtime by DrWatson
    ├── sims/                       #   raw simulation outputs (per exp_name)
    └── proc/                       #   processed plots
```

## Setup

The repo uses [Julia](https://julialang.org/) and
[DrWatson.jl](https://juliadynamics.github.io/DrWatson.jl/stable/). To
reproduce locally:

0. Clone this repository. Raw simulation data is **not** committed — it will
   be regenerated under `data/sims/` when you run the simulation scripts.
1. From a Julia REPL:
   ```julia
   julia> using Pkg
   julia> Pkg.add("DrWatson")              # install globally so @quickactivate works
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This installs all dependencies pinned in `Manifest.toml` and enables DrWatson's
local-path handling.

Most scripts begin with
```julia
using DrWatson
@quickactivate
```
which auto-activates the project from any subdirectory.

> **Important — Julia version:**
> On **macOS**, do not use Julia versions above **1.9.4**. There are versioning
> issues with the packages `LoopVectorization` and `VectorizationBase`. On
> **Ubuntu** the latest versions (1.12) work fine.

## Reproducing the figures

| Figure | Simulation | Plotting |
|---|---|---|
| Fig 2 (a, b), Fig A1, A3 | `scripts/simulations/loss_levels.jl` | `scripts/data_process/loss_levels_plots.jl`, `loss_levels_plots_appendix.jl` |
| Fig 2 (c, d), Fig A2 | inputs in `paper_data/fig1A_saddle_perturbation/86/split_2/` | `scripts/simulations/fig2_perturbation_plot.jl` |
| Fig 3 | `scripts/simulations/fig3.jl` (reads `paper_data/scaling/`) | inline in the same script |
| Fig 4, Fig B series | `scripts/simulations/scaling.jl` | `scripts/data_process/scaling_plots.jl`, `plot_gp.jl` |
| Fig 5, Fig 6 (a–c) | inputs in `paper_data/plots_csv/` | `scripts/data_process/plots_csv.jl` |
| Fig 6 (d) | `scripts/simulations/edge_of_stability_training.jl` (reads `paper_data/edge_of_stability/initialization.dat`) | `scripts/data_process/edge_of_stability_plotting.jl` |

Figure 1 is a hand-drawn schematic and is not reproduced from code.

The single file in `paper_data/scaling/`
(`Din=4_f=g_input=quasimontecarlo_input_r=4_seed=18_teacher=rosenbrock.jld2`)
is the scaling-experiment network used as the seed for Figure 3. It is
produced by `scripts/simulations/scaling.jl` with the settings encoded in the
filename: `Din=4`, `r=4`, activation `g`, input `quasimontecarlo_input`,
teacher `rosenbrock`, seed `18`.

### Example: Figure 2

```bash
# Simulation (run on a machine with ~50 cores; see distributed flags below)
julia --project scripts/simulations/loss_levels.jl --slot 1 --nprocs 50

# Plotting (one figure per (activation, bias) pair)
julia --project scripts/data_process/loss_levels_plots.jl --bias false --f g
julia --project scripts/data_process/loss_levels_plots.jl --bias true  --f g
julia --project scripts/data_process/loss_levels_plots_appendix.jl --bias false --f softplus
```

### Example: Figure 4

```bash
julia --project scripts/simulations/scaling.jl --slot 1 --nprocs 50
julia --project scripts/data_process/scaling_plots.jl --f softplus
```

## Distributed / cluster runs

The simulation scripts shard their seed range across `--slot` IDs so they can
be launched in parallel on a cluster. Each slot runs `seeds_per_slot`
simulations across `--nprocs` worker processes. To cover the full sweep used in
the paper, launch the same script with `--slot 1`, `--slot 2`, …, on as many
nodes as you can spare. Sweep sizes are configured per-experiment inside each
script (`seed_range(slot_id, seeds_per_slot = …)`).

See `src/argparsing.jl` for the CLI flag definitions and `src/distributed.jl`
for how workers are added.
