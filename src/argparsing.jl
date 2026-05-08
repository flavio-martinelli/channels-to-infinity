# CLI argument parsing for distributed simulation runs.
# Defines `--slot` (job slot id, 1-based) and `--nprocs` (worker count).

using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--slot"
        help = "Each slot performs seed_per_slot seed simulations. Starts at --slot 1. Slot 2 will run seeds from seed_per_slot+1 to 2*seed_per_slot and so on. This keeps Ids unique."
        arg_type = Int
        required = false
        default = 1
    "--nprocs"
        help = "Number of CPU processes."
        arg_type = Int
        required = false
        default = 50
end
parsed_args = parse_args(ARGS, s)
