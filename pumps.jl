#!/usr/bin/env julia
# pumps.jl -- run the whole puMPS pipeline from one TOML file.
#
#     julia --project=. pumps.jl examples/quickstart.toml
#
# This file contains no numerics. It reads the config, checks it, and then runs the
# existing scripts in src/ one after another, each in its own Julia process:
#
#     ground state  ->  climb in N  ->  grow D  ->  spectrum  ->  levels.csv  [-> ED]
#
# A stage whose output file already exists is skipped, so an interrupted run can be
# restarted with the same command. `--dry-run` prints the commands without running them.
using TOML, Printf, Serialization, Dates

const ROOT = @__DIR__
const SRC = joinpath(ROOT, "src")

# ----------------------------------------------------------------------------------
# Parameters. This table is the single source for defaults, `--template`, and the
# parameter table in README.md (`--params-table`).
# ----------------------------------------------------------------------------------
struct Required end          # the user must set this key
struct Param
    section::String
    key::String
    default::Any             # Required(), nothing (optional, no fixed default), or a value
    example::Any             # shown in the template
    doc::String
end

const SECTIONS = [
    "physics" => "The model: lattice Schwinger model on a ring of N sites.",
    "ground" => "Ground state. Runs first; every later stage starts from its output.",
    "spectrum" => "Excitation spectrum. Delete this whole table to stop after the ground state.",
    "ed" => "Exact diagonalization reference. Only practical for N up to about 20.",
    "output" => "Where results go.",
]

const PARAMS = [
    Param("physics", "x", Required(), 10.0, "x = 1/(a e)^2, so the lattice spacing is a e = 1/sqrt(x)"),
    Param("physics", "me", Required(), 0.3, "fermion mass m/e"),
    Param("physics", "theta", 0.0, 0.0, "background angle (use 3.141592653589793 for theta = pi)"),
    Param("physics", "hmax", 3, 3, "holonomy cutoff: 2*hmax+1 flux sectors (3 in all production runs)"),

    Param("ground", "N", Required(), 12, "number of sites (even). A list like [12, 16, 20] climbs in N, each ring seeded from the one before"),
    Param("ground", "D", Required(), [4, 6], "bond dimension. A list grows D step by step at the final N (steps of 2 work well)"),
    Param("ground", "tol", 1e-7, 1e-6, "converged when the gradient norm eta < tol"),
    Param("ground", "maxit", 800, 2500, "maximum optimizer iterations per stage"),
    Param("ground", "blas_threads", 1, 1, "BLAS threads; keep at or below the number of free physical cores"),
    Param("ground", "obc_D", nothing, 16, "bond dimension of the open-chain DMRG seed, >= first D (default: first D)"),
    Param("ground", "obc_sweeps", 12, 12, "DMRG sweeps for the open-chain seed"),
    Param("ground", "start", nothing, "old_ground.jls", "start from this saved ground state instead of an open-chain seed (needs the same D, x, hmax)"),
    Param("ground", "seed", 20260930, 20260930, "random seed"),

    Param("spectrum", "momenta", "all", "all", "\"all\" = k = 0..L/2 with L = N/2 (the others follow from E(k) = E(L-k)), or a list like [0, 1, 2]"),
    Param("spectrum", "nev", 4, 4, "number of levels per momentum"),
    Param("spectrum", "solver", "krylov", "dense", "\"dense\": about 10x faster at gapped points. \"krylov\": better near criticality"),
    Param("spectrum", "threads", 1, 4, "Julia threads for the spectrum"),
    Param("spectrum", "at", "last", "last", "\"last\" = only the largest D. \"all\" = every D in ground.D, to check convergence in D"),

    Param("ed", "enabled", false, false, "set to true to also run exact diagonalization at the final N"),
    Param("ed", "momenta", nothing, [0, 1], "default: the same as spectrum.momenta"),
    Param("ed", "nev", nothing, 4, "default: the same as spectrum.nev"),
    Param("ed", "threads", 1, 4, "Julia threads for ED"),

    Param("output", "dir", nothing, "runs/quickstart", "results folder, relative to this config file (default: runs/<config file name>)"),
]

struct ConfigError <: Exception
    msg::String
end
fail(msg) = throw(ConfigError(msg))

tomlvalue(v::AbstractString) = repr(String(v))
tomlvalue(v::AbstractVector) = "[" * join(tomlvalue.(v), ", ") * "]"
tomlvalue(v) = string(v)

function template()
    io = IOBuffer()
    println(io, "# puMPS run configuration. Run it with:")
    println(io, "#     julia --project=. pumps.jl this_file.toml")
    println(io, "# Lines starting with # are optional settings, shown with an example value.")
    for (section, about) in SECTIONS
        println(io, "\n# ", about, "\n[", section, "]")
        for p in filter(p -> p.section == section, PARAMS)
            setting = "$(p.key) = $(tomlvalue(p.example))"
            if p.default isa Required
                println(io, rpad(setting, 28), " # (required) ", p.doc)
            elseif p.default === nothing
                println(io, rpad("# " * setting, 28), " # ", p.doc)
            else
                note = p.default == p.example ? "" : " (default $(tomlvalue(p.default)))"
                println(io, rpad(setting, 28), " # ", p.doc, note)
            end
        end
    end
    String(take!(io))
end

function params_table()
    io = IOBuffer()
    println(io, "| Section | Key | Default | Meaning |")
    println(io, "|---|---|---|---|")
    for p in PARAMS
        default = p.default isa Required ? "**required**" : p.default === nothing ? "" : "`$(tomlvalue(p.default))`"
        println(io, "| `[$(p.section)]` | `$(p.key)` | $default | $(replace(p.doc, "|" => "\\|")) |")
    end
    String(take!(io))
end

# ----------------------------------------------------------------------------------
# Reading and checking the config
# ----------------------------------------------------------------------------------
isint(v) = v isa Integer && !(v isa Bool)
isnumber(v) = v isa Real && !(v isa Bool)

function intlist(v, name)
    list = v isa AbstractVector ? collect(v) : [v]
    !isempty(list) && all(isint, list) || fail("$name must be an integer or a list of integers, got $(repr(v))")
    Int.(list)
end
function number(v, name)
    isnumber(v) || fail("$name must be a number, got $(repr(v))")
    Float64(v)
end
function integer(v, name; min=1)
    isint(v) || fail("$name must be an integer, got $(repr(v))")
    v >= min || fail("$name must be at least $min, got $v")
    Int(v)
end
function choice(v, name, options)
    v isa AbstractString && v in options || fail("$name must be one of $(join(repr.(options), ", ")), got $(repr(v))")
    String(v)
end
function momenta(v, name, N)
    L = N ÷ 2
    v == "all" && return collect(0:L÷2)
    v isa AbstractVector || fail("$name must be \"all\" or a list of integers like [0, 1, 2], got $(repr(v))")
    ks = intlist(v, name)
    allunique(ks) || fail("$name has repeated entries: $ks")
    all(k -> 0 <= k < L, ks) || fail("$name must lie in 0..$(L - 1) for N = $N (L = N/2 = $L unit cells), got $ks")
    ks
end

function load_config(path)
    isfile(path) || fail("config file not found: $path")
    raw = try
        TOML.parsefile(path)
    catch err
        fail("could not read $path as TOML:\n" * sprint(showerror, err))
    end
    sections = first.(SECTIONS)
    keysof(section) = [p.key for p in PARAMS if p.section == section]
    for (name, table) in raw
        if !(table isa AbstractDict)
            home = [p.section for p in PARAMS if p.key == name]
            hint = isempty(home) ? "" : " Put it under [$(first(home))]."
            fail("`$name` is outside any section.$hint Every setting belongs to one of: " * join("[" .* sections .* "]", " "))
        end
        name in sections || fail("unknown section [$name]. Valid sections: " * join("[" .* sections .* "]", " "))
        for key in keys(table)
            key in keysof(name) || fail("unknown setting `$key` in [$name]. Valid settings there: " * join(keysof(name), ", "))
        end
    end
    function setting(section, key)
        p = only(filter(p -> p.section == section && p.key == key, PARAMS))
        table = get(raw, section, Dict())
        haskey(table, key) && return table[key]
        p.default isa Required && fail("missing required setting `$key` in [$section]  ($(p.doc))")
        p.default
    end
    for section in ("physics", "ground")
        haskey(raw, section) || fail("missing section [$section]. Run `julia pumps.jl --template` for a complete example.")
    end

    x = number(setting("physics", "x"), "physics.x")
    x > 0 || fail("physics.x must be positive, got $x")
    me = number(setting("physics", "me"), "physics.me")
    theta = number(setting("physics", "theta"), "physics.theta")
    hmax = integer(setting("physics", "hmax"), "physics.hmax"; min=0)

    Ns = intlist(setting("ground", "N"), "ground.N")
    all(N -> N >= 4 && iseven(N), Ns) || fail("ground.N must be even and at least 4 (two sites per unit cell), got $Ns")
    issorted(Ns; lt=<=) || fail("ground.N must increase, got $Ns")
    Ds = intlist(setting("ground", "D"), "ground.D")
    all(>=(2), Ds) || fail("ground.D must be at least 2, got $Ds")
    issorted(Ds; lt=<=) || fail("ground.D must increase (D can be grown, not shrunk), got $Ds")
    tol = number(setting("ground", "tol"), "ground.tol")
    tol > 0 || fail("ground.tol must be positive, got $tol")
    maxit = integer(setting("ground", "maxit"), "ground.maxit")
    blas = integer(setting("ground", "blas_threads"), "ground.blas_threads")
    obc_sweeps = integer(setting("ground", "obc_sweeps"), "ground.obc_sweeps")
    seed = integer(setting("ground", "seed"), "ground.seed"; min=0)
    here = dirname(abspath(path))
    start = setting("ground", "start")
    if start !== nothing
        start isa AbstractString || fail("ground.start must be a file name in quotes, got $(repr(start))")
        start = abspath(here, start)
        isfile(start) || fail("ground.start file not found: $start")
        haskey(get(raw, "ground", Dict()), "obc_D") && fail("ground.obc_D has no effect together with ground.start; remove one of them")
    end
    obcD = setting("ground", "obc_D")
    obcD = obcD === nothing ? first(Ds) : integer(obcD, "ground.obc_D")
    obcD >= first(Ds) || fail("ground.obc_D ($obcD) must be at least the first D ($(first(Ds)))")

    N = last(Ns)
    spectrum = nothing
    if haskey(raw, "spectrum")
        spectrum = (momenta=momenta(setting("spectrum", "momenta"), "spectrum.momenta", N),
            nev=integer(setting("spectrum", "nev"), "spectrum.nev"),
            solver=choice(setting("spectrum", "solver"), "spectrum.solver", ("dense", "krylov")),
            threads=integer(setting("spectrum", "threads"), "spectrum.threads"),
            Ds=choice(setting("spectrum", "at"), "spectrum.at", ("last", "all")) == "all" ? Ds : [last(Ds)])
    end
    ed = nothing
    enabled = setting("ed", "enabled")
    enabled isa Bool || fail("ed.enabled must be true or false, got $(repr(enabled))")
    if enabled
        ks = setting("ed", "momenta")
        nev = setting("ed", "nev")
        ed = (momenta=ks === nothing ? (spectrum === nothing ? momenta("all", "", N) : spectrum.momenta) : momenta(ks, "ed.momenta", N),
            nev=nev === nothing ? (spectrum === nothing ? 4 : spectrum.nev) : integer(nev, "ed.nev"),
            threads=integer(setting("ed", "threads"), "ed.threads"))
    end

    dir = setting("output", "dir")
    dir === nothing || dir isa AbstractString || fail("output.dir must be a folder name in quotes, got $(repr(dir))")
    dir = abspath(here, dir === nothing ? joinpath("runs", splitext(basename(path))[1]) : dir)

    (; file=abspath(path), x, me, theta, hmax, Ns, Ds, tol, maxit, blas, obcD, obc_sweeps, seed, start, spectrum, ed, dir)
end

# ----------------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------------
struct Stage
    group::Symbol                         # :ground, :spectrum or :ed
    label::String                         # also the log file name
    title::String
    script::String
    args::Vector{String}
    threads::Int
    env::Vector{Pair{String,String}}
    inputs::Vector{String}                # files that must exist before it can run
    outputs::Vector{String}               # the stage is done when all of these exist
    prepare::Function                     # writes generated files; not called on --dry-run
end

ground_file(cfg, N, D) = joinpath(cfg.dir, "ground_N$(N)_D$(D).jls")
spectrum_file(cfg, N, D) = joinpath(cfg.dir, "spectrum_N$(N)_D$(D).jls")
physics_args(cfg, N) = [string(N), string(cfg.x), string(cfg.me), string(cfg.theta), string(cfg.hmax)]

function build_stages(cfg)
    stages = Stage[]
    D1 = first(cfg.Ds)
    N = last(cfg.Ns)
    # 1+2. Ground state at the first D: seed the first ring, then carry it to larger rings.
    previous = cfg.start
    for (i, Ni) in enumerate(cfg.Ns)
        out = ground_file(cfg, Ni, D1)
        stagecfg = joinpath(cfg.dir, "stage_configs", "ground_N$(Ni)_D$(D1).toml")
        settings = Dict{String,Any}("N" => Ni, "D" => D1, "x" => cfg.x, "me" => cfg.me, "theta" => cfg.theta,
            "hmax" => cfg.hmax, "tol" => cfg.tol, "maxit" => cfg.maxit, "blas_threads" => cfg.blas,
            "seed" => cfg.seed, "output" => out)
        if previous === nothing
            settings["obc_D"] = cfg.obcD
            settings["obc_sweeps"] = cfg.obc_sweeps
            title = "ground state N=$Ni D=$D1 (open-chain DMRG seed)"
        else
            settings["source"] = previous
            title = "ground state N=$Ni D=$D1 (seeded from $(basename(previous)))"
        end
        prepare = () -> (mkpath(dirname(stagecfg)); open(io -> TOML.print(io, settings; sorted=true), stagecfg, "w"))
        push!(stages, Stage(:ground, "ground_N$(Ni)_D$(D1)", title, joinpath(SRC, "run_seed_comparison.jl"),
            [stagecfg], 1, [], previous === nothing ? [] : [previous], [out], prepare))
        previous = out
    end
    # 3. Grow D at the final N.
    for D in cfg.Ds[2:end]
        out = ground_file(cfg, N, D)
        push!(stages, Stage(:ground, "ground_N$(N)_D$(D)", "ground state N=$N D=$D (grown from $(basename(previous)))",
            joinpath(SRC, "prepare_bond_ground_tol.jl"),
            [previous; physics_args(cfg, N); string(D); string(cfg.maxit); out; string(cfg.tol)],
            1, ["PUMPS_BLAS" => string(cfg.blas)], [previous], [out], () -> nothing))
        previous = out
    end
    # 4. Spectrum.
    if cfg.spectrum !== nothing
        s = cfg.spectrum
        for D in s.Ds
            ground = ground_file(cfg, N, D)
            out = spectrum_file(cfg, N, D)
            push!(stages, Stage(:spectrum, "spectrum_N$(N)_D$(D)", "spectrum N=$N D=$D, k = $(join(s.momenta, ",")), $(s.solver) solver",
                joinpath(SRC, "run_local_excitation.jl"),
                [ground; physics_args(cfg, N); join(s.momenta, ","); string(s.nev); out],
                s.threads, s.solver == "dense" ? ["PUMPS_SOLVER" => "dense"] : [], [ground], [out], () -> nothing))
        end
    end
    # 5. Exact diagonalization. One file per momentum, so only the missing momenta are run.
    if cfg.ed !== nothing
        e = cfg.ed
        dir = joinpath(cfg.dir, "ed")
        file(k) = joinpath(dir, "ed_k$(k).jls")
        missing_ks = filter(k -> !isfile(file(k)), e.momenta)
        ks = isempty(missing_ks) ? e.momenta : missing_ks
        push!(stages, Stage(:ed, "ed_N$(N)", "exact diagonalization N=$N, k = $(join(ks, ","))",
            joinpath(SRC, "run_flux_ed.jl"), [physics_args(cfg, N); join(ks, ","); string(e.nev); dir],
            e.threads, [], [], file.(e.momenta), () -> nothing))
    end
    stages
end

isdone(stage) = all(isfile, stage.outputs)

# Paths are shown relative to the current folder so the printed command can be pasted as is.
short(path) = isabspath(path) ? relpath(path) : path
function shown(script, args, threads, env)
    prefix = join(["$k=$v " for (k, v) in env])
    "$(prefix)julia --project=$(short(ROOT)) -t $threads $(short(script)) $(join(short.(args), " "))"
end
shown(stage::Stage) = shown(stage.script, stage.args, stage.threads, stage.env)

function command(script, args, threads, env)
    # The config decides the solver and the BLAS threads, not whatever is set in the shell.
    environment = Dict{String,String}(k => v for (k, v) in ENV if !(k in ("PUMPS_SOLVER", "PUMPS_BLAS")))
    merge!(environment, Dict(env))
    setenv(`$(Base.julia_cmd()) --project=$ROOT --startup-file=no -t $threads $script $args`, environment)
end

function duration(seconds)
    s = round(Int, seconds)
    s < 60 ? "$(s) s" : s < 3600 ? "$(s ÷ 60) min $(s % 60) s" : "$(s ÷ 3600) h $((s % 3600) ÷ 60) min"
end

"Run one stage, copying its output to the terminal and to a log file. Returns the wall time."
function run_stage(stage, logfile)
    mkpath(dirname(logfile))
    stage.prepare()
    started = time()
    output = Pipe()
    process = run(pipeline(ignorestatus(command(stage.script, stage.args, stage.threads, stage.env));
            stdout=output, stderr=output); wait=false)
    close(output.in)
    open(logfile, "w") do log
        println(log, "# ", shown(stage), "\n# started ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        for line in eachline(output)
            println("    ", line)
            println(log, line)
            flush(log)
        end
    end
    wait(process)
    success(process) || error("stage `$(stage.label)` failed. Its full output is in $(short(logfile))")
    isdone(stage) || error("stage `$(stage.label)` finished but did not write $(join(short.(filter(!isfile, stage.outputs)), ", "))")
    time() - started
end

function physical_cores()
    try
        cores = Set{Tuple{String,String}}()
        package = ""
        for line in eachline("/proc/cpuinfo")
            startswith(line, "physical id") && (package = strip(split(line, ":")[2]))
            startswith(line, "core id") && push!(cores, (package, strip(split(line, ":")[2])))
        end
        isempty(cores) ? Sys.CPU_THREADS : length(cores)
    catch
        Sys.CPU_THREADS
    end
end

# One results folder holds one physical point. Reusing a folder after editing [physics]
# would silently mix results, so the point is recorded on the first run and checked later.
function check_physics_record(cfg; write)
    record = joinpath(cfg.dir, "physics.toml")
    now_ = Dict("x" => cfg.x, "me" => cfg.me, "theta" => cfg.theta, "hmax" => cfg.hmax)
    if isfile(record)
        before = TOML.parsefile(record)
        changed = [k for k in sort(collect(keys(now_))) if get(before, k, nothing) != now_[k]]
        isempty(changed) || fail("the folder $(short(cfg.dir)) already holds results for different physics:\n" *
            join(["    $k = $(get(before, k, "?")) there, $(now_[k]) in the config" for k in changed], "\n") *
            "\n  Choose a new [output] dir for the new parameters, or delete that folder to start over.")
    elseif write
        mkpath(cfg.dir)
        open(io -> TOML.print(io, now_; sorted=true), record, "w")
    end
end

function check_start(cfg)
    cfg.start === nothing && return
    saved = try
        deserialize(cfg.start)
    catch
        return          # unreadable here; the ground-state script will report it
    end
    hasproperty(saved, :physics) && hasproperty(saved, :result) || fail("ground.start: $(short(cfg.start)) is not a saved ground state")
    D = size(first(saved.result.As), 1)
    D == first(cfg.Ds) || fail("ground.start has D = $D but the first entry of ground.D is $(first(cfg.Ds)). Make them equal; later entries of ground.D grow it.")
    saved.physics.x == cfg.x || fail("ground.start was computed at x = $(saved.physics.x), the config has x = $(cfg.x)")
    saved.physics.hmax == cfg.hmax || fail("ground.start was computed at hmax = $(saved.physics.hmax), the config has hmax = $(cfg.hmax)")
end

# ----------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------
function summarize(cfg)
    N = last(cfg.Ns)
    println("\nResults in $(short(cfg.dir))")
    grounds = unique([[(Ni, first(cfg.Ds)) for Ni in cfg.Ns]; [(N, D) for D in cfg.Ds]])
    println("\n  Ground states")
    @printf("    %5s %4s %16s %14s %10s\n", "N", "D", "E0", "E0/N", "eta")
    for (Ni, D) in grounds
        file = ground_file(cfg, Ni, D)
        isfile(file) || continue
        try
            r = deserialize(file).result
            flag = r.eta < cfg.tol ? "" : "   <- not converged (eta > tol = $(cfg.tol))"
            @printf("    %5d %4d %16.8f %14.8f %10.1e%s\n", Ni, D, r.E, r.E / Ni, r.eta, flag)
        catch
            println("    N=$Ni D=$D: see $(short(file))")
        end
    end
    ed_levels = Dict{Int,Vector{Float64}}()
    if cfg.ed !== nothing
        for k in cfg.ed.momenta
            file = joinpath(cfg.dir, "ed", "ed_k$(k).jls")
            isfile(file) && (ed_levels[k] = try deserialize(file).levels catch; Float64[] end)
        end
    end
    ed_ground = isempty(get(ed_levels, 0, Float64[])) ? nothing : first(ed_levels[0])
    if cfg.spectrum !== nothing
        for D in cfg.spectrum.Ds
            file = spectrum_file(cfg, N, D)
            isfile(file) || continue
            s = deserialize(file)
            println("\n  Spectrum N=$N D=$D: energies above the ground state (momentum per site = 2 pi k / N)")
            for (i, k) in enumerate(s.settings.momenta)
                println("    k=$k   ", join([@sprintf("%9.5f", abs(e - s.ground_energy) < 5e-6 ? 0.0 : e - s.ground_energy) for e in s.levels[i]], " "))
                ed_ground !== nothing && haskey(ed_levels, k) &&
                    println("     ED    ", join([@sprintf("%9.5f", e - ed_ground) for e in ed_levels[k]], " "))
            end
        end
        println("\n  (At k=0 the first entry is the ground state itself, so the first gap is the second entry.)")
        isfile(joinpath(cfg.dir, "levels.csv")) && println("  All levels as a table: $(short(joinpath(cfg.dir, "levels.csv")))")
    elseif !isempty(ed_levels)
        println("\n  ED levels: $(short(joinpath(cfg.dir, "ed", "ed_levels.csv")))")
    end
end

function export_levels(cfg; dry)
    N = last(cfg.Ns)
    files = filter(isfile, [spectrum_file(cfg, N, D) for D in cfg.spectrum.Ds])
    script = joinpath(SRC, "export_bloch.jl")
    csv = joinpath(cfg.dir, "levels.csv")
    if dry
        planned = [spectrum_file(cfg, N, D) for D in cfg.spectrum.Ds]
        println("[table]    levels.csv\n           ", shown(script, planned, 1, []), " > ", short(csv))
    elseif !isempty(files)
        temporary = csv * ".tmp"
        run(pipeline(command(script, files, 1, []); stdout=temporary))
        mv(temporary, csv; force=true)
    end
end

# ----------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------
const USAGE = """
Usage:
    julia --project=. pumps.jl [options] CONFIG.toml

Runs the puMPS pipeline described by CONFIG.toml:
    ground state -> climb in N -> grow D -> spectrum -> levels.csv [-> ED]
Stages whose output already exists are skipped, so the same command resumes a run.

Options:
    --dry-run        print what would run, and the exact commands, without running anything
    --only STAGE     run only one part: ground, spectrum or ed
    --template       print a complete, commented config file and exit
    --params-table   print the parameter table (Markdown) and exit
    --help           show this message

Start from an example:
    julia --project=. pumps.jl examples/quickstart.toml
"""

function main(args)
    dry = false
    only_group = nothing
    files = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("--help", "-h")
            print(USAGE)
            return 0
        elseif a == "--template"
            print(template())
            return 0
        elseif a == "--params-table"
            print(params_table())
            return 0
        elseif a == "--dry-run"
            dry = true
        elseif a == "--only"
            i += 1
            i <= length(args) && args[i] in ("ground", "spectrum", "ed") || fail("--only needs one of: ground, spectrum, ed")
            only_group = Symbol(args[i])
        elseif startswith(a, "-")
            fail("unknown option $a\n\n$USAGE")
        else
            push!(files, a)
        end
        i += 1
    end
    length(files) == 1 || fail("give exactly one config file\n\n$USAGE")

    cfg = load_config(files[1])
    check_start(cfg)
    check_physics_record(cfg; write=!dry)
    only_group == :spectrum && cfg.spectrum === nothing && fail("--only spectrum, but the config has no [spectrum] table")
    only_group == :ed && cfg.ed === nothing && fail("--only ed, but the config does not have `enabled = true` under [ed]")
    stages = filter(s -> only_group === nothing || s.group == only_group, build_stages(cfg))

    cores = physical_cores()
    most = maximum(s.group == :ground ? cfg.blas : s.threads for s in stages)
    most > cores && println("warning: $most threads requested but this machine has $cores physical cores; ",
        "oversubscribed runs can be 50-100x slower.")

    println("puMPS run: x=$(cfg.x) m/e=$(cfg.me) theta=$(cfg.theta) hmax=$(cfg.hmax)   ->   $(short(cfg.dir))")
    if dry
        for (n, stage) in enumerate(stages)
            println("[$n/$(length(stages))] ", stage.title, isdone(stage) ? "   -- already done, would be skipped" : "")
            println("       ", shown(stage))
        end
        cfg.spectrum !== nothing && only_group in (nothing, :spectrum) && export_levels(cfg; dry=true)
        println("\nDry run: nothing was executed.")
        return 0
    end

    mkpath(cfg.dir)
    cp(cfg.file, joinpath(cfg.dir, "config_used.toml"); force=true)
    total = time()
    for (n, stage) in enumerate(stages)
        if isdone(stage)
            println("[$n/$(length(stages))] ", stage.title, "   -- already done, skipped")
            continue
        end
        absent = filter(!isfile, stage.inputs)
        isempty(absent) || error("stage `$(stage.label)` needs $(join(short.(absent), ", ")), which does not exist. " *
            "Run without --only so the earlier stages produce it.")
        println("[$n/$(length(stages))] ", stage.title)
        println("       ", shown(stage))
        seconds = run_stage(stage, joinpath(cfg.dir, "logs", stage.label * ".log"))
        println("       done in ", duration(seconds))
    end
    cfg.spectrum !== nothing && only_group in (nothing, :spectrum) && export_levels(cfg; dry=false)
    summarize(cfg)
    println("\nFinished in ", duration(time() - total), ".")
    0
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(main(ARGS))
    catch err
        if err isa ConfigError
            println(stderr, "config error: ", err.msg)
            exit(2)
        elseif err isa ErrorException
            println(stderr, "error: ", err.msg)
            exit(1)
        elseif err isa InterruptException
            println(stderr, "\ninterrupted. Finished stages are kept; run the same command to continue.")
            exit(130)
        end
        rethrow()
    end
end
