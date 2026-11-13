# Tests for pumps.jl.
#
#     julia --project=. test/runtests.jl          # config handling only (seconds)
#     julia --project=. test/runtests.jl --full   # also runs examples/quickstart.toml (about 8 minutes)
using Test, TOML, Serialization

const REPO = dirname(@__DIR__)
include(joinpath(REPO, "pumps.jl"))

const PUMPS = joinpath(REPO, "pumps.jl")
const QUICKSTART = joinpath(REPO, "examples", "quickstart.toml")

"Write a config into a fresh folder and return its path."
function config(text; dir=mktempdir())
    path = joinpath(dir, "run.toml")
    write(path, text)
    path
end

const BASE = """
[physics]
x = 10.0
me = 0.3
[ground]
N = 12
D = [4, 6]
"""

"Run pumps.jl as a user would; returns (exit code, combined output)."
function pumps(args...)
    out = IOBuffer()
    cmd = `$(Base.julia_cmd()) --project=$REPO --startup-file=no $PUMPS $(collect(args))`
    process = run(pipeline(ignorestatus(cmd); stdout=out, stderr=out))
    process.exitcode, String(take!(out))
end

rejects(text, fragment) = try
    load_config(config(text))
    false
catch err
    err isa ConfigError && occursin(fragment, err.msg)
end

@testset "config" begin
    cfg = load_config(QUICKSTART)
    @test (cfg.x, cfg.me, cfg.theta, cfg.hmax) == (10.0, 0.3, 0.0, 3)
    @test cfg.Ns == [12] && cfg.Ds == [4, 6]
    @test cfg.spectrum.momenta == [0, 1, 2, 3] && cfg.spectrum.Ds == [6] && cfg.spectrum.solver == "dense"
    @test cfg.ed === nothing
    @test cfg.dir == joinpath(REPO, "runs", "quickstart")

    # defaults are those of the scripts in src/
    cfg = load_config(config(BASE))
    @test cfg.tol == 1e-7 && cfg.maxit == 800 && cfg.blas == 1 && cfg.obcD == 4 && cfg.obc_sweeps == 12
    @test cfg.spectrum === nothing
    @test basename(cfg.dir) == "run" && basename(dirname(cfg.dir)) == "runs"

    # scalars and lists are both accepted; ED inherits the spectrum settings
    cfg = load_config(config(BASE * "[spectrum]\nmomenta = [0, 2]\nnev = 3\nat = \"all\"\n[ed]\nenabled = true\n"))
    @test cfg.spectrum.Ds == [4, 6] && cfg.ed.momenta == [0, 2] && cfg.ed.nev == 3
    @test load_config(config(replace(BASE, "D = [4, 6]" => "D = 8"))).Ds == [8]

    for example in readdir(joinpath(REPO, "examples"); join=true)
        @test load_config(example) isa NamedTuple
    end
end

@testset "config errors" begin
    @test rejects(replace(BASE, "N = 12" => "N = 13"), "even")
    @test rejects(replace(BASE, "N = 12" => "N = [16, 12]"), "ground.N must increase")
    @test rejects(replace(BASE, "D = [4, 6]" => "D = [6, 4]"), "ground.D must increase")
    @test rejects(replace(BASE, "x = 10.0" => "x = -1"), "positive")
    @test rejects(replace(BASE, "me = 0.3" => "me = \"0.3\""), "must be a number")
    @test rejects(replace(BASE, "me = 0.3\n" => ""), "missing required setting `me`")
    @test rejects(replace(BASE, "me = 0.3" => "me = 0.3\nthet = 1.0"), "unknown setting `thet`")
    @test rejects("N = 12\n" * BASE, "Put it under [ground]")
    @test rejects(BASE * "[spectra]\nnev = 2\n", "unknown section [spectra]")
    @test rejects(BASE * "obc_D = 2\n", "obc_D")
    @test rejects(BASE * "[spectrum]\nmomenta = [0, 6]\n", "0..5")
    @test rejects(BASE * "[spectrum]\nsolver = \"fast\"\n", "spectrum.solver")
    @test rejects(BASE * "start = \"nothing_here.jls\"\n", "not found")
    @test rejects("[ground]\nN = 12\nD = 4\n", "missing section [physics]")
end

@testset "stages" begin
    cfg = load_config(config(replace(BASE, "N = 12" => "N = [12, 16]") * "[spectrum]\nat = \"all\"\n[ed]\nenabled = true\n"))
    stages = build_stages(cfg)
    @test [s.label for s in stages] ==
          ["ground_N12_D4", "ground_N16_D4", "ground_N16_D6", "spectrum_N16_D4", "spectrum_N16_D6", "ed_N16"]
    @test [basename(only(s.inputs)) for s in stages[2:5]] ==
          ["ground_N12_D4.jls", "ground_N16_D4.jls", "ground_N16_D4.jls", "ground_N16_D6.jls"]
    # arguments in the order the scripts in src/ expect
    @test stages[3].args[2:end] == ["16", "10.0", "0.3", "0.0", "3", "6", "800", ground_file(cfg, 16, 6), "1.0e-7"]
    @test stages[5].args[2:end] == ["16", "10.0", "0.3", "0.0", "3", "0,1,2,3,4", "4", spectrum_file(cfg, 16, 6)]
    @test stages[6].args[1:7] == ["16", "10.0", "0.3", "0.0", "3", "0,1,2,3,4", "4"]
    @test all(s -> isfile(s.script), stages)
end

@testset "documentation matches the code" begin
    readme = read(joinpath(REPO, "README.md"), String)
    @test occursin(params_table(), readme)          # regenerate with: julia pumps.jl --params-table
    # the template is itself a valid config
    @test load_config(config(template())) isa NamedTuple
end

@testset "command line" begin
    path = config(BASE * "[spectrum]\nsolver = \"dense\"\n")
    code, out = pumps("--dry-run", path)
    @test code == 0
    @test occursin("src/run_seed_comparison.jl", out) && occursin("src/prepare_bond_ground_tol.jl", out)
    @test occursin("PUMPS_SOLVER=dense", out) && occursin("nothing was executed", out)
    @test !isdir(joinpath(dirname(path), "runs"))    # a dry run writes nothing

    code, out = pumps(config(replace(BASE, "N = 12" => "N = 13")))
    @test code == 2 && occursin("config error", out)
    @test first(pumps("--help")) == 0

    # a results folder is tied to one physical point
    dir = dirname(path)
    mkpath(joinpath(dir, "runs", "run"))
    write(joinpath(dir, "runs", "run", "physics.toml"), "x = 10.0\nme = 0.5\ntheta = 0.0\nhmax = 3\n")
    code, out = pumps("--dry-run", path)
    @test code == 2 && occursin("different physics", out) && occursin("me = 0.5 there, 0.3 in the config", out)
end

if "--full" in ARGS
    @testset "quickstart end to end" begin
        dir = mktempdir()
        settings = TOML.parsefile(QUICKSTART)
        settings["output"]["dir"] = "out"
        settings["ed"]["enabled"] = true
        path = joinpath(dir, "run.toml")
        open(io -> TOML.print(io, settings), path, "w")

        code, out = pumps(path)
        code == 0 || print(out)
        @test code == 0
        result(name) = deserialize(joinpath(dir, "out", name))

        # reference values: docs/manual_steps.md, and exact diagonalization
        @test isapprox(result("ground_N12_D4.jls").result.E / 12, -0.624784; atol=2e-6)
        ground = result("ground_N12_D6.jls").result
        @test isapprox(ground.E, -7.497977; atol=2e-5)
        @test ground.eta < 1e-6
        spectrum = result("spectrum_N12_D6.jls")
        @test spectrum.settings.momenta == [0, 1, 2, 3]
        gap = spectrum.levels[1][2] - spectrum.ground_energy
        @test isapprox(gap, 0.6659; atol=2e-4)
        ed = result(joinpath("ed", "ed_k0.jls")).levels
        @test isapprox(ed[1], -7.498096; atol=2e-6)
        @test isapprox(ed[2] - ed[1], 0.6643; atol=2e-4)
        @test ground.E > ed[1]                          # variational
        @test abs(gap - (ed[2] - ed[1])) < 5e-3
        @test countlines(joinpath(dir, "out", "levels.csv")) == 1 + 4 * 4

        # running again computes nothing
        code, out = pumps(path)
        @test code == 0 && count("already done, skipped", out) == 4 && !occursin("done in", out)

        # editing the physics without a new folder is refused
        settings["physics"]["me"] = 0.5
        open(io -> TOML.print(io, settings), path, "w")
        code, out = pumps(path)
        @test code == 2 && occursin("different physics", out)
    end
end
