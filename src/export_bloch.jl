# Export one-insertion Bloch spectra (run_matrixfree_excitation.jl outputs) and ground
# caches to CSV so the Python analysis can compare with ED. Run inside the capsule:
#   julia --project=matrixfree_env --startup-file=no export_bloch.jl bloch/*.jls > bloch_levels.csv
using Serialization
println("file,N,D,x,me,theta,hmax,k,level,E,gap,ground_E,ground_eta,rank,seconds")
for f in ARGS
    s = deserialize(f); p = s.physics
    # D is not stored in the excitation output; recover it from the ground cache named in the file
    D = hasproperty(s, :D) ? s.D : (m = match(r"_D(\d+)_", basename(f)); m === nothing ? -1 : parse(Int, m.captures[1]))
    for (ik, k) in enumerate(s.settings.momenta)
        lv = s.levels[ik]; rk = s.ranks[ik]
        for (j, E) in enumerate(lv)
            println(join([basename(f), p.N, D, p.x, p.me, p.theta, p.hmax, k, j - 1, E, E - s.ground_energy,
                          s.ground_energy, "", rk, s.seconds], ","))
        end
    end
end
