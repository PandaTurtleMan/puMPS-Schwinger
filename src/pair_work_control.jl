# Optional cooperative deadlines and content-addressed contraction checkpoints.
# Checkpoints store completed additive contributions, never partially mutated H.
using Serialization, SHA

struct PairBudgetExceeded <: Exception end
Base.showerror(io::IO,::PairBudgetExceeded)=print(io,"pair work budget reached at a safe boundary")
mutable struct PairWorkControl
    directory::Union{Nothing,String}
    deadline_ns::Float64
    columns::Int
    stop_after_chunks::Int
    completed::Threads.Atomic{Int}
    source_id::String
end
function pairSourceDigest()
    paths=sort(filter(p->endswith(p,".jl"),readdir(@__DIR__;join=true)))
    io=IOBuffer()
    write(io,string(VERSION))
    for p in paths;write(io,basename(p));write(io,read(p));end
    for name in ("Project.toml","Manifest.toml")
        p=joinpath(@__DIR__,"matrixfree_env",name);isfile(p)&&write(io,read(p))
    end
    bytes2hex(sha256(take!(io)))
end
function PairWorkControl(;directory=nothing,max_seconds=Inf,columns=4,stop_after_chunks=typemax(Int))
    max_seconds>0 && columns>0 && stop_after_chunks>0 || throw(ArgumentError("positive work limits required"))
    dir=isnothing(directory) ? nothing : abspath(directory)
    isnothing(dir)||mkpath(dir)
    PairWorkControl(dir,Float64(time_ns())+1e9*max_seconds,columns,stop_after_chunks,Threads.Atomic{Int}(0),pairSourceDigest())
end
checkPairWork(::Nothing)=nothing
function checkPairWork(c::PairWorkControl)
    (Float64(time_ns())>=c.deadline_ns || c.completed[]>=c.stop_after_chunks) && throw(PairBudgetExceeded())
    nothing
end
function pairDigest(x)
    io=IOBuffer();serialize(io,x);bytes2hex(sha256(take!(io)))
end
function pairAtomicSave(path,x)
    mkpath(dirname(path));tmp,io=mktemp(dirname(path))
    try
        serialize(io,x);flush(io);close(io);mv(tmp,path;force=true)
    finally
        isopen(io)&&close(io)
        isfile(tmp)&&rm(tmp)
    end
    path
end
pairCheckpointKey(::Nothing,args...)=nothing
function pairCheckpointKey(c::PairWorkControl,args...)
    isnothing(c.directory) ? nothing : joinpath(c.directory,pairDigest((c.source_id,args)))
end
function pairLoadChunk(key,range)
    isnothing(key)&&return nothing
    path=joinpath(key,"$(first(range))_$(last(range)).jls")
    isfile(path)||return nothing
    x=deserialize(path)
    x.range==range && x.digest==pairDigest(x.blocks) || error("corrupt pair checkpoint: $path")
    x.blocks
end
function pairSaveChunk(c,key,range,blocks)
    isnothing(key)&&return
    pairAtomicSave(joinpath(key,"$(first(range))_$(last(range)).jls"),
        (range=range,blocks=blocks,digest=pairDigest(blocks)))
    Threads.atomic_add!(c.completed,1)
end
# Julia wraps errors from worker tasks in TaskFailedException/CompositeException.
pairBudgetError(e::PairBudgetExceeded)=true
pairBudgetError(e::CompositeException)=all(pairBudgetError,e.exceptions)
pairBudgetError(e::TaskFailedException)=pairBudgetError(e.task.result)
pairBudgetError(e)=false

# Account for failed/interrupted attempts as well as replay and remaining work.
# Otherwise a resumed many-hour stage would report only its final short slice.
function pairTimedPhase(f,path)
    start=time_ns();value=nothing;completed=false;elapsed=0.;attempts=NamedTuple[]
    try
        value=f();completed=true
    finally
        elapsed=(time_ns()-start)/1e9
        attempts=isfile(path) ? deserialize(path) : NamedTuple[]
        push!(attempts,(seconds=elapsed,completed=completed))
        pairAtomicSave(path,attempts)
    end
    (value=value,time=elapsed,total_seconds=sum(a.seconds for a in attempts),attempts=length(attempts))
end
