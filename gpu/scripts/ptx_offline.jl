# PTX of a kernel WITHOUT a GPU (from the code review): GPUCompiler + the NVPTX back end + CUDA's
# device method table, no libdevice and no ptxas (no register counts). Catches InvalidIRError
# (dynamic calls, allocations, Float64) and shows local memory, calls and the instruction mix.
using CUDA
const CC = CUDA.CUDACore
const GPUC = CC.GPUCompiler
const LLV = CC.LLVM

struct RevParams <: CC.AbstractCUDACompilerParams
    sm::CC.SMVersion
    ptx::VersionNumber
end
GPUC.link_libraries!(job::GPUC.CompilerJob{GPUC.PTXCompilerTarget, RevParams}, mod::LLV.Module) = nothing
GPUC.isintrinsic(job::GPUC.CompilerJob{GPUC.PTXCompilerTarget, RevParams}, fn::String) =
    startswith(fn, "__nv_") || invoke(GPUC.isintrinsic,
        Tuple{GPUC.CompilerJob{GPUC.PTXCompilerTarget, <:CC.AbstractCUDACompilerParams}, String}, job, fn)

function ptx_of(f, tt; cap = v"8.9", ptx = v"8.4")
    target = GPUC.PTXCompilerTarget(; cap, ptx, debuginfo = false)
    params = RevParams(CC.SMVersion(cap), ptx)
    config = GPUC.CompilerConfig(target, params; kernel = true, name = nothing, always_inline = true)
    source = GPUC.methodinstance(typeof(f), Base.to_tuple_type(tt))
    job = GPUC.CompilerJob(source, config)
    return GPUC.JuliaContext() do ctx
        String(GPUC.compile(:asm, job)[1])
    end
end

const DevVec{T} = CC.CuDeviceArray{T, 1, CC.AS.Global}

function ptxstats(asm)
    lines = split(asm, '\n')
    ins = [strip(l) for l in lines if occursin(r"^\s+[a-z@%]", l) && !occursin(r"^\s+(\.|//|\{|\})", l)]
    op(l) = (m = match(r"^(@%p\d+\s+)?([a-z][a-z0-9_.]*)", l); m === nothing ? "" : m[2])
    ops = op.(ins)
    has(r) = count(o -> occursin(r, o), ops)
    depot = [parse(Int, m[1]) for m in eachmatch(r"__local_depot\d+\[(\d+)\]", asm)]
    funcs = [m[1] for m in eachmatch(r"\.func\s+(?:\([^)]*\)\s*)?([A-Za-z_$][\w$]*)", asm)]
    return (n = length(ops), local_depot = depot, f64 = has(r"\.f64"), f32 = has(r"\.f32"),
        f16 = has(r"\.f16"), f16x2 = has(r"f16x2"), cvt16 = has(r"^cvt.*f16"), fma32 = has(r"^fma.*\.f32"),
        mul32 = has(r"^mul.*\.f32"), add32 = has(r"^(add|sub).*\.f32"), div32 = has(r"^(div|rcp).*\.f32"),
        fma16 = has(r"^fma.*\.f16"), mul16 = has(r"^mul.*\.f16"), add16 = has(r"^(add|sub).*\.f16"),
        ldlocal = has(r"^ld\.local"), stlocal = has(r"^st\.local"), ldparam = has(r"^ld\.param"),
        ldglobal = has(r"^ld\.global"), calls = has(r"^call"), bra = has(r"^bra"),
        nvcalls = sort(unique([m[1] for m in eachmatch(r"call\.uni[^,]*,\s*(__nv_\w+)|call[^;]*?(__nv_\w+)", asm) if m[1] !== nothing])),
        funcs = funcs)
end
