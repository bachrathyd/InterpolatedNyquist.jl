# minimal PNG writer (stored deflate blocks), no packages
function crc32(data)
    c = 0xffffffff
    for b in data
        c ⊻= b
        for _ in 1:8
            c = (c & 1) != 0 ? (0xedb88320 ⊻ (c >> 1)) : (c >> 1)
        end
    end
    return ~c
end
function adler(data)
    a, b = UInt32(1), UInt32(0)
    for x in data
        a = (a + x) % 65521; b = (b + a) % 65521
    end
    return (b << 16) | a
end
be32(x) = reinterpret(UInt8, [hton(UInt32(x))])
chunk(t, d) = vcat(be32(length(d)), Vector{UInt8}(t), d, be32(crc32(vcat(Vector{UInt8}(t), d))))
function writepng(fn, rgb::Array{UInt8,3})  # (3, w, h)
    _, w, h = size(rgb)
    raw = UInt8[]
    for y in 1:h
        push!(raw, 0x00); append!(raw, vec(rgb[:, :, y]))
    end
    z = UInt8[0x78, 0x01]
    i = 1
    while i <= length(raw)
        j = min(i + 65534, length(raw)); n = j - i + 1
        push!(z, j == length(raw) ? 0x01 : 0x00)
        append!(z, reinterpret(UInt8, [UInt16(n)])); append!(z, reinterpret(UInt8, [~UInt16(n)]))
        append!(z, raw[i:j]); i = j + 1
    end
    append!(z, be32(adler(raw)))
    ihdr = vcat(be32(w), be32(h), UInt8[8, 2, 0, 0, 0])
    open(fn, "w") do io
        write(io, UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
        write(io, chunk("IHDR", ihdr)); write(io, chunk("IDAT", z)); write(io, chunk("IEND", UInt8[]))
    end
end
