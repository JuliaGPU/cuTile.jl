@testset "lazy toolchain discovery" begin
    script = """
        using cuTile
        @assert cuTile.tileir_toolchain_cache.value === nothing
        @assert cuTile.tileir_disassembler_cache.value === nothing
        cuTile.bytecode_version()
        @assert cuTile.tileir_toolchain_cache.value !== nothing
        @assert cuTile.tileir_disassembler_cache.value === nothing
        """
    project = dirname(Base.active_project())
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project -e $script`
    @test success(run(ignorestatus(cmd)))
end

@testset "compiler timeout" begin
    @test cuTile.parse_compiler_timeout(nothing) === nothing
    @test cuTile.parse_compiler_timeout(1) == 1.0
    @test_throws ArgumentError cuTile.parse_compiler_timeout(true)
    @test_throws ArgumentError cuTile.parse_compiler_timeout(0)

    err = try
        cuTile.run_and_collect(`$(Base.julia_cmd()) -e 'sleep(10)'`; timeout=0.1)
        nothing
    catch err
        err
    end
    @test err isa cuTile.TileCompilerTimeoutError
    @test occursin("reducing the tile size", sprint(showerror, err))
end
