spec1d = ct.ArraySpec{1}(16, true, (0,), (16,))
AT = ct.TileArray{Float32, 1, Int32, spec1d}

@testset "programmatic dependent launch" begin
    if ct.tileir_disassembler_version() >= v"13.4"
        @test @filecheck begin
            @check "[[STORE:%.+]] = store_view_tko"
            @check "[[JOIN:%.+]] = join_tokens{{.*}}[[STORE]]"
            @check "gdc_launch_dependents_tko token = [[JOIN]]"
            code_tiled(Tuple{AT, AT}; bytecode_version=v"13.4") do a, b
                ct.store(b, 1, ct.load(a, 1, (16,)))
                ct.grid_dependency_control_launch_dependents()
                return
            end
        end

        @test @filecheck begin
            @check "[[WAIT:%.+]] = gdc_wait_tko"
            @check "[[JOIN:%.+]] = join_tokens{{.*}}[[WAIT]]"
            @check "load_view_tko{{.*}} token = [[JOIN]]"
            code_tiled(Tuple{AT, AT, AT}; bytecode_version=v"13.4") do a, b, c
                Base.donotdelete(ct.load(a, 1, (16,)))
                ct.grid_dependency_control_wait()
                ct.store(c, 1, ct.load(b, 1, (16,)))
                return
            end
        end

        @test @filecheck begin
            @check "if"
            @check "gdc_wait_tko"
            code_tiled(Tuple{Bool}; bytecode_version=v"13.4") do condition
                if condition
                    ct.grid_dependency_control_wait()
                end
                return
            end
        end
    end

    @test_throws "requires Tile IR v13.4+" code_tiled(
        Tuple{}; bytecode_version=v"13.3") do
        ct.grid_dependency_control_wait()
        return
    end
end
