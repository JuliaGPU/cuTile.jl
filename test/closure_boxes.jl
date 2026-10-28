# load the package extensions
using DLFP8Types, Microfloats

# captured variables that get boxed (`Core.Box`) lose their type
if isdefined(Test, :detect_closure_boxes)
    exts = [Base.get_extension(cuTile, ext) for ext in (:DLFP8TypesExt, :MicrofloatsExt)]
    @test Test.detect_closure_boxes(cuTile, exts...) == []
end
