using Serialization

for expr in deserialize(stdin)
    Core.eval(Main, expr)
end
