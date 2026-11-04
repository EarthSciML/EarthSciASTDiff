# Structural sharing (hashcons.jl `_Interner`): one object per distinct
# subtree, structure unchanged.

using EarthSciASTDiff: skey, sid, StructIndex, _Interner, _intern

# Every operator node reachable from `es`, by identity.
function _op_nodes(es)
    seen = IdDict{OpExpr, Nothing}()
    todo = ASTExpr[e for e in es if e !== nothing]
    while !isempty(todo)
        e = pop!(todo)
        (e isa OpExpr && !haskey(seen, e)) || continue
        seen[e] = nothing
        EarthSciAST.map_children(c -> (push!(todo, c); c), e)
    end
    return collect(keys(seen))
end

# Whether the trees `es` hold exactly one object per distinct subtree.
function _one_per_subtree(es)
    nodes = _op_nodes(es)
    ix = StructIndex()
    return length(nodes) == length(Set(sid(ix, n) for n in nodes))
end

_band_exprs(entries) =
    ASTExpr[x for en in entries for x in (en.band.coef, en.band.guard, en.band.cidx...) if x !== nothing]

@testset "interning" begin
    x = VarExpr("x")
    tx() = OpExpr("*", ASTExpr[IntExpr(2), x])
    a = OpExpr("+", ASTExpr[tx(), IntExpr(1)])
    b = OpExpr("-", ASTExpr[tx()])
    t = _Interner()
    a2, b2 = _intern(t, a), _intern(t, b)
    # structure (and so serialization) is unchanged
    @test skey(a2) == skey(a) && skey(b2) == skey(b)
    # the equal `2*x` subtrees are now one object
    @test a2.args[1] === b2.args[1]
    # the first occurrence of a subtree is its own representative
    @test a2 === a
    # a structurally equal copy maps onto it, root included
    @test _intern(t, OpExpr("+", ASTExpr[tx(), IntExpr(1)])) === a
    # equal-looking but structurally different trees stay apart
    c = _intern(t, OpExpr("+", ASTExpr[tx(), NumExpr(1.0)]))
    @test c !== a && skey(c) != skey(a) && c.args[1] === a.args[1]
    # leaves pass through
    @test _intern(t, x) === x
    @test _one_per_subtree(ASTExpr[a2, b2, c])
end

@testset "interning: band lists share every distinct subtree" begin
    file = fixture("minmod_adv.esm")
    model = first(keys(file.models))
    e1 = jacobian_bands(file, model)
    @test _one_per_subtree(_band_exprs(e1))
    file = fixture("arrhenius_chem.esm")
    for wrt in ((:states, :states), (:states, :parameters))
        e2 = jacobian_bands2(file, "Chem"; wrt = wrt)
        @test !isempty(e2)
        @test _one_per_subtree(_band_exprs(e2))
    end
end
