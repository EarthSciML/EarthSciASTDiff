# Structural interning — the "are these two subtrees equal?" primitive, at a
# cost that does not grow with subtree SIZE.
#
# WHY THIS EXISTS. [`skey`](@ref) answers structural equality by serializing
# the FULL subtree. That is fine once, but every pass that asks the question
# at each node of a tree then pays O(|subtree|) per node — Θ(n·depth) on the
# tree, quadratic on the deep `ifelse` chains that `index(makearray…)`
# lowering (inline.jl) and the product rule build. A flattened Earth-system
# document is exactly where that bites: `flatten` materializes an operator's
# stencil PER CELL, so one D-equation's RHS grows linearly with the grid
# (ReSEACT at 6×6×8: 420k nodes per equation, 5.0M over the system), and a
# Θ(n·depth) pass over it does not finish.
#
# WHAT IT DOES. Bottom-up interning: each structurally distinct subtree gets
# an integer id, so equality is an `Int` compare. A node's table key is its
# own SHELL — the node with every child replaced by a placeholder carrying
# that child's id — which is O(node fields), not O(subtree). Keying a whole
# tree is therefore Θ(n) and one comparison is O(1). This is the same trick
# `_Intern` (cse.jl) uses; that one keeps occurrence counts and
# representatives because CSE needs them, this is the equality-only version
# and it additionally memoizes on node IDENTITY, so a structurally shared
# subtree (`map_children` is identity-preserving, so rewrites keep sharing)
# is keyed once no matter how many parents reach it.
#
# SCOPE OF AN INDEX. Ids are meaningful only within one `StructIndex`; two
# indices assign unrelated numbers. Create one per pass and thread it — never
# compare ids across indices.

"""
    StructIndex()

Interning table for structural expression equality. See [`sid`](@ref).
"""
struct StructIndex
    ids::Dict{String,Int}        # shell key → id
    memo::IdDict{OpExpr,Int}     # node identity → id (0 is "absent")
end
StructIndex() = StructIndex(Dict{String,Int}(), IdDict{OpExpr,Int}())

# Distinct from cse.jl's `__cse⟦…⟧` so a tree that already carries CSE
# placeholders interns without collision.
_sid_placeholder(id::Int) = VarExpr("__sid⟦$id⟧")

"""
    sid(ix::StructIndex, e::ASTExpr) -> Int

Structural id of `e` in `ix`: `sid(ix, a) == sid(ix, b)` iff `skey(a) ==
skey(b)`. Θ(nodes) to key a tree, O(1) per repeated query.
"""
function sid(ix::StructIndex, e::ASTExpr)::Int
    if e isa OpExpr
        cached = get(ix.memo, e, 0)
        cached == 0 || return cached
        kids = Int[]
        map_children(c -> (push!(kids, sid(ix, c)); c), e)
        i = Ref(0)
        shell = map_children(_ -> (i[] += 1; _sid_placeholder(kids[i[]])), e)
        id = _intern_key!(ix, stable_json(EarthSciAST.serialize_expression(shell)))
        ix.memo[e] = id
        return id
    end
    # Leaves are immutable, so `===` identity is not a reliable memo key for
    # them (types.jl); their shell key IS the whole node and costs O(1).
    return _intern_key!(ix, stable_json(EarthSciAST.serialize_expression(e)))
end

_intern_key!(ix::StructIndex, k::String)::Int =
    get!(ix.ids, k, length(ix.ids) + 1)
