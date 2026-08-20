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
    # them (types.jl) and every occurrence is re-keyed. Roughly half the nodes
    # of a lowered RHS are leaves, so they get a direct key instead of a
    # serialize + JSON round trip. Any INJECTIVE encoding will do — ids are
    # private to one index — and the `{` an `OpExpr` shell key always starts
    # with keeps the two spaces apart. `IntExpr(1)` and `NumExpr(1.0)` are
    # different values (types.jl) and get different keys, as they must.
    e isa VarExpr && return _intern_key!(ix, "v" * e.name)
    e isa IntExpr && return _intern_key!(ix, "i" * string(e.value))
    e isa NumExpr && return _intern_key!(ix, "n" * string(e.value))
    return _intern_key!(ix, stable_json(EarthSciAST.serialize_expression(e)))
end

_intern_key!(ix::StructIndex, k::String)::Int =
    get!(ix.ids, k, length(ix.ids) + 1)

# ── the cheap half: a structural hash used as a NEGATIVE filter ──────────────
#
# `sid` is exact, but it still pays a (shallow) serialization per distinct
# node, and the band calculus asks "are these two subtrees equal?" once per
# node — where the answer is almost always NO. A hash that equal trees are
# GUARANTEED to share turns that common case into one `UInt` compare and
# leaves `sid` to adjudicate the rare match. Measured on the ReSEACT whole
# system at 6×6×8, interning inside `simplify_branches` was 36% of
# `jacobian_bands` after the Θ(n·depth) passes were fixed; nearly all of it
# is this negative case.
#
# The hash covers the operator and the children only — NOT the non-expression
# fields (`name`, `value`, `output_idx`, `ranges`, …) — so unequal trees may
# collide. That is sound BY CONSTRUCTION here and nowhere else: a mismatch
# proves inequality, a match proves nothing and must be confirmed with
# [`sid`](@ref). Never use it as an identity.

"""
    StructHash()

Identity-memoized structural hash. See [`shash`](@ref).
"""
struct StructHash
    memo::IdDict{OpExpr,UInt}    # 0 is "absent"
end
StructHash() = StructHash(IdDict{OpExpr,UInt}())

"""
    shash(hx::StructHash, e::ASTExpr) -> UInt

Hash of `e`'s operator/child structure. `skey(a) == skey(b)` implies
`shash(hx, a) == shash(hx, b)`; the converse does NOT hold. O(1) per node,
memoized on node identity.
"""
function shash(hx::StructHash, e::ASTExpr)::UInt
    e isa VarExpr && return hash(e.name, 0x9e3779b97f4a7c11 % UInt)
    e isa IntExpr && return hash(e.value, 0xc2b2ae3d27d4eb4f % UInt)
    e isa NumExpr && return hash(e.value, 0x165667b19e3779f9 % UInt)
    cached = get(hx.memo, e, UInt(0))
    cached == 0 || return cached
    h = Ref(hash(e.op, 0x27d4eb2f165667c5 % UInt))
    map_children(c -> (h[] = hash(shash(hx, c), h[]); c), e)
    h[] == 0 && (h[] = one(UInt))          # keep 0 as the absent sentinel
    hx.memo[e] = h[]
    return h[]
end
