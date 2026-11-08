# Structural interning — the "are these two subtrees equal?" primitive, at a
# cost that does not grow with subtree SIZE.
#
# WHY THIS EXISTS. [`skey`](@ref) answers structural equality by serializing
# the FULL subtree. That is fine once, but every pass that asks the question
# at each node of a tree then pays O(|subtree|) per node — Θ(n·depth) on the
# tree, quadratic on the deep `ifelse` chains that `index(makearray…)`
# lowering (inline.jl) and the product rule build. A coupled Earth-system
# document is exactly where that bites: lowering one 3-D flux-divergence
# operator produces a very large expression, and the coupled RHS of a single
# state carries it (MEASURED on ReSEACT: 4.2×10⁵ nodes per D-equation,
# 5.0×10⁶ over the system, and a genuine TREE — the path count and the
# distinct-node count agree to 1.0x, so there is no sharing to lean on).
#
# WHAT IT DOES. Bottom-up interning: each structurally distinct subtree gets
# an integer id, so equality is an `Int` compare. A node's table key is its
# own SHELL — the node with every child replaced by a placeholder carrying
# that child's id — which is O(node fields), not O(subtree); a node with no
# field but `op` and `args` (most of them) is keyed by its operator and
# children's ids directly, with no serialization at all. Keying a whole
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
    ids::Dict{String, Int}        # shell key → id
    # leaves, by value (`Dict` keys compare by `isequal`: `0.0` and `-0.0`
    # differ, every `NaN` is one key, exactly as their `string`s do)
    vars::Dict{String, Int}
    ints::Dict{Int64, Int}
    floats::Dict{Float64, Int}
    # an operator node with no field but `op` and `args`, by operator and
    # argument ids: the bulk of a coefficient, keyed without serializing
    plain::Dict{Tuple{String, Vector{Int}}, Int}
    n::Base.RefValue{Int}         # ids handed out; both tables draw from it
    memo::IdDict{OpExpr, Int}     # node identity → id (0 is "absent")
end
StructIndex() = StructIndex(
    Dict{String, Int}(), Dict{String, Int}(), Dict{Int64, Int}(), Dict{Float64, Int}(),
    Dict{Tuple{String, Vector{Int}}, Int}(), Ref(0), IdDict{OpExpr, Int}()
)

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
        id = _shell_id!(ix, e, kids)
        ix.memo[e] = id
        return id
    end
    # Leaves are immutable, so `===` identity is not a reliable memo key for
    # them (types.jl) and every occurrence is re-keyed. Roughly half the nodes
    # of a lowered RHS are leaves, so they are keyed by value, in a table per
    # kind, instead of by a serialize + JSON round trip (or a key string built
    # per occurrence). `IntExpr(1)` and `NumExpr(1.0)` are different values
    # (types.jl) and land in different tables, as they must.
    e isa VarExpr && return _id_in!(ix.vars, ix.n, e.name)
    e isa IntExpr && return _id_in!(ix.ints, ix.n, e.value)
    e isa NumExpr && return _id_in!(ix.floats, ix.n, e.value)
    return _intern_key!(ix, stable_json(EarthSciAST.serialize_expression(e)))
end

_intern_key!(ix::StructIndex, k::String)::Int = _id_in!(ix.ids, ix.n, k)
_id_in!(d::Dict{K, Int}, n::Base.RefValue{Int}, k::K) where {K} =
    get!(() -> (n[] += 1), d, k)

# Whether `e` has no field but `op` and `args` set: it then serializes as
# `{"op", "args"}` alone, since `serialize_expression` writes a field only
# when it is not `nothing`.
@generated function _plain_node(e::OpExpr)
    fs = [f for f in fieldnames(OpExpr) if f !== :op && f !== :args]
    isempty(fs) && return true
    return Expr(:&&, (:(getfield(e, $(QuoteNode(f))) === nothing) for f in fs)...)
end

# The id of operator node `e` whose children have ids `kids`. A node with
# only `op` and `args` is keyed by its operator and argument ids, with no
# serialization. Any other is keyed by its SHELL, every child replaced by a
# placeholder carrying that child's id; a shell that serializes as
# `{"op", "args"}` alone (a set field that is never written) joins the plain
# table, so the two tables split nodes exactly where `skey` does.
function _shell_id!(ix::StructIndex, e::OpExpr, kids::Vector{Int})::Int
    _plain_node(e) && return _id_in!(ix.plain, ix.n, (e.op, kids))
    i = Ref(0)
    shell = map_children(_ -> (i[] += 1; _sid_placeholder(kids[i[]])), e)
    d = EarthSciAST.serialize_expression(shell)
    length(d) == 2 && haskey(d, "op") && haskey(d, "args") &&
        return _id_in!(ix.plain, ix.n, (e.op, kids))
    return _intern_key!(ix, stable_json(d))
end

# ── the cheap half: a structural hash used as a NEGATIVE filter ──────────────
#
# `sid` is exact, but it still pays a table insert per distinct node (and a
# shallow serialization for a node with fields beyond `op`/`args`), and the
# band calculus asks "are these two subtrees equal?" once per node — where
# the answer is almost always NO. A hash that equal trees are GUARANTEED to
# share turns that common case into one `UInt` compare and leaves `sid` to
# adjudicate the rare match. Measured by sampling the ReSEACT
# whole system, interning inside `simplify_branches` was 36% of
# `jacobian_bands` once the Θ(n·depth) passes were fixed; nearly all of it is
# this negative case.
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
    memo::IdDict{OpExpr, UInt}    # 0 is "absent"
end
StructHash() = StructHash(IdDict{OpExpr, UInt}())

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

# ── sharing: one object per distinct subtree ─────────────────────────────────
#
# The band calculus builds every coefficient on its own: a derivative, then
# `simplify`, then `normalize_band`'s substitution. Equal subtrees of
# different coefficients — the stencil weights, the upwind switches and the
# face fluxes that every band of one region of a lowered transport operator
# carries — come out as distinct objects. On ReSEACT's advection term the
# 146 bands' coefficients are 2.1×10⁶ tree nodes but only 6×10³ distinct
# subtrees, and they held 209 MiB. Interning maps each subtree to ONE
# representative per structural id ([`sid`](@ref)), so every coefficient
# references the shared copies instead of holding its own. The trees are
# unchanged structurally (same `skey`, same serialization); only identity
# changes, which nothing downstream may observe (no pass mutates a node).

#
# One bottom-up pass: a node's id is keyed on its shell over its children's
# ids (exactly as `sid` keys it), and its representative is rebuilt from the
# children's representatives, so each node is visited and shell-serialized
# once.

struct _Interner
    ix::StructIndex
    canon::Dict{Int, OpExpr}                  # structural id → representative
    memo::IdDict{OpExpr, Tuple{OpExpr, Int}}  # node identity → (representative, id)
end
_Interner() = _Interner(StructIndex(), Dict{Int, OpExpr}(), IdDict{OpExpr, Tuple{OpExpr, Int}}())

_intern(t::_Interner, e::ASTExpr)::ASTExpr = first(_intern_id(t, e))

# (representative, structural id) of `e`. A leaf is its own representative.
function _intern_id(t::_Interner, e::ASTExpr)::Tuple{ASTExpr, Int}
    e isa OpExpr || return (e, sid(t.ix, e))
    r = get(t.memo, e, nothing)
    r === nothing || return r
    kids = Tuple{ASTExpr, Int}[]
    map_children(c -> (push!(kids, _intern_id(t, c)); c), e)
    id = _shell_id!(t.ix, e, Int[k[2] for k in kids])
    c = get(t.canon, id, nothing)
    if c === nothing
        i = Ref(0)
        c = map_children(_ -> (i[] += 1; kids[i[]][1]), e)
        t.canon[id] = c
    end
    t.memo[e] = (c, id)
    return (c, id)
end
