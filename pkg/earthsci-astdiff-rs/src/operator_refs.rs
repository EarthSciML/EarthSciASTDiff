//! Operator references (esm-spec §9.6): the ONE module of this crate that
//! knows what an `apply_operator` node is (the reference's operator_refs.jl,
//! which documents the design in full).
//!
//! A reference `{op: apply_operator, name: D, bindings: B}` is never
//! expanded. Each definition is analysed once (per binding shape where the
//! analysis needs shapes) and the result is reused at every use:
//!
//! * a SCALAR definition is differentiated once per parameter into a derived
//!   definition `<D>.d_<p>`; the derivative table's reference row is
//!   `∂D(B)/∂s = Σ_p D.d_p(B) · ∂B[p]/∂s` ([`dref`]);
//! * an ARRAY (stencil) definition is differentiated once per parameter and
//!   binding shape into bands of its own index space ([`def_bands`]), each
//!   coefficient a derived definition `<D>.d_<p>.b<k>@<shape>`; a use reads the array
//!   as an intermediate placeholder whose bands are composed with the
//!   definition's ([`ref_bands`], [`compose`]).
//!
//! Derived documents carry the definitions their references reach
//! ([`operators_blocks`]). Both spellings of the op, `apply_operator` and
//! `apply_expression_template`, are read alike.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use earthsci_ast::extension::types::RangeSpec;
use earthsci_ast::{Expr, ExpressionNode};
use indexmap::{IndexMap, IndexSet};
use serde_json::Value;

use crate::bands::{
    bands_in, merge_bands, normalize_band, placeholder, range_of, region_bounds, subst_free, Band,
    BandError, Ctx,
};
use crate::branches::{canon_lits, simp};
use crate::clip::RIdx;
use crate::helpers::{add, free_vars, is_zero, mul, op, ser_expr, skey, Site, SiteScan};
use crate::inline::{all_names, binders};
use crate::reductions::{conj, conj_list, fresh, split_guard};
use crate::rules::{dscalar_in, DerivativeRuleError};
use crate::simplify_port::simplify_jl;

type R<T> = Result<T, BandError>;

pub const REF_OPS: [&str; 2] = ["apply_operator", "apply_expression_template"];

/// The node of an operator reference, or `None`.
pub fn ref_node(e: &Expr) -> Option<&ExpressionNode> {
    match e {
        Expr::Operator(n) if REF_OPS.contains(&n.op.as_str()) => Some(n),
        _ => None,
    }
}

pub fn is_ref(e: &Expr) -> bool {
    ref_node(e).is_some()
}

/// A parsed definition and its parameters' roles (esm-spec §9.6.11).
#[derive(Debug)]
pub(crate) struct DefInfo {
    pub name: String,
    pub params: Vec<String>,
    pub body: Expr,
    pub entry: Value,
    pub value: HashSet<String>,
    pub array: HashSet<String>,
    pub structural: HashSet<String>,
    pub rank: usize,
}

#[derive(Default)]
struct State {
    defs: HashMap<String, Rc<DefInfo>>,
    derived: IndexMap<String, Value>,
    dparam: HashMap<(String, String), Expr>,
    defbands: HashMap<String, Rc<Vec<Band>>>,
    extents: HashMap<String, (Vec<i64>, Vec<i64>)>,
}

/// The reference-aware analysis state of one view: the registry its
/// references resolve against, the derived definitions written, and the
/// memo tables that keep each analysis to once per definition (and binding
/// shape).
pub struct RefLayer {
    pub registry: IndexMap<String, Value>,
    pub refop: String,
    pub index_sets: HashMap<String, i64>,
    st: RefCell<State>,
}

impl std::fmt::Debug for RefLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RefLayer({} definitions)", self.registry.len())
    }
}

impl RefLayer {
    pub fn new(registry: &IndexMap<String, Value>, index_sets: HashMap<String, i64>) -> RefLayer {
        // EarthSciAST >= 0.15 reads `apply_expression_template` as
        // `apply_operator` at load (esm-spec §9.6.12), so derived references
        // use the current name.
        RefLayer {
            registry: registry.clone(),
            refop: "apply_operator".to_string(),
            index_sets,
            st: RefCell::new(State::default()),
        }
    }

    /// The derived definitions written so far, in creation order.
    pub fn derived(&self) -> IndexMap<String, Value> {
        self.st.borrow().derived.clone()
    }
}

// ── the current layer and the definition mode ───────────────────────────────

thread_local! {
    static LAYER: RefCell<Vec<Rc<RefLayer>>> = const { RefCell::new(Vec::new()) };
    static DEFPARAMS: RefCell<Vec<Option<HashSet<String>>>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn with_layer<T>(l: &Option<Rc<RefLayer>>, f: impl FnOnce() -> T) -> T {
    let Some(l) = l else { return f() };
    LAYER.with(|s| s.borrow_mut().push(l.clone()));
    let out = f();
    LAYER.with(|s| s.borrow_mut().pop());
    out
}

pub(crate) fn current_layer() -> Option<Rc<RefLayer>> {
    LAYER.with(|s| s.borrow().last().cloned())
}

fn layer() -> Result<Rc<RefLayer>, DerivativeRuleError> {
    current_layer().ok_or_else(|| {
        DerivativeRuleError(
            "an operator reference (`apply_operator`) reached the band calculus without \
             the registry of the system it belongs to"
                .into(),
        )
    })
}

fn with_defparams<T>(ps: Option<HashSet<String>>, f: impl FnOnce() -> T) -> T {
    DEFPARAMS.with(|s| s.borrow_mut().push(ps));
    let out = f();
    DEFPARAMS.with(|s| s.borrow_mut().pop());
    out
}

/// Inside a definition's body two parameters are independent inputs: an
/// array parameter read at a subscript that depends on another is
/// piecewise constant in it (the derivative table's `index` rule).
pub(crate) fn index_is_independent_param(n: &ExpressionNode, s: &Site) -> bool {
    let site_var = match &s.expr {
        Expr::Variable(v) => v.clone(),
        Expr::Operator(sn) => match sn.args.first() {
            Some(Expr::Variable(v)) => v.clone(),
            _ => return false,
        },
        _ => return false,
    };
    let Some(Expr::Variable(a)) = n.args.first() else {
        return false;
    };
    DEFPARAMS
        .with(|st| matches!(st.borrow().last(), Some(Some(ps)) if ps.contains(a) && *a != site_var))
}

// ── definitions ─────────────────────────────────────────────────────────────

/// `<D>.d_<p>` (a scalar definition's derivative) or `<D>.d_<p>.b<k>@<shape>`
/// (band k of an array definition's at one binding shape, [`shape_tag`]):
/// (D, p, (k, shape)). The rightmost `.d_` that parses wins, as operator_refs.jl's
/// `_DERIVED_NAME` regex.
/// (D, p, band (k, shape) or `None` for a scalar definition's derivative).
type DerivedName = (String, String, Option<(usize, String)>);

fn derived_name(name: &str) -> Option<DerivedName> {
    let mut end = name.len();
    while let Some(i) = name[..end].rfind(".d_") {
        end = i;
        if i == 0 {
            break;
        }
        let rest = &name[i + 3..];
        let plen = rest.find(['.', '@']).unwrap_or(rest.len());
        if plen == 0 {
            continue;
        }
        let (p, tail) = rest.split_at(plen);
        if tail.is_empty() {
            return Some((name[..i].to_string(), p.to_string(), None));
        }
        // `.b<k>@` (an array definition's band) or `.s<k>@` (a scalar
        // definition's band w.r.t. an array parameter, its sites)
        let Some(t) = tail.strip_prefix(".b").or_else(|| tail.strip_prefix(".s")) else {
            continue;
        };
        let Some(at) = t.find('@') else { continue };
        let digits = &t[..at];
        if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let Ok(k) = digits.parse() else { continue };
        return Some((
            name[..i].to_string(),
            p.to_string(),
            Some((k, t[at + 1..].to_string())),
        ));
    }
    None
}

/// The shape key a derived band name's `<shape>` tag denotes ([`shape_tag`]
/// read back): one part per array parameter of `d`, in parameter order.
fn parse_shape_tag(d: &DefInfo, tag: &str, name: &str) -> R<ShapeKey> {
    let bad = || {
        BandError(format!(
            "derived definition `{name}`: `{tag}` is not a binding shape of `{}`",
            d.name
        ))
    };
    let keyed: Vec<&String> = d
        .params
        .iter()
        .filter(|p| d.structural.contains(*p) || d.array.contains(*p))
        .collect();
    let parts: Vec<&str> = if tag.is_empty() && keyed.is_empty() {
        vec![]
    } else {
        tag.split(',').collect()
    };
    if parts.len() != keyed.len() {
        return Err(bad());
    }
    let mut key = Vec::new();
    for (p, x) in keyed.into_iter().zip(parts) {
        if d.structural.contains(p) {
            key.push((p.clone(), format!("#{x}")));
            continue;
        }
        let dims: Vec<i64> = if x.is_empty() {
            vec![]
        } else {
            x.split('x')
                .map(|y| y.parse().map_err(|_| bad()))
                .collect::<R<_>>()?
        };
        key.push((p.clone(), format!("{dims:?}")));
    }
    Ok(key)
}

/// The binding shape `sk` as a derived band name carries it: per array or
/// structural parameter, in parameter order, its argument's extents joined
/// by `x` or its literal, comma-separated (operator_refs.jl `_shape_tag`). Band k at one shape and
/// band k at another are different definitions.
fn shape_tag(sk: &ShapeKey) -> String {
    sk.iter()
        .map(|(_, v)| {
            if let Some(lit) = v.strip_prefix('#') {
                return lit.to_string();
            }
            v.trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .map(str::trim)
                .filter(|x| !x.is_empty())
                .collect::<Vec<_>>()
                .join("x")
        })
        .collect::<Vec<_>>()
        .join(",")
}

impl RefLayer {
    /// The definition `name`: a registry entry, or a derived definition (one
    /// this layer wrote, or regenerated from its name alone).
    pub(crate) fn def(&self, name: &str) -> R<Rc<DefInfo>> {
        if let Some(d) = self.st.borrow().defs.get(name) {
            return Ok(d.clone());
        }
        if let Some(entry) = self.registry.get(name) {
            let params: Vec<String> = entry
                .get("params")
                .and_then(|p| p.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let body: Expr = serde_json::from_value(
                entry.get("body").cloned().unwrap_or(Value::Null),
            )
            .map_err(|e| {
                BandError(format!(
                    "operator `{name}`: its body has no typed form ({e})"
                ))
            })?;
            let d = self.roles(name, params, body, entry.clone())?;
            self.st
                .borrow_mut()
                .defs
                .insert(name.to_string(), d.clone());
            return Ok(d);
        }
        let (base, p, k) = derived_name(name).ok_or_else(|| {
            BandError(format!(
                "operator reference to `{name}`, which the system does not define"
            ))
        })?;
        let bd = self.def(&base)?;
        if !bd.params.contains(&p) {
            return Err(BandError(format!(
                "operator reference to `{name}`: `{base}` has no parameter `{p}`"
            )));
        }
        match k {
            None => {
                self.dparam(&bd, &p)?;
            }
            Some((_, tag)) => {
                // the binding shape is the name's own: a derived band
                // definition binds only the parameters its coefficient reads
                let sk = parse_shape_tag(&bd, &tag, name)?;
                self.def_bands(&bd, &sk, &p, true)?;
            }
        }
        self.st.borrow().defs.get(name).cloned().ok_or_else(|| {
            BandError(format!(
                "derived definition `{name}` is not produced by differentiating `{base}` at this use"
            ))
        })
    }

    fn roles(&self, name: &str, params: Vec<String>, body: Expr, entry: Value) -> R<Rc<DefInfo>> {
        let ps: HashSet<String> = params.iter().cloned().collect();
        let mut d = DefInfo {
            name: name.to_string(),
            params,
            body: body.clone(),
            entry,
            value: HashSet::new(),
            array: HashSet::new(),
            structural: HashSet::new(),
            rank: 0,
        };
        self.walk_roles(&body, false, &ps, &mut d)?;
        d.rank = self.body_rank(&body)?;
        Ok(Rc::new(d))
    }

    fn walk_roles(&self, e: &Expr, array: bool, ps: &HashSet<String>, d: &mut DefInfo) -> R<()> {
        let n = match e {
            Expr::Variable(v) => {
                if ps.contains(v) {
                    if array {
                        d.array.insert(v.clone())
                    } else {
                        d.value.insert(v.clone())
                    };
                }
                return Ok(());
            }
            Expr::Operator(n) => n,
            _ => return Ok(()),
        };
        for f in [&n.wrt, &n.dim, &n.int_var].into_iter().flatten() {
            if ps.contains(f) {
                d.structural.insert(f.clone());
            }
        }
        for r in n.ranges.iter().flat_map(|r| r.values()) {
            match r {
                RangeSpec::IndexSetRef { from, .. } if ps.contains(from) => {
                    d.structural.insert(from.clone());
                }
                // a parameter as a range bound (`[1, "n"]`)
                RangeSpec::Symbolic(_) => {
                    for x in bound_names(r) {
                        if ps.contains(&x) {
                            d.structural.insert(x);
                        }
                    }
                }
                _ => {}
            }
        }
        if REF_OPS.contains(&n.op.as_str()) {
            let callee = self.def(n.name.as_deref().unwrap_or(""))?;
            let mut bs: Vec<_> = n.bindings.iter().flatten().collect();
            bs.sort_by(|a, b| a.0.cmp(b.0));
            for (q, b) in bs {
                match b {
                    Expr::Variable(v) if ps.contains(v) => {
                        if callee.value.contains(q) {
                            d.value.insert(v.clone());
                        }
                        if callee.array.contains(q) {
                            d.array.insert(v.clone());
                        }
                        if callee.structural.contains(q) {
                            d.structural.insert(v.clone());
                        }
                    }
                    _ => self.walk_roles(b, callee.array.contains(q), ps, d)?,
                }
            }
            return Ok(());
        }
        match n.op.as_str() {
            "index" => {
                if let Some(a) = n.args.first() {
                    self.walk_roles(a, true, ps, d)?;
                }
                for x in n.args.iter().skip(1) {
                    self.walk_roles(x, false, ps, d)?;
                }
            }
            "faq" => {
                for x in &n.args {
                    self.walk_roles(x, true, ps, d)?;
                }
                if let Some(b) = &n.expr {
                    self.walk_roles(b, false, ps, d)?;
                }
                if let Some(f) = &n.filter {
                    self.walk_roles(f, false, ps, d)?;
                }
            }
            "makearray" => {
                for x in n.values.iter().flatten() {
                    self.walk_roles(x, true, ps, d)?;
                }
            }
            _ => {
                let mut err = None;
                n.for_each_child(&mut |c: &Expr| {
                    if err.is_none() {
                        if let Err(e) = self.walk_roles(c, array, ps, d) {
                            err = Some(e);
                        }
                    }
                });
                if let Some(e) = err {
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    fn body_rank(&self, e: &Expr) -> R<usize> {
        let Expr::Operator(n) = e else { return Ok(0) };
        Ok(match n.op.as_str() {
            "faq" => n.output_idx.as_ref().map(|o| o.len()).unwrap_or(0),
            "makearray" => n
                .regions
                .as_ref()
                .and_then(|r| r.first())
                .map(|r| r.len())
                .unwrap_or(0),
            "index" => 0,
            o if REF_OPS.contains(&o) => self.def(n.name.as_deref().unwrap_or(""))?.rank,
            _ => {
                let mut r = 0;
                for c in &n.args {
                    r = r.max(self.body_rank(c)?);
                }
                r
            }
        })
    }

    pub(crate) fn ref_expr(&self, name: &str, params: &[String]) -> Expr {
        let mut e = op(&self.refop, vec![]);
        let n = e.node_mut().expect("an operator node");
        n.name = Some(name.to_string());
        n.bindings = Some(
            params
                .iter()
                .map(|p| (p.clone(), Expr::Variable(p.clone())))
                .collect(),
        );
        e
    }

    fn register_derived(&self, name: &str, params: Vec<String>, body: Expr, what: &str) -> R<()> {
        if self.registry.contains_key(name) {
            return Err(BandError(format!(
                "the derived definition name `{name}` ({what}) is already a definition of the system"
            )));
        }
        if let Some(old) = self.st.borrow().defs.get(name) {
            if old.params == params && skey(&old.body) == skey(&body) {
                return Ok(());
            }
            return Err(BandError(format!(
                "the derived definition `{name}` ({what}) was produced twice with \
                 different bodies"
            )));
        }
        let entry = serde_json::json!({
            "params": params,
            "body": ser_expr(&body),
            "description": what,
            "reference_only": true,
        });
        let d = self.roles(name, params, body, entry.clone())?;
        let mut st = self.st.borrow_mut();
        st.defs.insert(name.to_string(), d);
        st.derived.insert(name.to_string(), entry);
        Ok(())
    }

    /// `∂D/∂p` of a scalar definition, generic in D's parameter names: `0`, a
    /// leaf, a lone reference, or a reference to `D.d_p` (registered once).
    pub(crate) fn dparam(&self, d: &DefInfo, p: &str) -> R<Expr> {
        let key = (d.name.clone(), p.to_string());
        if let Some(t) = self.st.borrow().dparam.get(&key) {
            return Ok(t.clone());
        }
        if d.rank != 0 {
            return Err(BandError(format!(
                "`{}` is an array definition; its derivative is a band list",
                d.name
            )));
        }
        let ps: HashSet<String> = d.params.iter().cloned().collect();
        // A body must be closed (esm-spec §9.6.4 rule 7, as `flatten` closes
        // the merged registry): differentiated only w.r.t. its parameters, a
        // name it read freely would contribute no derivative, a structurally
        // nonzero entry silently dropped (operator_refs.jl closes the
        // registry in `sysview` instead).
        let mut open: Vec<String> = free_vars(&d.body)
            .into_iter()
            .filter(|x| !ps.contains(x))
            .collect();
        if !open.is_empty() {
            open.sort();
            return Err(BandError(format!(
                "operator `{}` reads {} without binding it as a parameter (the body is \
                 not closed)",
                d.name,
                open.iter()
                    .map(|x| format!("`{x}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let site = Site::of(&Expr::Variable(p.to_string()));
        let body = with_defparams(Some(ps), || dscalar_in(&d.body, &site, &SiteScan::plain()))?;
        let body = simp(&body);
        let t = if leafy(&body) {
            body
        } else {
            let name = format!("{}.d_{}", d.name, p);
            let used = used_params(&body, &d.params);
            self.register_derived(&name, used.clone(), body, &format!("∂{}/∂{}", d.name, p))?;
            self.ref_expr(&name, &used)
        };
        self.st.borrow_mut().dparam.insert(key, t.clone());
        Ok(t)
    }
}

fn leafy(e: &Expr) -> bool {
    !matches!(e, Expr::Operator(_)) || is_ref(e)
}

/// The parameters of `params` a derived body reads, in their order.
fn used_params(body: &Expr, params: &[String]) -> Vec<String> {
    fn walk(e: &Expr, used: &mut HashSet<String>) {
        match e {
            Expr::Variable(v) => {
                used.insert(v.clone());
            }
            Expr::Operator(n) => {
                for f in [&n.wrt, &n.dim, &n.int_var].into_iter().flatten() {
                    used.insert(f.clone());
                }
                for r in n.ranges.iter().flat_map(|r| r.values()) {
                    match r {
                        RangeSpec::IndexSetRef { from, .. } => {
                            used.insert(from.clone());
                        }
                        RangeSpec::Symbolic(_) => {
                            used.extend(bound_names(r));
                        }
                        _ => {}
                    }
                }
                n.for_each_child(&mut |c: &Expr| walk(c, used));
            }
            _ => {}
        }
    }
    let mut used = HashSet::new();
    walk(body, &mut used);
    params
        .iter()
        .filter(|p| used.contains(*p))
        .cloned()
        .collect()
}

/// A generic expression over a definition's parameter names at a use
/// binding them to `b`: a name is its argument, a reference has `b`
/// substituted into its bindings; nothing else is substituted.
pub(crate) fn bind(t: &Expr, b: &HashMap<String, Expr>) -> Expr {
    match t {
        Expr::Variable(v) => b.get(v).cloned().unwrap_or_else(|| t.clone()),
        Expr::Operator(n) if REF_OPS.contains(&n.op.as_str()) => {
            let mut n2 = (**n).clone();
            n2.bindings = n.bindings.as_ref().map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), subst_names(v, b)))
                    .collect()
            });
            Expr::operator(n2)
        }
        _ => t.clone(),
    }
}

fn subst_names(e: &Expr, b: &HashMap<String, Expr>) -> Expr {
    if b.is_empty() {
        e.clone()
    } else {
        subst_free(e, b)
    }
}

/// The derivative table's reference row: `Σ_p D.d_p(B) · ∂B[p]/∂s` over the
/// value parameters whose argument holds the site. An array parameter whose
/// argument holds the site is refused (never a silent 0).
pub(crate) fn dref(
    n: &ExpressionNode,
    s: &Site,
    sc: &SiteScan,
) -> Result<Expr, DerivativeRuleError> {
    let l = layer()?;
    let be = |e: BandError| DerivativeRuleError(e.0);
    let d = l.def(n.name.as_deref().unwrap_or("")).map_err(be)?;
    if d.rank != 0 {
        return Err(DerivativeRuleError(format!(
            "an array-valued operator reference (`{}`) in a scalar position",
            d.name
        )));
    }
    let empty = HashMap::new();
    let b = n.bindings.as_ref().unwrap_or(&empty);
    let mut terms = Vec::new();
    for p in &d.params {
        let arg = b.get(p).ok_or_else(|| {
            DerivativeRuleError(format!(
                "operator reference to `{}` does not bind parameter `{p}`",
                d.name
            ))
        })?;
        let da = dscalar_in(arg, s, sc)?;
        if is_zero(&da) {
            continue;
        }
        if d.array.contains(p) {
            return Err(DerivativeRuleError(format!(
                "operator `{}` reads its parameter `{p}` as an array, and the argument bound \
                 to it depends on the differentiation site; differentiating through an array \
                 argument of a scalar definition is not supported",
                d.name
            )));
        }
        let t = l.dparam(&d, p).map_err(be)?;
        if is_zero(&t) {
            continue;
        }
        terms.push(mul(vec![bind(&t, b), da]));
    }
    Ok(add(terms))
}

/// [`dref`] without the products, for the structure-only mode.
pub(crate) fn dref_check(
    n: &ExpressionNode,
    check: &mut dyn FnMut(&Expr) -> Result<bool, DerivativeRuleError>,
) -> Result<bool, DerivativeRuleError> {
    let l = layer()?;
    let be = |e: BandError| DerivativeRuleError(e.0);
    let d = l.def(n.name.as_deref().unwrap_or("")).map_err(be)?;
    let empty = HashMap::new();
    let b = n.bindings.as_ref().unwrap_or(&empty);
    let mut nz = false;
    for p in &d.params {
        let Some(arg) = b.get(p) else { continue };
        if !check(arg)? {
            continue;
        }
        if d.rank != 0 || d.array.contains(p) {
            return Err(DerivativeRuleError(format!(
                "operator `{}`: differentiating through its parameter `{p}` here is not supported",
                d.name
            )));
        }
        nz |= !is_zero(&l.dparam(&d, p).map_err(be)?);
    }
    Ok(nz)
}

// ── array definitions ───────────────────────────────────────────────────────

/// A binding shape: per array parameter its extents (`[6, 4]`), per
/// structural one its literal (`#6`, `#lat`), in parameter order, as text.
type ShapeKey = Vec<(String, String)>;

fn key_text(sk: &ShapeKey) -> String {
    sk.iter()
        .map(|(p, v)| format!("{p}={v}"))
        .collect::<Vec<_>>()
        .join(";")
}

impl RefLayer {
    fn expr_shape(&self, e: &Expr, ctx: &Ctx) -> R<Vec<i64>> {
        match e {
            Expr::Variable(v) => Ok(ctx.shapes.get(v).cloned().unwrap_or_default()),
            Expr::Operator(n) if REF_OPS.contains(&n.op.as_str()) => {
                let d = self.def(n.name.as_deref().unwrap_or(""))?;
                if d.rank == 0 {
                    return Ok(vec![]);
                }
                Ok(self
                    .ref_extents(&d, n.bindings.as_ref().unwrap_or(&HashMap::new()), ctx)?
                    .0)
            }
            Expr::Operator(n) if n.op == "faq" || n.op == "makearray" => {
                Ok(body_extents(n, ctx)?.0)
            }
            Expr::Operator(n) if n.op == "const" => {
                let v = serde_json::to_value(n.value.as_ref()).unwrap_or(Value::Null);
                let mut dims = vec![];
                let mut x = &v;
                while let Value::Array(a) = x {
                    dims.push(a.len() as i64);
                    match a.first() {
                        Some(y) => x = y,
                        None => break,
                    }
                }
                Ok(dims)
            }
            Expr::Operator(n) if n.op == "index" => Ok(vec![]),
            Expr::Operator(n) => {
                for c in &n.args {
                    let s = self.expr_shape(c, ctx)?;
                    if !s.is_empty() {
                        return Ok(s);
                    }
                }
                Ok(vec![])
            }
            _ => Ok(vec![]),
        }
    }

    fn shapekey(&self, d: &DefInfo, b: &HashMap<String, Expr>, ctx: &Ctx) -> R<ShapeKey> {
        let mut key = Vec::new();
        for p in &d.params {
            if d.structural.contains(p) {
                let a = b
                    .get(p)
                    .ok_or_else(|| BandError(format!("`{}` does not bind `{p}`", d.name)))?;
                let lit = structural_literal(a).ok_or_else(|| {
                    BandError(format!(
                        "operator `{}`: parameter `{p}` fills a structural field and must be \
                         bound to an integer or a name, not {}",
                        d.name,
                        skey(a)
                    ))
                })?;
                key.push((p.clone(), format!("#{lit}")));
            } else if d.array.contains(p) {
                let a = b
                    .get(p)
                    .ok_or_else(|| BandError(format!("`{}` does not bind `{p}`", d.name)))?;
                key.push((p.clone(), format!("{:?}", self.expr_shape(a, ctx)?)));
            }
        }
        Ok(key)
    }

    fn def_ctx(&self, sk: &ShapeKey) -> Ctx {
        let mut shapes = HashMap::new();
        for (p, v) in sk.iter().filter(|(_, v)| !v.starts_with('#')) {
            let dims: Vec<i64> = v
                .trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect();
            if !dims.is_empty() {
                shapes.insert(p.clone(), dims);
            }
        }
        Ctx {
            index_sets: self.index_sets.clone(),
            shapes,
            ..Default::default()
        }
    }

    pub(crate) fn ref_extents(
        &self,
        d: &DefInfo,
        b: &HashMap<String, Expr>,
        ctx: &Ctx,
    ) -> R<(Vec<i64>, Vec<i64>)> {
        let sk = self.shapekey(d, b, ctx)?;
        let key = format!("{}\0{}", d.name, key_text(&sk));
        if let Some(x) = self.st.borrow().extents.get(&key) {
            return Ok(x.clone());
        }
        let body = specialized(d, &sk)?;
        let Expr::Operator(n) = &body else {
            return Err(BandError(format!(
                "`{}`: an array definition's body is a leaf",
                d.name
            )));
        };
        let x = body_extents(n, &self.def_ctx(&sk))?;
        self.st.borrow_mut().extents.insert(key, x.clone());
        Ok(x)
    }

    /// The bands of array definition `d`'s body w.r.t. its parameter `p` at
    /// binding shape `sk`, generic in d's parameter names (memoized).
    fn def_bands(&self, d: &DefInfo, sk: &ShapeKey, p: &str, co: bool) -> R<Rc<Vec<Band>>> {
        let key = format!("{}\0{}\0{}\0{}", d.name, key_text(sk), p, co);
        if let Some(x) = self.st.borrow().defbands.get(&key) {
            return Ok(x.clone());
        }
        // a scalar definition has bands only w.r.t. an array parameter (its
        // sites, §3.6); a value parameter's derivative is `D.d_p`
        if d.rank == 0 && !d.array.contains(p) {
            return Err(BandError(format!("`{}` is a scalar definition", d.name)));
        }
        let mut ctx = self.def_ctx(sk);
        let me = current_layer();
        ctx.refs = me.clone();
        let body = specialized(d, sk)?;
        let Expr::Operator(n) = &body else {
            return Err(BandError(format!(
                "`{}`: an array definition's body is a leaf",
                d.name
            )));
        };
        let ext = if d.rank == 0 {
            Vec::new()
        } else {
            body_extents(n, &ctx)?.0
        };
        let ps: HashSet<String> = d.params.iter().cloned().collect();
        let raw = with_defparams(Some(ps), || {
            if d.rank > 0 {
                return ref_bands(&body, p, &ctx, &ext, co);
            }
            // a scalar body's sites: its value parameters index like names
            crate::reductions::with_index_params(Some(d.value.clone()), || {
                ref_bands(&body, p, &ctx, &ext, co)
            })
        })?;
        let bs: Vec<Band> = merge_bands(raw).iter().map(normalize_band).collect();
        let mut out = Vec::new();
        for (k, mut b) in bs.into_iter().enumerate() {
            if co {
                b.coef = self.generic_coef(d, sk, p, k + 1, &b)?;
            }
            out.push(b);
        }
        let out = Rc::new(out);
        self.st.borrow_mut().defbands.insert(key, out.clone());
        Ok(out)
    }

    fn generic_coef(&self, d: &DefInfo, sk: &ShapeKey, p: &str, k: usize, b: &Band) -> R<Expr> {
        let c = &b.coef;
        let mut names: HashSet<String> = b
            .ridx
            .iter()
            .filter_map(|r| match r {
                RIdx::Name(n) => Some(n.clone()),
                _ => None,
            })
            .collect();
        names.extend(b.contracted.iter().map(|(n, _, _)| n.clone()));
        let fv = free_vars(c);
        for n in &fv {
            if !names.contains(n) && !d.params.contains(n) {
                return Err(BandError(format!(
                    "a band coefficient of `{}` reads `{n}`, which is neither a parameter nor \
                     an index of the band",
                    d.name
                )));
            }
        }
        if leafy(c) {
            return Ok(c.clone());
        }
        let mut extra: Vec<String> = fv.into_iter().filter(|n| names.contains(n)).collect();
        extra.sort();
        let letter = if d.rank == 0 { "s" } else { "b" };
        let name = format!("{}.d_{}.{letter}{}@{}", d.name, p, k, shape_tag(sk));
        let mut params = used_params(c, &d.params);
        params.extend(extra);
        self.register_derived(
            &name,
            params.clone(),
            c.clone(),
            &format!("band {k} of ∂{}/∂{p}", d.name),
        )?;
        Ok(self.ref_expr(&name, &params))
    }
}

/// The names among a symbolic range's bounds (a parameter as a bound,
/// `[1, "n"]`); its integer bounds are not names.
fn bound_names(r: &RangeSpec) -> Vec<String> {
    match serde_json::to_value(r) {
        Ok(Value::Array(bounds)) => bounds
            .into_iter()
            .filter_map(|b| match b {
                Value::String(s) => Some(s),
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

/// The literal a structural parameter is bound to: an integer (an integral
/// number) or a name (an index-set name, an axis name).
fn structural_literal(e: &Expr) -> Option<String> {
    match e {
        Expr::Integer(i) => Some(i.to_string()),
        Expr::Number(x) if x.fract() == 0.0 && x.abs() < 9.0e15 => Some((*x as i64).to_string()),
        Expr::Variable(v) => Some(v.clone()),
        _ => None,
    }
}

/// The body of `d` at shape key `sk`: its structural parameters replaced by
/// their literals everywhere they occur (a variable leaf, a scalar field, a
/// range bound or `from`, a region bound), so the analysis reads its ranges
/// and regions as numbers (operator_refs.jl `_specialized`). The definition
/// itself is unchanged. The substitution is made on the wire form, where
/// each of those is a string equal to the parameter's name; a closed body
/// binds no loop symbol of a parameter's name.
fn specialized(d: &DefInfo, sk: &ShapeKey) -> R<Expr> {
    let lits: HashMap<&str, Value> = sk
        .iter()
        .filter_map(|(p, v)| {
            let lit = v.strip_prefix('#')?;
            Some((
                p.as_str(),
                lit.parse::<i64>()
                    .map(Value::from)
                    .unwrap_or_else(|_| Value::from(lit)),
            ))
        })
        .collect();
    if lits.is_empty() {
        return Ok(d.body.clone());
    }
    fn walk(x: &mut Value, lits: &HashMap<&str, Value>) {
        match x {
            Value::String(s) => {
                if let Some(v) = lits.get(s.as_str()) {
                    *x = v.clone();
                }
            }
            Value::Array(a) => a.iter_mut().for_each(|y| walk(y, lits)),
            Value::Object(m) => {
                for (k, y) in m.iter_mut() {
                    // an op, a referenced definition's name, a binding's
                    // parameter (a key) and a loop symbol are never a read
                    if !matches!(k.as_str(), "op" | "name" | "output_idx") {
                        walk(y, lits);
                    }
                }
            }
            _ => {}
        }
    }
    let mut v = serde_json::to_value(&d.body)
        .map_err(|e| BandError(format!("`{}`: its body does not serialize: {e}", d.name)))?;
    walk(&mut v, &lits);
    serde_json::from_value(v).map_err(|e| {
        BandError(format!(
            "`{}`: its body specialized to its structural literals has no typed form: {e}",
            d.name
        ))
    })
}

/// The extents and origins of an array body: a faq's output ranges, a
/// makearray's region positions (origin 1).
fn body_extents(n: &ExpressionNode, ctx: &Ctx) -> R<(Vec<i64>, Vec<i64>)> {
    match n.op.as_str() {
        "faq" => {
            let empty = HashMap::new();
            let ranges = n.ranges.as_ref().unwrap_or(&empty);
            let mut ext = vec![];
            let mut org = vec![];
            for x in n.output_idx.iter().flatten() {
                let r = ranges.get(x).ok_or_else(|| {
                    BandError(format!(
                        "an array definition's output index `{x}` has no range"
                    ))
                })?;
                let (lo, hi) = range_of(r, ctx)?;
                ext.push(hi - lo + 1);
                org.push(lo);
            }
            Ok((ext, org))
        }
        "makearray" => {
            let regions = n
                .regions
                .as_ref()
                .ok_or_else(|| BandError("makearray without regions".into()))?;
            let k = regions.first().map(|r| r.len()).unwrap_or(0);
            let mut hi = vec![0; k];
            for reg in regions {
                for (j, r) in reg.iter().enumerate() {
                    hi[j] = hi[j].max(region_bounds(r)?.1);
                }
            }
            Ok((hi, vec![1; k]))
        }
        _ if ref_node(&Expr::operator(n.clone())).is_some() && ctx.refs.is_some() => {
            // a body that is itself a reference to an array definition: that
            // definition's extents at the reference's bindings
            let l = ctx.refs.clone().unwrap();
            let d2 = l.def(n.name.as_deref().unwrap_or(""))?;
            if d2.rank == 0 {
                return Err(BandError(
                    "an array definition's body is a scalar reference".into(),
                ));
            }
            let empty = HashMap::new();
            l.ref_extents(&d2, n.bindings.as_ref().unwrap_or(&empty), ctx)
        }
        o => Err(BandError(format!(
            "an array definition's body is a `{o}`, not a faq or makearray"
        ))),
    }
}

// ── uses of array definitions ──────────────────────────────────────────────

pub(crate) struct Placeholder {
    pub(crate) name: String,
    pub(crate) r: Expr,
    ext: Vec<i64>,
    org: Vec<i64>,
    /// the scope dimensions' index names (operator_refs.jl `_Pseudo`)
    scope: Vec<String>,
}

impl Placeholder {
    /// `ctx` with this placeholder's extents and origins declared.
    pub(crate) fn declare(&self, ctx: &mut Ctx) {
        ctx.shapes.insert(self.name.clone(), self.ext.clone());
        ctx.origins.insert(self.name.clone(), self.org.clone());
    }

    /// A band of the reference's value with the scope dimensions put first:
    /// rows over their ranges, named by the scope indices.
    pub(crate) fn with_scope(&self, mut b: Band) -> Band {
        if self.scope.is_empty() {
            return b;
        }
        let m = self.scope.len();
        let mut rows: Vec<(i64, i64)> = (0..m)
            .map(|d| (self.org[d], self.org[d] + self.ext[d] - 1))
            .collect();
        rows.extend(b.rows);
        let mut ridx: Vec<RIdx> = self.scope.iter().map(|n| RIdx::Name(n.clone())).collect();
        ridx.extend(b.ridx);
        b.rows = rows;
        b.ridx = ridx;
        b
    }
}

/// A band whose column reads a value at the use (a scalar definition's
/// parameter bound to data) made one-hot.
pub(crate) fn one_hot_band(mut b: Band, v: &str, ctx: &Ctx) -> R<Band> {
    let (cidx, coef, contracted) = crate::reductions::one_hot(
        std::mem::take(&mut b.cidx),
        b.coef.clone(),
        std::mem::take(&mut b.contracted),
        &b.ridx,
        v,
        ctx,
    )?;
    b.cidx = cidx;
    b.coef = coef;
    b.contracted = contracted;
    Ok(b)
}

/// `b` with every placeholder read back as its reference: a read
/// `index(placeholder, a…, i…)` is the reference with its scope indices
/// bound to `a…`, read at `i…` (itself without own dimensions).
pub(crate) fn restore_band(mut b: Band, ps: &[Placeholder]) -> R<Band> {
    let m: HashMap<&str, &Placeholder> = ps.iter().map(|p| (p.name.as_str(), p)).collect();
    b.cidx = b
        .cidx
        .iter()
        .map(|c| restore_expr(c, &m))
        .collect::<R<_>>()?;
    b.coef = restore_expr(&b.coef, &m)?;
    b.guard = match &b.guard {
        Some(g) => Some(restore_expr(g, &m)?),
        None => None,
    };
    Ok(b)
}

fn restore_expr(e: &Expr, ps: &HashMap<&str, &Placeholder>) -> R<Expr> {
    match e {
        Expr::Variable(v) => match ps.get(v.as_str()) {
            None => Ok(e.clone()),
            Some(p) if p.scope.is_empty() => Ok(p.r.clone()),
            Some(_) => Err(BandError(
                "a placeholder with scope dimensions read whole".into(),
            )),
        },
        Expr::Operator(n) => {
            if n.op == "index" {
                if let Some(Expr::Variable(v)) = n.args.first() {
                    if let Some(p) = ps.get(v.as_str()) {
                        let a: Vec<Expr> = n.args[1..]
                            .iter()
                            .map(|x| restore_expr(x, ps))
                            .collect::<R<_>>()?;
                        let m = p.scope.len();
                        let sub: HashMap<String, Expr> =
                            (0..m).map(|d| (p.scope[d].clone(), a[d].clone())).collect();
                        let r = if sub.is_empty() {
                            p.r.clone()
                        } else {
                            bind(&p.r, &sub)
                        };
                        let own = a[m..].to_vec();
                        return Ok(if own.is_empty() {
                            r
                        } else {
                            let mut args = vec![r];
                            args.extend(own);
                            op("index", args)
                        });
                    }
                }
            }
            let mut err = None;
            let out = Expr::operator(n.map_children(&mut |c: &Expr| match restore_expr(c, ps) {
                Ok(x) => x,
                Err(er) => {
                    err.get_or_insert(er);
                    c.clone()
                }
            }));
            match err {
                Some(e) => Err(e),
                None => Ok(out),
            }
        }
        _ => Ok(e.clone()),
    }
}

type Scope = Vec<(String, Option<RangeSpec>)>;
type Found = IndexMap<String, (Expr, String, Scope)>;

/// `e` with every reference read as an intermediate replaced by a
/// placeholder (operator_refs.jl `_abstract_refs`): array-valued ones, and a
/// scalar one that binds an array parameter to an argument reading `v`. A
/// placeholder's cells are its scope dimensions (the indices bound around
/// it that its bindings read, in the order they are bound), then its own.
pub(crate) fn abstract_refs(
    l: &RefLayer,
    e: &Expr,
    ctx: &Ctx,
    v: Option<&str>,
) -> R<(Expr, Vec<Placeholder>)> {
    let mut found: Found = IndexMap::new();
    let out = abstract_walk(l, e, &mut found, &Vec::new(), v)?;
    let mut ps = Vec::new();
    for (_, (r, name, sc)) in found {
        let n = ref_node(&r).unwrap();
        let d = l.def(n.name.as_deref().unwrap_or(""))?;
        let (ext, org) = if d.rank > 0 {
            l.ref_extents(&d, n.bindings.as_ref().unwrap_or(&HashMap::new()), ctx)?
        } else {
            (Vec::new(), Vec::new())
        };
        let (mut sext, mut sorg) = (Vec::new(), Vec::new());
        for (nm, spec) in &sc {
            let (lo, hi) = scope_range(spec.as_ref(), nm, &d.name, ctx)?;
            sext.push(hi - lo + 1);
            sorg.push(lo);
        }
        sext.extend(ext);
        sorg.extend(org);
        let scope = sc.into_iter().map(|(n, _)| n).collect();
        ps.push(Placeholder {
            name,
            r,
            ext: sext,
            org: sorg,
            scope,
        });
    }
    Ok((out, ps))
}

/// The static `[lo, hi]` of a scope index's range.
fn scope_range(spec: Option<&RangeSpec>, nm: &str, dname: &str, ctx: &Ctx) -> R<(i64, i64)> {
    match spec {
        Some(r @ (RangeSpec::Interval(_) | RangeSpec::IndexSetRef { .. })) => {
            crate::bands::range_of(r, ctx)
        }
        _ => Err(BandError(format!(
            "operator reference to `{dname}` binds an argument that reads the index `{nm}` \
             bound around it, whose range is not a static interval or index set; not supported"
        ))),
    }
}

fn reads_through_array(d: &DefInfo, b: &HashMap<String, Expr>, v: Option<&str>) -> bool {
    let Some(v) = v else { return false };
    d.params
        .iter()
        .any(|p| d.array.contains(p) && b.get(p).is_some_and(|x| free_vars(x).contains(v)))
}

fn placeholder_of(r: &Expr, found: &mut Found, scope: &Scope) -> (String, Vec<String>) {
    let fv = free_vars(r);
    let sc: Scope = scope
        .iter()
        .filter(|(n, _)| fv.contains(n))
        .cloned()
        .collect();
    let key = format!(
        "{}\0{}",
        skey(r),
        sc.iter()
            .map(|(n, s)| format!(
                "{n}={}",
                s.as_ref().map(|x| format!("{x:?}")).unwrap_or_default()
            ))
            .collect::<Vec<_>>()
            .join(",")
    );
    let len = found.len();
    let ent = found
        .entry(key)
        .or_insert_with(|| (r.clone(), format!("§ref{}", len + 1), sc));
    (
        ent.1.clone(),
        ent.2.iter().map(|(n, _)| n.clone()).collect(),
    )
}

fn abstract_walk(
    l: &RefLayer,
    e: &Expr,
    found: &mut Found,
    scope: &Scope,
    v: Option<&str>,
) -> R<Expr> {
    let Expr::Operator(n) = e else {
        return Ok(e.clone());
    };
    // `index(ref, i…)` of an array-valued reference: its own indices follow
    // its scope dimensions
    if n.op == "index" {
        if let Some(r @ Expr::Operator(rn)) = n.args.first() {
            if REF_OPS.contains(&rn.op.as_str())
                && l.def(rn.name.as_deref().unwrap_or(""))?.rank > 0
            {
                let own: Vec<Expr> = n.args[1..]
                    .iter()
                    .map(|x| abstract_walk(l, x, found, scope, v))
                    .collect::<R<_>>()?;
                let (name, sc) = placeholder_of(r, found, scope);
                let mut args = vec![Expr::Variable(name)];
                args.extend(sc.into_iter().map(Expr::Variable));
                args.extend(own);
                return Ok(op("index", args));
            }
        }
    }
    if REF_OPS.contains(&n.op.as_str()) {
        let d = l.def(n.name.as_deref().unwrap_or(""))?;
        let empty = HashMap::new();
        if d.rank > 0 || reads_through_array(&d, n.bindings.as_ref().unwrap_or(&empty), v) {
            let (name, sc) = placeholder_of(e, found, scope);
            if sc.is_empty() {
                return Ok(Expr::Variable(name));
            }
            if d.rank > 0 {
                return Err(BandError(format!(
                    "the array-valued operator reference `{}` binds an argument that reads an \
                     index bound around it ({}) and is read whole, not through `index`; not \
                     supported",
                    d.name,
                    sc.join(", ")
                )));
            }
            let mut args = vec![Expr::Variable(name)];
            args.extend(sc.into_iter().map(Expr::Variable));
            return Ok(op("index", args));
        }
    }
    let mut scope2 = scope.clone();
    if let (true, Some(rs)) = (n.op == "faq", n.ranges.as_ref()) {
        for bname in binders(n) {
            scope2.retain(|(x, _)| x != &bname);
            scope2.push((bname.clone(), rs.get(&bname).cloned()));
        }
    }
    let mut err = None;
    let out = Expr::operator(n.map_children(&mut |c: &Expr| match abstract_walk(
        l, c, found, &scope2, v,
    ) {
        Ok(x) => x,
        Err(er) => {
            err.get_or_insert(er);
            c.clone()
        }
    }));
    match err {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// The bands of an expression that may hold operator references: scalar
/// references through the derivative table's reference row, every array
/// reference read as an intermediate whose own bands are composed with the
/// expression's bands with respect to it. Not merged.
pub(crate) fn ref_bands(rhs: &Expr, v: &str, ctx: &Ctx, shape_u: &[i64], co: bool) -> R<Vec<Band>> {
    let Some(l) = ctx.refs.clone() else {
        return bands_in(rhs, v, ctx, shape_u, co);
    };
    with_layer(&Some(l.clone()), || {
        let (rhs2, ps) = abstract_refs(&l, rhs, ctx, Some(v))?;
        if ps.is_empty() {
            return bands_in(rhs, v, ctx, shape_u, co);
        }
        let mut ctx2 = ctx.clone();
        for p in &ps {
            p.declare(&mut ctx2);
        }
        let fv = earthsci_ast::free_variables(&rhs2);
        let mut out = Vec::new();
        if fv.contains(v) {
            out.extend(bands_in(&rhs2, v, &ctx2, shape_u, co)?);
        }
        for p in &ps {
            if !fv.contains(&p.name) || !earthsci_ast::free_variables(&p.r).contains(v) {
                continue;
            }
            let inner = ref_value_bands(&l, &p.r, v, ctx, co)?;
            if inner.is_empty() {
                continue;
            }
            let inner: Vec<Band> = inner
                .into_iter()
                .map(|b| one_hot_band(p.with_scope(b), v, ctx))
                .collect::<R<_>>()?;
            let outer = bands_in(&rhs2, &p.name, &ctx2, shape_u, co)?;
            out.extend(compose(&outer, &inner, co)?);
        }
        out.into_iter().map(|b| restore_band(b, &ps)).collect()
    })
}

pub(crate) fn ref_value_bands(
    l: &RefLayer,
    r: &Expr,
    v: &str,
    ctx: &Ctx,
    co: bool,
) -> R<Vec<Band>> {
    let n = ref_node(r).unwrap();
    let d = l.def(n.name.as_deref().unwrap_or(""))?;
    let empty = HashMap::new();
    let b = n.bindings.as_ref().unwrap_or(&empty);
    let sk = l.shapekey(&d, b, ctx)?;
    let mut out = Vec::new();
    for p in &d.params {
        let arg = b
            .get(p)
            .ok_or_else(|| BandError(format!("`{}` does not bind `{p}`", d.name)))?;
        if !earthsci_ast::free_variables(arg).contains(v) {
            continue;
        }
        let db = if d.rank == 0 && !d.array.contains(p) {
            // a value parameter of a scalar definition: `D.d_p`, one band
            let t = if co { l.dparam(&d, p)? } else { placeholder() };
            if co && is_zero(&t) {
                continue;
            }
            Rc::new(vec![Band {
                rows: vec![],
                ridx: vec![],
                cidx: vec![],
                coef: t,
                contracted: vec![],
                guard: None,
                join: None,
            }])
        } else {
            l.def_bands(&d, &sk, p, co)?
        };
        if db.is_empty() {
            continue;
        }
        let shape_p = if d.array.contains(p) {
            l.expr_shape(arg, ctx)?
        } else {
            vec![]
        };
        let inner = with_defparams(None, || ref_bands(arg, v, ctx, &shape_p, co))?;
        if inner.is_empty() {
            continue;
        }
        let bound: Vec<Band> = db
            .iter()
            .map(|x| bind_band(x, b, &d.params, &d.name))
            .collect::<R<_>>()?;
        out.extend(compose(&bound, &inner, co)?);
    }
    Ok(out)
}

fn bind_band(b: &Band, args: &HashMap<String, Expr>, params: &[String], dname: &str) -> R<Band> {
    let mut argnames = HashSet::new();
    for x in args.values() {
        argnames.extend(free_vars(x));
    }
    let mut own: Vec<String> = b
        .ridx
        .iter()
        .filter_map(|r| match r {
            RIdx::Name(n) => Some(n.clone()),
            _ => None,
        })
        .collect();
    own.extend(b.contracted.iter().map(|(n, _, _)| n.clone()));
    let mut taken: HashSet<String> = argnames.clone();
    taken.extend(own.iter().cloned());
    taken.extend(params.iter().cloned());
    let mut ren: HashMap<String, Expr> = HashMap::new();
    for n in &own {
        if argnames.contains(n) {
            let f = fresh(&mut taken, &format!("{n}_r"), "_r");
            ren.insert(n.clone(), Expr::Variable(f));
        }
    }
    let rn = |n: &String| match ren.get(n) {
        Some(Expr::Variable(f)) => f.clone(),
        _ => n.clone(),
    };
    let mut b = b.clone();
    if !ren.is_empty() {
        b.ridx = b
            .ridx
            .iter()
            .map(|r| match r {
                RIdx::Name(n) => RIdx::Name(rn(n)),
                f => f.clone(),
            })
            .collect();
        b.cidx = b.cidx.iter().map(|c| subst_names(c, &ren)).collect();
        b.coef = bind(&b.coef, &ren);
        b.contracted = b
            .contracted
            .iter()
            .map(|(n, lo, hi)| (rn(n), *lo, *hi))
            .collect();
        b.guard = b.guard.as_ref().map(|g| subst_names(g, &ren));
        let rm: HashMap<String, String> = ren
            .iter()
            .filter_map(|(k, v)| match v {
                Expr::Variable(f) => Some((k.clone(), f.clone())),
                _ => None,
            })
            .collect();
        b.join = b.join.as_ref().map(|j| j.renamed(&rm));
    }
    b.cidx = b.cidx.iter().map(|c| subst_names(c, args)).collect();
    b.coef = bind(&b.coef, args);
    b.guard = b.guard.as_ref().map(|g| subst_names(g, args));
    // a join clause names the definition's parameters (envelope factors, key
    // columns) as strings: each is the variable its argument binds (§3.5)
    if let Some(j) = &b.join {
        let text = crate::helpers::stable_json(&serde_json::Value::Array(j.clauses.clone()));
        let mut jm = HashMap::new();
        for (p, x) in args {
            match x {
                Expr::Variable(v) => {
                    jm.insert(p.clone(), v.clone());
                }
                _ if text.contains(&format!("\"{p}\"")) => {
                    return Err(BandError(format!(
                        "a join of `{dname}` names parameter `{p}`, bound to an expression: a join reads variables only"
                    )))
                }
                _ => {}
            }
        }
        b.join = Some(j.renamed(&jm));
    }
    Ok(b)
}

fn int_lit(e: &Expr) -> Option<i64> {
    match e {
        Expr::Integer(i) => Some(*i),
        Expr::Number(f) if f.fract() == 0.0 => Some(*f as i64),
        _ => None,
    }
}

/// The chain rule through an intermediate array `W`: `outer` are bands
/// w.r.t. `W` (columns: cells of `W`), `inner` the bands of `W` (rows: cells
/// of `W`); see operator_refs.jl `_compose`.
/// Whether `e` is an `ifelse` with a literal-zero branch: a guard.
fn is_guard(e: &Expr) -> bool {
    matches!(e, Expr::Operator(n) if n.op == "ifelse" && n.args.len() == 3
        && (is_zero(&n.args[1]) || is_zero(&n.args[2])))
}

/// `outer · inner`, with a guard of `outer` (an `ifelse` with a literal-zero
/// branch, `outer` itself or the first such factor of a product) kept
/// outside the product: `ifelse(c, a, 0) · y` is `ifelse(c, a · y, 0)`. The
/// guard of a definition (`ifelse(s > 0, …/s, …)`) then also guards its
/// composed derivative where the inner derivative `y` is NaN (spec §3.6).
pub(crate) fn guarded_mul(outer: &Expr, inner: Expr) -> Expr {
    if let Expr::Operator(n) = outer {
        if is_guard(outer) {
            return op(
                "ifelse",
                vec![
                    n.args[0].clone(),
                    guarded_mul(&n.args[1], inner.clone()),
                    guarded_mul(&n.args[2], inner),
                ],
            );
        }
        if n.op == "*" {
            if let Some(k) = n.args.iter().position(is_guard) {
                let Expr::Operator(g) = &n.args[k] else {
                    unreachable!("a guard is an operator")
                };
                let branch = |x: &Expr| {
                    let mut f: Vec<Expr> = n.args[..k].to_vec();
                    f.push(x.clone());
                    f.extend(n.args[k + 1..].iter().cloned());
                    guarded_mul(&mul(f), inner.clone())
                };
                return op(
                    "ifelse",
                    vec![g.args[0].clone(), branch(&g.args[1]), branch(&g.args[2])],
                );
            }
        }
    }
    mul(vec![outer.clone(), inner])
}

pub(crate) fn compose(outer: &[Band], inner: &[Band], co: bool) -> R<Vec<Band>> {
    let mut out = Vec::new();
    for o in outer {
        for b in inner {
            if o.cidx.len() != b.ridx.len() {
                return Err(BandError(format!(
                    "composing through an intermediate array: a read of rank {} of an array of rank {}",
                    o.cidx.len(),
                    b.ridx.len()
                )));
            }
            let mut taken: HashSet<String> = HashSet::new();
            for r in &o.ridx {
                if let RIdx::Name(n) = r {
                    taken.insert(n.clone());
                }
            }
            taken.extend(o.contracted.iter().map(|(n, _, _)| n.clone()));
            for c in &o.cidx {
                taken.extend(free_vars(c));
            }
            all_names(&o.coef, &mut taken);
            let mut sigma: HashMap<String, Expr> = HashMap::new();
            let mut bcon = Vec::new();
            for (n, lo, hi) in &b.contracted {
                let f = if taken.contains(n) {
                    fresh(&mut taken, &format!("{n}_c"), "_c")
                } else {
                    taken.insert(n.clone());
                    n.clone()
                };
                if f != *n {
                    sigma.insert(n.clone(), Expr::Variable(f.clone()));
                }
                bcon.push((f, *lo, *hi));
            }
            let mut conds = Vec::new();
            let mut dead = false;
            let jnames = crate::bands::join_names(b);
            let mut tie = Vec::new();
            for (d, r) in b.ridx.iter().enumerate() {
                let (lo, hi) = b.rows[d];
                if let RIdx::Name(n) = r {
                    if jnames.contains(n) && !matches!(o.cidx[d], Expr::Variable(_)) {
                        // a joined row name read at a non-bare index stays a
                        // name of the composite (the join reads it):
                        // contracted over its row range, tied to the index by
                        // an equality (§3.5)
                        let f = if taken.contains(n) {
                            fresh(&mut taken, &format!("{n}_c"), "_c")
                        } else {
                            taken.insert(n.clone());
                            n.clone()
                        };
                        sigma.insert(n.clone(), Expr::Variable(f.clone()));
                        bcon.push((f.clone(), lo, hi));
                        tie.push(op("==", vec![Expr::Variable(f), o.cidx[d].clone()]));
                    } else {
                        sigma.insert(n.clone(), o.cidx[d].clone());
                    }
                }
                for (cmp, bound) in [(">=", lo), ("<=", hi)] {
                    if let Some(x) = int_lit(&o.cidx[d]) {
                        let ok = if cmp == ">=" { x >= bound } else { x <= bound };
                        if !ok {
                            dead = true;
                        }
                        continue;
                    }
                    conds.push(op(cmp, vec![o.cidx[d].clone(), Expr::Integer(bound)]));
                }
                if dead {
                    break;
                }
            }
            if dead {
                continue;
            }
            let sub = |x: &Expr| subst_names(x, &sigma);
            // a join travels with the band it gates (§3.5): the inner's names
            // renamed as the composite reads them; two joins are refused
            let mut jn = o.join.clone();
            if let Some(bj) = &b.join {
                if o.join.is_some() {
                    return Err(BandError(
                        "a band through two joined faq nodes is not supported".into(),
                    ));
                }
                let mut jm = HashMap::new();
                for n in bj.names() {
                    match sigma.get(&n) {
                        None => {}
                        Some(Expr::Variable(x)) => {
                            jm.insert(n.clone(), x.clone());
                        }
                        Some(x) => {
                            return Err(BandError(format!(
                                "a joined band composed at a non-bare index ({}) is not supported",
                                serde_json::to_string(&ser_expr(x)).unwrap_or_default()
                            )))
                        }
                    }
                }
                jn = Some(bj.renamed(&jm));
            }
            let mut g = conj(o.guard.clone(), b.guard.as_ref().map(sub));
            if !conds.is_empty() {
                g = conj(g, conj_list(conds));
            }
            if !tie.is_empty() {
                g = conj(g, conj_list(tie));
            }
            let cidx: Vec<Expr> = b
                .cidx
                .iter()
                .map(|c| canon_lits(&simplify_jl(&sub(c))))
                .collect();
            let coef = if co {
                // The inner coefficient is any expression (an argument's
                // band, or a composed band `ref * argcoef`), not only a
                // generic leaf: substitute everywhere, bindings included.
                simp(&guarded_mul(&o.coef, sub(&b.coef)))
            } else {
                placeholder()
            };
            if co && is_zero(&coef) {
                continue;
            }
            let mut contracted = o.contracted.clone();
            contracted.extend(bcon);
            let mut names: Vec<String> = o
                .ridx
                .iter()
                .filter_map(|r| match r {
                    RIdx::Name(n) => Some(n.clone()),
                    _ => None,
                })
                .collect();
            let nr = names.len();
            names.extend(contracted.iter().map(|(n, _, _)| n.clone()));
            let mut bx: Vec<(i64, i64)> = o
                .ridx
                .iter()
                .zip(&o.rows)
                .filter(|(r, _)| matches!(r, RIdx::Name(_)))
                .map(|(_, x)| *x)
                .collect();
            bx.extend(contracted.iter().map(|(_, lo, hi)| (*lo, *hi)));
            let pieces = match &g {
                None => vec![(bx.clone(), None)],
                Some(g) => split_guard(g, &names, &bx),
            };
            for (cell, g2) in pieces {
                let mut rows = o.rows.clone();
                let mut j = 0;
                for (k, r) in o.ridx.iter().enumerate() {
                    if matches!(r, RIdx::Name(_)) {
                        rows[k] = cell[j];
                        j += 1;
                    }
                }
                let con: Vec<(String, i64, i64)> = contracted
                    .iter()
                    .enumerate()
                    .map(|(i, (n, _, _))| (n.clone(), cell[nr + i].0, cell[nr + i].1))
                    .collect();
                out.push(normalize_band(&Band {
                    rows,
                    ridx: o.ridx.clone(),
                    cidx: cidx.clone(),
                    coef: coef.clone(),
                    contracted: con,
                    guard: g2,
                    join: jn.clone(),
                }));
            }
        }
    }
    Ok(out)
}

/// Whether `e` is an array-valued reference (for `is_array_expr`).
pub(crate) fn is_array_ref(e: &Expr) -> bool {
    let (Some(n), Some(l)) = (ref_node(e), current_layer()) else {
        return false;
    };
    l.def(n.name.as_deref().unwrap_or(""))
        .map(|d| d.rank > 0)
        .unwrap_or(false)
}

// ── closing a registry ──────────────────────────────────────────────────────
//
// A model's own `operators` bodies may read names of the model freely: the
// reference's site supplies them (esm-spec §9.6.4 rules 1–2). `flatten`
// closes them (rule 7): every body's free names (its own, and those of the
// definitions it references) become trailing parameters, sorted, and every
// reference binds each to its own name. The view of a single model is closed
// the same way (operator_refs.jl `_close_registry`), so every analysis meets
// closed bodies whatever the input. A flattened registry is closed already
// (a no-op).

/// A definition name → the free names its references must bind.
pub(crate) type Closing = HashMap<String, Vec<String>>;

/// (closed registry, [`Closing`]; empty when nothing reads a name freely).
pub(crate) fn close_registry(
    raw: &IndexMap<String, Value>,
) -> R<(IndexMap<String, Value>, Closing)> {
    use earthsci_ast::operator_refs::Definition;
    let defs: IndexMap<&String, Definition> = raw
        .iter()
        .map(|(n, e)| (n, Definition::from_json(n, e)))
        .filter(|(_, d)| d.body.is_some() && !d.is_rule)
        .collect();
    fn fr(
        n: &str,
        defs: &IndexMap<&String, earthsci_ast::operator_refs::Definition>,
        stack: &mut Vec<String>,
        free: &mut HashMap<String, Vec<String>>,
    ) -> R<Vec<String>> {
        if let Some(hit) = free.get(n) {
            return Ok(hit.clone());
        }
        if stack.iter().any(|s| s == n) {
            return Err(BandError(format!("operator `{n}` references itself")));
        }
        let Some(d) = defs.get(&n.to_string()) else {
            return Ok(Vec::new());
        };
        let ps: HashSet<&str> = d.params.iter().map(String::as_str).collect();
        let mut acc: std::collections::BTreeSet<String> = d
            .free_names()
            .iter()
            .filter(|x| !ps.contains(x.as_str()))
            .cloned()
            .collect();
        stack.push(n.to_string());
        for c in d.callees() {
            for x in fr(c, defs, stack, free)? {
                if !ps.contains(x.as_str()) {
                    acc.insert(x);
                }
            }
        }
        stack.pop();
        let out: Vec<String> = acc.into_iter().collect();
        free.insert(n.to_string(), out.clone());
        Ok(out)
    }
    let mut free = HashMap::new();
    let mut closing = HashMap::new();
    for n in defs.keys() {
        let c = fr(n, &defs, &mut Vec::new(), &mut free)?;
        if !c.is_empty() {
            closing.insert((*n).clone(), c);
        }
    }
    if closing.is_empty() {
        return Ok((raw.clone(), closing));
    }
    let mut out = IndexMap::new();
    for (n, e) in raw {
        let Some(d) = defs.get(n) else {
            out.insert(n.clone(), e.clone());
            continue;
        };
        let mut e = e.clone();
        let mut params: Vec<Value> = d.params.iter().map(|p| Value::from(p.as_str())).collect();
        for x in closing.get(n).into_iter().flatten() {
            if !d.params.contains(x) {
                params.push(Value::from(x.as_str()));
            }
        }
        let body = d.body.as_ref().expect("a definition with a body");
        let closed = close_refs(body, &closing);
        if let Value::Object(m) = &mut e {
            m.insert("params".into(), Value::Array(params));
            if closed != *body {
                m.insert(
                    "body".into(),
                    serde_json::to_value(&closed).expect("a body serializes"),
                );
            }
        }
        out.insert(n.clone(), e);
    }
    Ok((out, closing))
}

/// `e` with every reference binding its definition's closing names to
/// themselves (a binding already given is kept).
pub(crate) fn close_refs(e: &Expr, closing: &HashMap<String, Vec<String>>) -> Expr {
    let Expr::Operator(n) = e else {
        return e.clone();
    };
    let mut n2 = n.map_children(&mut |c| close_refs(c, closing));
    if REF_OPS.contains(&n2.op.as_str()) {
        if let Some(names) = n2.name.as_ref().and_then(|d| closing.get(d)) {
            let b = n2.bindings.get_or_insert_with(HashMap::new);
            for x in names {
                b.entry(x.clone())
                    .or_insert_with(|| Expr::Variable(x.clone()));
            }
        }
    }
    Expr::Operator(std::sync::Arc::new(n2))
}

// ── carrying definitions into derived documents ─────────────────────────────

fn raw_refs(x: &Value, acc: &mut Vec<(String, Option<Value>)>) {
    match x {
        Value::Object(m) => {
            if let (Some(Value::String(o)), Some(Value::String(name))) =
                (m.get("op"), m.get("name"))
            {
                if REF_OPS.contains(&o.as_str()) {
                    acc.push((name.clone(), m.get("bindings").cloned()));
                }
            }
            for v in m.values() {
                raw_refs(v, acc);
            }
        }
        Value::Array(a) => a.iter().for_each(|v| raw_refs(v, acc)),
        _ => {}
    }
}

impl RefLayer {
    fn reached(self: &Rc<Self>, refs: Vec<(String, Option<Value>)>) -> R<IndexSet<String>> {
        let mut seen = IndexSet::new();
        let mut work: std::collections::VecDeque<_> = refs.into();
        while let Some((name, _)) = work.pop_front() {
            if seen.contains(&name) {
                continue;
            }
            let d = with_layer(&Some(self.clone()), || self.def(&name))?;
            seen.insert(name);
            let mut inner = Vec::new();
            if let Some(body) = d.entry.get("body") {
                raw_refs(body, &mut inner);
            }
            work.extend(inner.into_iter().filter(|(n, _)| !seen.contains(n)));
        }
        Ok(seen)
    }

    fn block(&self, names: &IndexSet<String>) -> serde_json::Map<String, Value> {
        let mut out = serde_json::Map::new();
        for (n, e) in &self.registry {
            if names.contains(n) {
                out.insert(n.clone(), e.clone());
            }
        }
        let mut derived: Vec<&String> = names
            .iter()
            .filter(|n| !self.registry.contains_key(*n))
            .collect();
        derived.sort();
        let st = self.st.borrow();
        for n in derived {
            if let Some(d) = st.defs.get(n) {
                out.insert(n.clone(), d.entry.clone());
            }
        }
        out
    }
}

/// Give each top-level model of a derived document that holds an operator
/// reference the `operators` block of the definitions its references reach.
pub(crate) fn operators_blocks(doc: &mut Value, l: &Option<Rc<RefLayer>>) -> R<()> {
    let Some(l) = l else { return Ok(()) };
    let key = if l.refop == "apply_operator" {
        "operators"
    } else {
        "expression_templates"
    };
    let Some(Value::Object(models)) = doc.get_mut("models") else {
        return Ok(());
    };
    for (_, m) in models.iter_mut() {
        let mut refs = Vec::new();
        raw_refs(m, &mut refs);
        if refs.is_empty() {
            continue;
        }
        let names = l.reached(refs)?;
        if let Value::Object(mm) = m {
            mm.insert(key.to_string(), Value::Object(l.block(&names)));
        }
    }
    Ok(())
}

#[cfg(test)]
mod name_tests {
    use super::*;

    #[test]
    fn derived_names_parse() {
        assert_eq!(
            derived_name("M.rate.d_x"),
            Some(("M.rate".into(), "x".into(), None))
        );
        assert_eq!(
            derived_name("M.adv.d_f.b2@6x4,3"),
            Some(("M.adv".into(), "f".into(), Some((2, "6x4,3".into()))))
        );
        // the scalar derivative of a band definition
        assert_eq!(
            derived_name("M.adv.d_f.b2@6x4.d_u"),
            Some(("M.adv.d_f.b2@6x4".into(), "u".into(), None))
        );
        assert_eq!(derived_name("M.adv.d_f.b2"), None);
        assert_eq!(derived_name("rate"), None);
    }

    #[test]
    fn shape_tags_round_trip() {
        let sk: ShapeKey = vec![
            ("f".into(), format!("{:?}", vec![6i64, 4])),
            ("g".into(), format!("{:?}", vec![2i64])),
        ];
        assert_eq!(shape_tag(&sk), "6x4,2");
        let d = DefInfo {
            name: "D".into(),
            params: vec!["f".into(), "k".into(), "g".into()],
            body: Expr::Variable("f".into()),
            entry: Value::Null,
            value: HashSet::from(["k".to_string()]),
            array: HashSet::from(["f".to_string(), "g".to_string()]),
            structural: HashSet::new(),
            rank: 1,
        };
        assert_eq!(parse_shape_tag(&d, "6x4,2", "D.d_f.b1@6x4,2").unwrap(), sk);
        assert!(parse_shape_tag(&d, "6x4", "D.d_f.b1@6x4").is_err());
    }
}
