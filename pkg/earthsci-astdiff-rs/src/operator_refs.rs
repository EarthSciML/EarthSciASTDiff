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
//!   coefficient a derived definition `<D>.d_<p>.b<k>`; a use reads the array
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
        let refop = registry
            .values()
            .find_map(|d| d.get("body").and_then(raw_refop))
            .unwrap_or_else(|| "apply_operator".to_string());
        RefLayer {
            registry: registry.clone(),
            refop,
            index_sets,
            st: RefCell::new(State::default()),
        }
    }

    /// The derived definitions written so far, in creation order.
    pub fn derived(&self) -> IndexMap<String, Value> {
        self.st.borrow().derived.clone()
    }
}

fn raw_refop(x: &Value) -> Option<String> {
    match x {
        Value::Object(m) => {
            if let Some(Value::String(o)) = m.get("op") {
                if REF_OPS.contains(&o.as_str()) {
                    return Some(o.clone());
                }
            }
            m.values().find_map(raw_refop)
        }
        Value::Array(a) => a.iter().find_map(raw_refop),
        _ => None,
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

fn derived_name(name: &str) -> Option<(String, String, Option<usize>)> {
    let (head, k) = match name.rfind(".b") {
        Some(i) if name[i + 2..].chars().all(|c| c.is_ascii_digit()) && i + 2 < name.len() => {
            (&name[..i], Some(name[i + 2..].parse().ok()?))
        }
        _ => (name, None),
    };
    let i = head.rfind(".d_")?;
    let p = &head[i + 3..];
    if p.is_empty() || p.contains('.') {
        return None;
    }
    Some((head[..i].to_string(), p.to_string(), k))
}

impl RefLayer {
    /// The definition `name`: a registry entry, or a derived definition (one
    /// this layer wrote, or regenerated from its name and the use's
    /// bindings, `ctx` their context).
    pub(crate) fn def(
        &self,
        name: &str,
        bindings: Option<&HashMap<String, Expr>>,
        ctx: Option<&Ctx>,
    ) -> R<Rc<DefInfo>> {
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
                    "operator `{name}`: its body has no typed form here ({e}); a \
                         parameter in a range bound needs EarthSciAST's typed IR to hold it"
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
        let bd = self.def(&base, None, ctx)?;
        if !bd.params.contains(&p) {
            return Err(BandError(format!(
                "operator reference to `{name}`: `{base}` has no parameter `{p}`"
            )));
        }
        match k {
            None => {
                self.dparam(&bd, &p)?;
            }
            Some(_) => {
                let (Some(b), Some(c)) = (bindings, ctx) else {
                    return Err(BandError(format!(
                        "derived definition `{name}` is reached without the bindings of a use"
                    )));
                };
                let sk = self.shapekey(&bd, b, c)?;
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
            if let RangeSpec::IndexSetRef { from, .. } = r {
                if ps.contains(from) {
                    d.structural.insert(from.clone());
                }
            }
        }
        if REF_OPS.contains(&n.op.as_str()) {
            let callee = self.def(n.name.as_deref().unwrap_or(""), None, None)?;
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
            o if REF_OPS.contains(&o) => {
                self.def(n.name.as_deref().unwrap_or(""), None, None)?.rank
            }
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
                "`{name}` ({what}) differs between two binding shapes of one definition; a \
                 definition used at more than one binding shape is not supported"
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
        let site = Site::of(&Expr::Variable(p.to_string()));
        let ps: HashSet<String> = d.params.iter().cloned().collect();
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
                    if let RangeSpec::IndexSetRef { from, .. } = r {
                        used.insert(from.clone());
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
    let d = l
        .def(n.name.as_deref().unwrap_or(""), None, None)
        .map_err(be)?;
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
    let d = l
        .def(n.name.as_deref().unwrap_or(""), None, None)
        .map_err(be)?;
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

/// A binding shape: per array parameter its extents, per structural one its
/// literal (in parameter order), keyed as text.
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
                let d = self.def(n.name.as_deref().unwrap_or(""), None, None)?;
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
                return Err(BandError(format!(
                    "operator `{}`: its parameter `{p}` fills a structural field; an array \
                     definition with one is not supported by this binding yet",
                    d.name
                )));
            }
            if d.array.contains(p) {
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
        for (p, v) in sk {
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
        let Expr::Operator(n) = &d.body else {
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
        if d.rank == 0 {
            return Err(BandError(format!("`{}` is a scalar definition", d.name)));
        }
        let mut ctx = self.def_ctx(sk);
        let me = current_layer();
        ctx.refs = me.clone();
        let Expr::Operator(n) = &d.body else {
            return Err(BandError(format!(
                "`{}`: an array definition's body is a leaf",
                d.name
            )));
        };
        let (ext, _) = body_extents(n, &ctx)?;
        let ps: HashSet<String> = d.params.iter().cloned().collect();
        let raw = with_defparams(Some(ps), || ref_bands(&d.body, p, &ctx, &ext, co))?;
        let bs: Vec<Band> = merge_bands(raw).iter().map(normalize_band).collect();
        let mut out = Vec::new();
        for (k, mut b) in bs.into_iter().enumerate() {
            if co {
                b.coef = self.generic_coef(d, p, k + 1, &b)?;
            }
            out.push(b);
        }
        let out = Rc::new(out);
        self.st.borrow_mut().defbands.insert(key, out.clone());
        Ok(out)
    }

    fn generic_coef(&self, d: &DefInfo, p: &str, k: usize, b: &Band) -> R<Expr> {
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
        let name = format!("{}.d_{}.b{}", d.name, p, k);
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
        o => Err(BandError(format!(
            "an array definition's body is a `{o}`, not a faq or makearray"
        ))),
    }
}

// ── uses of array definitions ──────────────────────────────────────────────

struct Placeholder {
    name: String,
    r: Expr,
    ext: Vec<i64>,
    org: Vec<i64>,
}

fn abstract_refs(l: &RefLayer, e: &Expr, ctx: &Ctx) -> R<(Expr, Vec<Placeholder>)> {
    let mut found: IndexMap<String, (Expr, String)> = IndexMap::new();
    let out = abstract_walk(l, e, &mut found, &HashSet::new())?;
    let mut ps = Vec::new();
    for (_, (r, name)) in found {
        let n = ref_node(&r).unwrap();
        let d = l.def(n.name.as_deref().unwrap_or(""), None, None)?;
        let (ext, org) = l.ref_extents(&d, n.bindings.as_ref().unwrap_or(&HashMap::new()), ctx)?;
        ps.push(Placeholder { name, r, ext, org });
    }
    Ok((out, ps))
}

fn abstract_walk(
    l: &RefLayer,
    e: &Expr,
    found: &mut IndexMap<String, (Expr, String)>,
    scope: &HashSet<String>,
) -> R<Expr> {
    let Expr::Operator(n) = e else {
        return Ok(e.clone());
    };
    if REF_OPS.contains(&n.op.as_str()) {
        let d = l.def(n.name.as_deref().unwrap_or(""), None, None)?;
        if d.rank > 0 {
            let fv = free_vars(e);
            if fv.iter().any(|x| scope.contains(x)) {
                return Err(BandError(format!(
                    "the array-valued operator reference `{}` binds an argument that reads an \
                     index bound around it; not supported",
                    d.name
                )));
            }
            let k = skey(e);
            let len = found.len();
            let name = found
                .entry(k)
                .or_insert_with(|| (e.clone(), format!("§ref{}", len + 1)))
                .1
                .clone();
            return Ok(Expr::Variable(name));
        }
    }
    let mut scope2 = scope.clone();
    if n.op == "faq" {
        scope2.extend(binders(n));
    }
    let mut err = None;
    let out = Expr::operator(n.map_children(&mut |c: &Expr| match abstract_walk(
        l, c, found, &scope2,
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
        let (rhs2, ps) = abstract_refs(&l, rhs, ctx)?;
        if ps.is_empty() {
            return bands_in(rhs, v, ctx, shape_u, co);
        }
        let mut ctx2 = ctx.clone();
        for p in &ps {
            ctx2.shapes.insert(p.name.clone(), p.ext.clone());
            ctx2.origins.insert(p.name.clone(), p.org.clone());
        }
        let fv = earthsci_ast::free_variables(&rhs2);
        let mut out = Vec::new();
        if fv.contains(v) {
            out.extend(bands_in(&rhs2, v, &ctx2, shape_u, co)?);
        }
        let restore: HashMap<String, Expr> =
            ps.iter().map(|p| (p.name.clone(), p.r.clone())).collect();
        for p in &ps {
            if !fv.contains(&p.name) || !earthsci_ast::free_variables(&p.r).contains(v) {
                continue;
            }
            let inner = ref_value_bands(&l, &p.r, v, ctx, co)?;
            if inner.is_empty() {
                continue;
            }
            let outer = bands_in(&rhs2, &p.name, &ctx2, shape_u, co)?;
            out.extend(compose(&outer, &inner, co)?);
        }
        Ok(out
            .into_iter()
            .map(|mut b| {
                b.cidx = b.cidx.iter().map(|c| subst_names(c, &restore)).collect();
                b.coef = subst_names(&b.coef, &restore);
                b.guard = b.guard.as_ref().map(|g| subst_names(g, &restore));
                b
            })
            .collect())
    })
}

fn ref_value_bands(l: &RefLayer, r: &Expr, v: &str, ctx: &Ctx, co: bool) -> R<Vec<Band>> {
    let n = ref_node(r).unwrap();
    let d = l.def(n.name.as_deref().unwrap_or(""), None, None)?;
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
        let db = l.def_bands(&d, &sk, p, co)?;
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
        let bound: Vec<Band> = db.iter().map(|x| bind_band(x, b, &d.params)).collect();
        out.extend(compose(&bound, &inner, co)?);
    }
    Ok(out)
}

fn bind_band(b: &Band, args: &HashMap<String, Expr>, params: &[String]) -> Band {
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
    }
    b.cidx = b.cidx.iter().map(|c| subst_names(c, args)).collect();
    b.coef = bind(&b.coef, args);
    b.guard = b.guard.as_ref().map(|g| subst_names(g, args));
    b
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
fn compose(outer: &[Band], inner: &[Band], co: bool) -> R<Vec<Band>> {
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
            for (d, r) in b.ridx.iter().enumerate() {
                let (lo, hi) = b.rows[d];
                if let RIdx::Name(n) = r {
                    sigma.insert(n.clone(), o.cidx[d].clone());
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
            let mut g = conj(o.guard.clone(), b.guard.as_ref().map(sub));
            if !conds.is_empty() {
                g = conj(g, conj_list(conds));
            }
            let cidx: Vec<Expr> = b
                .cidx
                .iter()
                .map(|c| canon_lits(&simplify_jl(&sub(c))))
                .collect();
            let coef = if co {
                simp(&mul(vec![o.coef.clone(), bind(&b.coef, &sigma)]))
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
    l.def(n.name.as_deref().unwrap_or(""), None, None)
        .map(|d| d.rank > 0)
        .unwrap_or(false)
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
    fn reached(
        self: &Rc<Self>,
        refs: Vec<(String, Option<Value>)>,
        ctx: &Ctx,
    ) -> R<IndexSet<String>> {
        let mut seen = IndexSet::new();
        let mut work: std::collections::VecDeque<_> = refs.into();
        while let Some((name, braw)) = work.pop_front() {
            if seen.contains(&name) {
                continue;
            }
            let b: HashMap<String, Expr> = match braw {
                Some(Value::Object(m)) => m
                    .into_iter()
                    .map(|(k, v)| {
                        Ok((
                            k,
                            serde_json::from_value(v).map_err(|e| BandError(e.to_string()))?,
                        ))
                    })
                    .collect::<R<_>>()?,
                _ => HashMap::new(),
            };
            let d = with_layer(&Some(self.clone()), || self.def(&name, Some(&b), Some(ctx)))?;
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
pub(crate) fn operators_blocks(doc: &mut Value, l: &Option<Rc<RefLayer>>, ctx: &Ctx) -> R<()> {
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
        let names = l.reached(refs, ctx)?;
        if let Value::Object(mm) = m {
            mm.insert(key.to_string(), Value::Object(l.block(&names)));
        }
    }
    Ok(())
}
