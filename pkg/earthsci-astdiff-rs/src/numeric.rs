//! Numeric assembly — a correctness surface (like the Python binding's):
//! per-cell read resolution to literals, then the official evaluator.

use std::collections::{BTreeMap, HashMap};

use earthsci_ast::expression::evaluate;
use earthsci_ast::substitute::substitute;
use earthsci_ast::types::{EsmFile, Expr};
use serde_json::Value;

use crate::bands::BandError;
use crate::clip::RIdx;
use crate::helpers::map_expr_children;
use crate::jacobian::{jacobian_bands, Wrt};
use crate::system::{ctx_of, sysview};

type R<T> = Result<T, BandError>;

pub struct Env {
    /// name → nested array (1-based cells encoded as serde Value) or scalar
    pub states: HashMap<String, Value>,
    pub params: HashMap<String, Value>,
    pub t: f64,
}

fn gather<'a>(v: &'a Value, cells: &[i64]) -> Option<&'a Value> {
    let mut cur = v;
    for c in cells {
        cur = cur.as_array()?.get((*c - 1) as usize)?;
    }
    Some(cur)
}

fn as_lit(e: &Expr) -> R<f64> {
    match e {
        Expr::Integer(i) => Ok(*i as f64),
        Expr::Number(x) => Ok(*x),
        _ => evaluate(e, &HashMap::new())
            .map_err(|er| BandError(format!("evaluate: {er:?}"))),
    }
}

fn resolve(e: &Expr, env: &Env) -> R<Expr> {
    match e {
        Expr::Variable(name) => {
            if name == "t" {
                return Ok(Expr::Number(env.t));
            }
            for m in [&env.params, &env.states] {
                if let Some(v) = m.get(name) {
                    if let Some(x) = v.as_f64() {
                        return Ok(Expr::Number(x));
                    }
                }
            }
            Err(BandError(format!("unbound name `{name}` in coefficient")))
        }
        Expr::Operator(n) if n.op == "index" => {
            let mut cells = Vec::new();
            for x in &n.args[1..] {
                cells.push(as_lit(&resolve(x, env)?)?.round() as i64);
            }
            match n.args.first() {
                Some(Expr::Variable(t)) => {
                    let arr = env.states.get(t).or_else(|| env.params.get(t))
                        .ok_or_else(|| BandError(format!(
                            "index into unbound array `{t}`")))?;
                    let v = gather(arr, &cells).ok_or_else(|| BandError(
                        format!("out-of-range gather into `{t}`")))?;
                    Ok(Expr::Number(v.as_f64().ok_or_else(|| BandError(
                        "non-numeric gather".into()))?))
                }
                Some(Expr::Operator(cn)) if cn.op == "const" => {
                    let v = cn.value.as_ref().ok_or_else(|| BandError(
                        "const without value".into()))?;
                    let v = gather(v, &cells).ok_or_else(|| BandError(
                        "out-of-range const gather".into()))?;
                    Ok(Expr::Number(v.as_f64().ok_or_else(|| BandError(
                        "non-numeric const gather".into()))?))
                }
                _ => Err(BandError("index into unsupported target".into())),
            }
        }
        Expr::Operator(n) if n.op == "aggregate" => {
            if n.output_idx.as_ref().map(|o| !o.is_empty()).unwrap_or(false) {
                return Err(BandError(
                    "array-producing aggregate in scalar coefficient".into()));
            }
            if let Some(red) = &n.reduce {
                if red != "+" {
                    return Err(BandError(format!(
                        "reduce=`{red}` in scalar coefficient")));
                }
            }
            let empty = HashMap::new();
            let ranges = n.ranges.as_ref().unwrap_or(&empty);
            let mut names: Vec<String> = ranges.keys().cloned().collect();
            names.sort();
            let ctx = crate::bands::Ctx::default();
            let boxes: Vec<(i64, i64)> = names.iter()
                .map(|nm| crate::bands::range_of(&ranges[nm], &ctx))
                .collect::<Result<_, _>>()?;
            let body = n.expr.as_deref().ok_or_else(|| BandError(
                "aggregate without body".into()))?;
            let mut total = 0.0;
            let mut cell: Vec<i64> = boxes.iter().map(|(lo, _)| *lo).collect();
            'outer: loop {
                let binds: HashMap<String, Expr> = names.iter().cloned()
                    .zip(cell.iter().map(|c| Expr::Integer(*c))).collect();
                total += as_lit(&resolve(&substitute(body, &binds), env)?)?;
                let mut k = 0;
                loop {
                    if k == cell.len() {
                        break 'outer;
                    }
                    cell[k] += 1;
                    if cell[k] <= boxes[k].1 {
                        break;
                    }
                    cell[k] = boxes[k].0;
                    k += 1;
                }
                if cell.is_empty() {
                    break;
                }
            }
            Ok(Expr::Number(total))
        }
        Expr::Operator(_) => {
            let mut err: Option<BandError> = None;
            let out = map_expr_children(e, &mut |x| match resolve(x, env) {
                Ok(v) => v,
                Err(er) => {
                    if err.is_none() {
                        err = Some(er);
                    }
                    x.clone()
                }
            });
            match err {
                Some(er) => Err(er),
                None => Ok(out),
            }
        }
        _ => Ok(e.clone()),
    }
}

fn cellname(v: &str, cell: &[i64]) -> String {
    if cell.is_empty() {
        v.to_string()
    } else {
        format!("{v}[{}]", cell.iter().map(|c| c.to_string())
                .collect::<Vec<_>>().join(","))
    }
}

fn fold_const_gathers(e: &Expr, bind: &HashMap<String, f64>) -> R<Expr> {
    let Expr::Operator(n) = e else { return Ok(e.clone()) };
    if n.op == "index" {
        if let Some(Expr::Operator(cn)) = n.args.first() {
            if cn.op == "const" {
                let mut v = cn.value.as_ref().ok_or_else(|| BandError(
                    "const without value".into()))?;
                for x in &n.args[1..] {
                    let i = eval_cidx(x, bind)?;
                    v = v.as_array().and_then(|a| a.get((i - 1) as usize))
                        .ok_or_else(|| BandError(
                            "out-of-range const gather".into()))?;
                }
                return Ok(match v {
                    Value::Number(num) if num.is_i64() =>
                        Expr::Integer(num.as_i64().unwrap()),
                    _ => Expr::Number(v.as_f64().ok_or_else(|| BandError(
                        "non-numeric const gather".into()))?),
                });
            }
        }
    }
    let mut err: Option<BandError> = None;
    let out = map_expr_children(e, &mut |x| {
        match fold_const_gathers(x, bind) {
            Ok(v) => v,
            Err(er) => {
                if err.is_none() {
                    err = Some(er);
                }
                x.clone()
            }
        }
    });
    match err {
        Some(er) => Err(er),
        None => Ok(out),
    }
}

fn eval_cidx(c: &Expr, bind: &HashMap<String, f64>) -> R<i64> {
    let folded = fold_const_gathers(c, bind)?;
    evaluate(&folded, bind)
        .map(|v| v.round() as i64)
        .map_err(|er| BandError(format!("evaluate cidx: {er:?}")))
}

fn product_first_fastest(ranges: &[(i64, i64)]) -> Vec<Vec<i64>> {
    let mut out = Vec::new();
    let mut cell: Vec<i64> = ranges.iter().map(|(lo, _)| *lo).collect();
    if ranges.is_empty() {
        return vec![vec![]];
    }
    loop {
        out.push(cell.clone());
        let mut k = 0;
        loop {
            if k == cell.len() {
                return out;
            }
            cell[k] += 1;
            if cell[k] <= ranges[k].1 {
                break;
            }
            cell[k] = ranges[k].0;
            k += 1;
        }
    }
}

/// Evaluate the band entries at a point: `{(rowname, colname): value}`,
/// contributions summed (§4 additivity, contracted points included).
pub fn assemble_jacobian(file: &EsmFile, model_name: Option<&str>, wrt: Wrt,
                         env: &Env) -> R<BTreeMap<(String, String), f64>> {
    let entries = jacobian_bands(file, model_name, wrt)?;
    let sv = sysview(file, model_name)?;
    let ctx = ctx_of(&sv)?;
    let mut out: BTreeMap<(String, String), f64> = BTreeMap::new();
    for en in &entries {
        let b = &en.band;
        for cell in product_first_fastest(&b.rows) {
            let cranges: Vec<(i64, i64)> =
                b.contracted.iter().map(|(_, lo, hi)| (*lo, *hi)).collect();
            for cc in product_first_fastest(&cranges) {
                let mut ibind: HashMap<String, i64> = HashMap::new();
                for (k, r) in b.ridx.iter().enumerate() {
                    if let RIdx::Name(nm) = r {
                        ibind.insert(nm.clone(), cell[k]);
                    }
                }
                for (k, (nm, _, _)) in b.contracted.iter().enumerate() {
                    ibind.insert(nm.clone(), cc[k]);
                }
                let fbind: HashMap<String, f64> =
                    ibind.iter().map(|(k, v)| (k.clone(), *v as f64)).collect();
                let cidx: Vec<i64> = b.cidx.iter()
                    .map(|c| eval_cidx(c, &fbind))
                    .collect::<Result<_, _>>()?;
                if let Some(vshape) = ctx.shapes.get(&en.v) {
                    if cidx.iter().zip(vshape.iter())
                        .any(|(c, s)| *c < 1 || c > s) {
                        continue;              // ghost read: no column
                    }
                }
                let coef = if ibind.is_empty() {
                    b.coef.clone()
                } else {
                    let sub: HashMap<String, Expr> = ibind.iter()
                        .map(|(k, v)| (k.clone(), Expr::Integer(*v))).collect();
                    substitute(&b.coef, &sub)
                };
                let val = as_lit(&resolve(&coef, env)?)?;
                let key = (cellname(&en.u, &cell), cellname(&en.v, &cidx));
                *out.entry(key).or_insert(0.0) += val;
            }
        }
    }
    Ok(out)
}
