//! The structural point walk over a band (the reference's `_foreach_point`):
//! every (row cell, contracted cell) a band's region and `where` guard admit,
//! with its column cell. Index arithmetic only — `const` gathers folded and
//! integer index expressions evaluated — never a coefficient: the band VALUES
//! are EarthSciAST's to compute, from the evaluation document.

use std::collections::{HashMap, HashSet};

use earthsci_ast::evaluate;
use earthsci_ast::Expr;
use serde_json::Value;

use crate::bands::{Band, BandError};
use crate::clip::RIdx;
use crate::helpers::map_expr_children;

type R<T> = Result<T, BandError>;

/// `foreach_point`'s callback: `(row cell, contracted cell, index binding,
/// column cell)`.
type PointFn<'a> = dyn FnMut(&[i64], &[i64], &HashMap<String, i64>, &[i64]) -> R<()> + 'a;
type PointWhileFn<'a> = dyn FnMut(&[i64], &[i64], &HashMap<String, i64>, &[i64]) -> R<bool> + 'a;

pub fn cellname(v: &str, cell: &[i64]) -> String {
    if cell.is_empty() {
        v.to_string()
    } else {
        format!(
            "{v}[{}]",
            cell.iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

fn fold_const_gathers(e: &Expr, bind: &HashMap<String, f64>) -> R<Expr> {
    let Expr::Operator(n) = e else {
        return Ok(e.clone());
    };
    if n.op == "index" {
        if let Some(Expr::Operator(cn)) = n.args.first() {
            if cn.op == "const" {
                let mut v = cn
                    .value
                    .as_ref()
                    .ok_or_else(|| BandError("const without value".into()))?;
                for x in &n.args[1..] {
                    let i = eval_cidx(x, bind)?;
                    v = v
                        .as_array()
                        .and_then(|a| a.get((i - 1) as usize))
                        .ok_or_else(|| BandError("out-of-range const gather".into()))?;
                }
                return Ok(match v {
                    Value::Number(num) if num.is_i64() => Expr::Integer(num.as_i64().unwrap()),
                    _ => Expr::Number(
                        v.as_f64()
                            .ok_or_else(|| BandError("non-numeric const gather".into()))?,
                    ),
                });
            }
        }
    }
    let mut err: Option<BandError> = None;
    let out = map_expr_children(e, &mut |x| match fold_const_gathers(x, bind) {
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

pub fn eval_cidx(c: &Expr, bind: &HashMap<String, f64>) -> R<i64> {
    let folded = fold_const_gathers(c, bind)?;
    evaluate(&folded, bind)
        .map(|v| v.round() as i64)
        .map_err(|er| BandError(format!("evaluate cidx: {er:?}")))
}

/// A band guard at one point (assemble.jl `_eval_guard`): `const` gathers
/// folded like column indices, then evaluated; nonzero holds.
pub fn eval_guard(g: &Expr, bind: &HashMap<String, f64>) -> R<bool> {
    let folded = fold_const_gathers(g, bind)?;
    evaluate(&folded, bind)
        .map(|v| v != 0.0)
        .map_err(|er| BandError(format!("evaluate guard: {er:?}")))
}

pub fn product_first_fastest(ranges: &[(i64, i64)]) -> Vec<Vec<i64>> {
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

/// Every point of one band (a point its `where` guard excludes is none):
/// `f(row cell, contracted cell, index binding,
/// column cell)` per (row cell, contracted cell), first index fastest, with
/// the column index evaluated there (ghost reads included). The one point
/// walk shared by the pattern and the scatter plans.
pub fn foreach_point(b: &Band, f: &mut PointFn<'_>) -> R<()> {
    all_points(b, &mut |cell, cc, ibind, cidx| {
        f(cell, cc, ibind, cidx).map(|()| true)
    })?;
    Ok(())
}

/// As [`foreach_point`], stopping at the first point where `f` returns
/// false; whether it never did.
pub fn all_points(b: &Band, f: &mut PointWhileFn<'_>) -> R<bool> {
    let cranges: Vec<(i64, i64)> = b.contracted.iter().map(|(_, lo, hi)| (*lo, *hi)).collect();
    for cell in product_first_fastest(&b.rows) {
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
            if let Some(g) = &b.guard {
                if !eval_guard(g, &fbind)? {
                    continue; // a point the `where` guard excludes
                }
            }
            let cidx: Vec<i64> = b
                .cidx
                .iter()
                .map(|c| eval_cidx(c, &fbind))
                .collect::<Result<_, _>>()?;
            if !f(&cell, &cc, &ibind, &cidx)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// A state column whose variable the layout does not hold at all resolves
/// nowhere — an observed the band calculus took for a state, say — and is
/// refused (assemble.jl `_colslot`); a cell past a state's extent is still a
/// ghost read.
pub fn check_state_column(states: &HashSet<String>, v: &str, u: &str) -> R<()> {
    if states.contains(v) {
        return Ok(());
    }
    Err(BandError(format!(
        "column `{v}` (row `{u}`) names no state of the layout; a band column must \
         resolve to a state slot or read past a state's extent"
    )))
}

/// Whether a column cell lies outside its variable's declared extent.
pub fn outside(shape: Option<&Vec<i64>>, cidx: &[i64]) -> bool {
    shape.is_some_and(|sh| cidx.iter().zip(sh).any(|(c, s)| *c < 1 || c > s))
}
