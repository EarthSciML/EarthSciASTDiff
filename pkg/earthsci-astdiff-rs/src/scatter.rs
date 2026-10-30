//! Scatter plans (the reference's scatter.jl, spec §8): the precomputed,
//! checked map from band-field values to a consumer's storage layout. Gated
//! byte-for-byte by `tests/goldens/<fixture>.<layout>.plan.json`.
//!
//! All plan indices are 0-based. Band field `k`, cell `f`: field `J_{k+1}` of
//! the derived evaluation document, column-major flat cell `f` of its shape
//! (row variable's shape, then each contracted extent). Lane layouts are
//! per-cell templates — lane `ℓ` is cell `ℓ` of the lane domain, and the lane
//! capacity `L` is a scatter-time argument, so one plan serves every leaf.

use std::collections::{BTreeMap, HashMap, HashSet};

use earthsci_ast::EsmFile;
use serde_json::{json, Value};

use crate::bands::{BandError, Ctx};
use crate::clip::RIdx;
use crate::emit::{block_diagonal_lu, detect_structure, offset_class, BlockLu};
use crate::jacobian::{jacobian_bands_view, JacEntry, Wrt};
use crate::numeric::{eval_cidx, foreach_point};
use crate::system::{ctx_of, observed_defs, sysview, view_states, SysView};

type R<T> = Result<T, BandError>;

/// Target storage for a [`ScatterPlan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Dense `n × n` block per cell, lane-SoA: `J[r, c]` of lane `ℓ` at
    /// `(r + n·c)·L + ℓ`; block order = the factorization plan's Markowitz
    /// `block_vars`, then untouched states (sorted).
    LaneBlocks,
    /// KPP-style sparse block per cell: `q·L + ℓ`, `q` the index of `(r, c)`
    /// in `lu_pattern` (fill-in included).
    LaneSparseBlocks,
    /// Block-banded per column along index set `axis`:
    /// `((((c·nvar + r)·(kl+ku+1) + (d+kl))·N + k)·L + ℓ`, `d = col − k`.
    ColumnBanded { axis: String },
    /// General sparse rows over the canonical state order.
    Csr,
}

impl Layout {
    fn name(&self) -> &'static str {
        match self {
            Layout::LaneBlocks => "LaneBlocks",
            Layout::LaneSparseBlocks => "LaneSparseBlocks",
            Layout::ColumnBanded { .. } => "ColumnBanded",
            Layout::Csr => "Csr",
        }
    }
}

/// The layout-specific half of a plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanKind {
    LaneBlocks {
        cells: Vec<i64>,
        block_vars: Vec<String>,
        /// (band k, block offset r + n·c), sorted by (offset, k).
        items: Vec<(usize, usize)>,
    },
    LaneSparseBlocks {
        cells: Vec<i64>,
        block_vars: Vec<String>,
        lu_pattern: Vec<(usize, usize)>,
        /// (band k, q), sorted by (q, k).
        items: Vec<(usize, usize)>,
    },
    ColumnBanded {
        cells: Vec<i64>,
        axis: String,
        axis_dim: usize,
        vars: Vec<String>,
        kl: i64,
        ku: i64,
        /// (band k, row level, offset without lane), sorted by (offset, k).
        items: Vec<(usize, usize, usize)>,
    },
    Csr {
        rowptr: Vec<usize>,
        colind: Vec<usize>,
        /// (band k, field cell, nz position), sorted by (position, k, cell).
        items: Vec<(usize, usize, usize)>,
    },
}

/// Band fields → storage positions for one (band list, layout). Plain data:
/// `Send + Sync`, built once, shared by every leaf using the operator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScatterPlan {
    /// Canonical states (sorted names) and their shapes (`[]` = scalar).
    pub states: Vec<(String, Vec<i64>)>,
    /// Per band: row variable, column variable, field shape.
    pub bands: Vec<(String, String, Vec<i64>)>,
    pub kind: PlanKind,
}

/// Field shape of a band: the row variable's shape, then each contracted
/// extent (1..hi).
pub fn field_shape(en: &JacEntry, ctx: &Ctx) -> Vec<i64> {
    let mut s = ctx.shapes.get(&en.u).cloned().unwrap_or_default();
    s.extend(en.band.contracted.iter().map(|(_, _, hi)| *hi));
    s
}

/// Column-major 0-based flat index of a 1-based cell.
pub fn flat(cell: &[i64], shape: &[i64]) -> usize {
    let (mut f, mut stride) = (0i64, 1i64);
    for (c, n) in cell.iter().zip(shape) {
        f += (c - 1) * stride;
        stride *= n;
    }
    f as usize
}

pub(crate) fn ncells(shape: &[i64]) -> usize {
    shape.iter().product::<i64>() as usize
}

/// 1-based cell of a 0-based column-major flat index.
pub(crate) fn unflat(mut f: usize, shape: &[i64]) -> Vec<i64> {
    shape
        .iter()
        .map(|n| {
            let c = f % *n as usize;
            f /= *n as usize;
            c as i64 + 1
        })
        .collect()
}

fn cellname(v: &str, cell: &[i64]) -> String {
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

fn err<T>(layout: &Layout, msg: String) -> R<T> {
    Err(BandError(format!("{}: {msg}", layout.name())))
}

impl ScatterPlan {
    /// The plan of a model's state Jacobian (the reference's
    /// `scatter_plan(input, layout)`).
    pub fn from_file(file: &EsmFile, model_name: Option<&str>, layout: &Layout) -> R<ScatterPlan> {
        let sv = sysview(file, model_name)?;
        let ctx = ctx_of(&sv)?;
        let obs = observed_defs(&sv)?;
        let entries = jacobian_bands_view(&sv, &ctx, &obs, Wrt::States, None)?;
        ScatterPlan::build(&sv, &entries, layout)
    }

    /// Build from a band list. Refuses (naming the reason) a layout that does
    /// not fit, and always an empty band list on a model with array states.
    pub fn build(sv: &SysView, entries: &[JacEntry], layout: &Layout) -> R<ScatterPlan> {
        let ctx = ctx_of(sv)?;
        let obs = observed_defs(sv)?;
        let names = view_states(sv, &obs);
        if names.is_empty() {
            return err(layout, "the model has no states".into());
        }
        let states: Vec<(String, Vec<i64>)> = names
            .iter()
            .map(|u| (u.clone(), ctx.shapes.get(u).cloned().unwrap_or_default()))
            .collect();
        if entries.is_empty() && states.iter().any(|(_, s)| !s.is_empty()) {
            return err(
                layout,
                "empty band list for a model with array states (an \
                all-zero Jacobian is never a valid plan — check the band calculus \
                against this document)"
                    .into(),
            );
        }
        let known: HashSet<&String> = names.iter().collect();
        for en in entries {
            if !known.contains(&en.v) {
                return err(
                    layout,
                    format!(
                        "band column `{}` is not a state; scatter plans cover wrt = :states",
                        en.v
                    ),
                );
            }
        }
        let bands: Vec<(String, String, Vec<i64>)> = entries
            .iter()
            .map(|en| (en.u.clone(), en.v.clone(), field_shape(en, &ctx)))
            .collect();
        let kind = match layout {
            Layout::LaneBlocks => {
                let cells = lane_cells(layout, &states)?;
                let (_, bv) = block_order(layout, entries, &ctx, &states)?;
                let pos: HashMap<&String, usize> =
                    bv.iter().enumerate().map(|(i, u)| (u, i)).collect();
                let n = bv.len();
                let mut items: Vec<(usize, usize)> = entries
                    .iter()
                    .enumerate()
                    .map(|(k, en)| (k, pos[&en.u] + n * pos[&en.v]))
                    .collect();
                items.sort_by_key(|&(k, off)| (off, k));
                PlanKind::LaneBlocks {
                    cells,
                    block_vars: bv,
                    items,
                }
            }
            Layout::LaneSparseBlocks => {
                let cells = lane_cells(layout, &states)?;
                let (fp, bv) = block_order(layout, entries, &ctx, &states)?;
                let n0 = fp.block_vars.len();
                let mut lu: Vec<(usize, usize)> = Vec::new();
                for i in 0..n0 {
                    for j in 0..n0 {
                        if fp.lu[i][j] {
                            lu.push((i, j));
                        }
                    }
                }
                lu.extend((n0..bv.len()).map(|i| (i, i)));
                lu.sort();
                let q: HashMap<(usize, usize), usize> =
                    lu.iter().enumerate().map(|(i, rc)| (*rc, i)).collect();
                let pos: HashMap<&String, usize> =
                    bv.iter().enumerate().map(|(i, u)| (u, i)).collect();
                let mut items: Vec<(usize, usize)> = entries
                    .iter()
                    .enumerate()
                    .map(|(k, en)| (k, q[&(pos[&en.u], pos[&en.v])]))
                    .collect();
                items.sort_by_key(|&(k, q)| (q, k));
                PlanKind::LaneSparseBlocks {
                    cells,
                    block_vars: bv,
                    lu_pattern: lu,
                    items,
                }
            }
            Layout::ColumnBanded { axis } => column_banded(layout, axis, sv, entries, &states)?,
            Layout::Csr => csr(entries, &states, &bands)?,
        };
        Ok(ScatterPlan {
            states,
            bands,
            kind,
        })
    }

    /// The golden form (`stable_json` of it is the `.plan.json` golden).
    pub fn to_json(&self) -> Value {
        let mut d = serde_json::Map::new();
        d.insert("wrt".into(), json!("states"));
        d.insert(
            "states".into(),
            Value::Array(
                self.states
                    .iter()
                    .map(|(u, s)| json!({"name": u, "shape": s}))
                    .collect(),
            ),
        );
        d.insert(
            "bands".into(),
            Value::Array(
                self.bands
                    .iter()
                    .map(|(u, v, s)| json!({"u": u, "v": v, "shape": s}))
                    .collect(),
            ),
        );
        let pairs = |it: &[(usize, usize)]| -> Value {
            Value::Array(it.iter().map(|(a, b)| json!([a, b])).collect())
        };
        let triples = |it: &[(usize, usize, usize)]| -> Value {
            Value::Array(it.iter().map(|(a, b, c)| json!([a, b, c])).collect())
        };
        match &self.kind {
            PlanKind::LaneBlocks {
                cells,
                block_vars,
                items,
            } => {
                d.insert("layout".into(), json!("lane_blocks"));
                d.insert("cells".into(), json!(cells));
                d.insert("block_vars".into(), json!(block_vars));
                d.insert("items".into(), pairs(items));
            }
            PlanKind::LaneSparseBlocks {
                cells,
                block_vars,
                lu_pattern,
                items,
            } => {
                d.insert("layout".into(), json!("lane_sparse_blocks"));
                d.insert("cells".into(), json!(cells));
                d.insert("block_vars".into(), json!(block_vars));
                d.insert("lu_pattern".into(), pairs(lu_pattern));
                d.insert("items".into(), pairs(items));
            }
            PlanKind::ColumnBanded {
                cells,
                axis,
                axis_dim,
                vars,
                kl,
                ku,
                items,
            } => {
                d.insert("layout".into(), json!("column_banded"));
                d.insert("cells".into(), json!(cells));
                d.insert("axis".into(), json!(axis));
                d.insert("axis_dim".into(), json!(axis_dim));
                d.insert("vars".into(), json!(vars));
                d.insert("kl".into(), json!(kl));
                d.insert("ku".into(), json!(ku));
                d.insert("items".into(), triples(items));
            }
            PlanKind::Csr {
                rowptr,
                colind,
                items,
            } => {
                d.insert("layout".into(), json!("csr"));
                d.insert("rowptr".into(), json!(rowptr));
                d.insert("colind".into(), json!(colind));
                d.insert("items".into(), triples(items));
            }
        }
        Value::Object(d)
    }

    /// Lanes of a lane plan (cells) or a column plan (columns); 1 for CSR.
    pub fn lanes(&self) -> usize {
        match &self.kind {
            PlanKind::LaneBlocks { cells, .. } | PlanKind::LaneSparseBlocks { cells, .. } => {
                ncells(cells)
            }
            PlanKind::ColumnBanded {
                cells, axis_dim, ..
            } => ncells(cells) / cells[*axis_dim] as usize,
            PlanKind::Csr { .. } => 1,
        }
    }

    /// Output length at lane capacity `l`.
    pub fn out_len(&self, l: usize) -> usize {
        match &self.kind {
            PlanKind::LaneBlocks { block_vars, .. } => block_vars.len().pow(2) * l,
            PlanKind::LaneSparseBlocks { lu_pattern, .. } => lu_pattern.len() * l,
            PlanKind::ColumnBanded {
                cells,
                axis_dim,
                vars,
                kl,
                ku,
                ..
            } => vars.len().pow(2) * (kl + ku + 1) as usize * cells[*axis_dim] as usize * l,
            PlanKind::Csr { colind, .. } => colind.len(),
        }
    }

    /// Field cell of column `lane` at row level `lev` (both 0-based).
    pub(crate) fn column_cell(cells: &[i64], axis_dim: usize, lev: usize, lane: usize) -> usize {
        let (mut f, mut stride, mut rest) = (0usize, 1usize, lane);
        for (m, n) in cells.iter().enumerate() {
            let n = *n as usize;
            let c = if m == axis_dim {
                lev
            } else {
                let c = rest % n;
                rest /= n;
                c
            };
            f += c * stride;
            stride *= n;
        }
        f
    }

    /// `(band k, field cell, output position)` per contribution, in
    /// accumulation order: plan order, lanes innermost.
    pub(crate) fn contributions(&self, l: usize) -> Vec<(usize, usize, usize)> {
        let nl = self.lanes();
        match &self.kind {
            PlanKind::LaneBlocks { items, .. } | PlanKind::LaneSparseBlocks { items, .. } => items
                .iter()
                .flat_map(|&(k, off)| (0..nl).map(move |ln| (k, ln, off * l + ln)))
                .collect(),
            PlanKind::ColumnBanded {
                cells,
                axis_dim,
                items,
                ..
            } => items
                .iter()
                .flat_map(|&(k, lev, off)| {
                    (0..nl).map(move |ln| {
                        (
                            k,
                            Self::column_cell(cells, *axis_dim, lev, ln),
                            off * l + ln,
                        )
                    })
                })
                .collect(),
            PlanKind::Csr { items, .. } => items.clone(),
        }
    }

    /// Check this plan against a model's own band list and precompute its
    /// scatter at lane capacity `l` (spec §8: a plan is never applied
    /// unchecked). The plan's `states` and `bands` must equal the model's,
    /// and every contribution is decoded against the model's bands: each must
    /// land on the (row, column) cell its band point differentiates, and every
    /// band point with a column must be covered exactly once — so a plan from
    /// another document, build or stencil is refused.
    pub fn bind(&self, file: &EsmFile, model_name: Option<&str>, l: usize) -> R<BoundScatter> {
        let sv = sysview(file, model_name)?;
        let ctx = ctx_of(&sv)?;
        let obs = observed_defs(&sv)?;
        let entries = jacobian_bands_view(&sv, &ctx, &obs, Wrt::States, None)?;
        self.bind_entries(&sv, &entries, l)
    }

    /// [`ScatterPlan::bind`] against an already computed band list.
    pub fn bind_entries(&self, sv: &SysView, entries: &[JacEntry], l: usize) -> R<BoundScatter> {
        let berr = |msg: String| -> R<BoundScatter> { Err(BandError(format!("bind: {msg}"))) };
        let ctx = ctx_of(sv)?;
        let obs = observed_defs(sv)?;
        let states: Vec<(String, Vec<i64>)> = view_states(sv, &obs)
            .into_iter()
            .map(|u| {
                let s = ctx.shapes.get(&u).cloned().unwrap_or_default();
                (u, s)
            })
            .collect();
        if states != self.states {
            return berr("the plan's states are not this model's".into());
        }
        if entries.len() != self.bands.len()
            || entries
                .iter()
                .zip(&self.bands)
                .any(|(en, (u, v, s))| &en.u != u || &en.v != v || &field_shape(en, &ctx) != s)
        {
            return berr(format!(
                "the plan was built from a different band list than this model ({} vs \
                 {} bands, or a band's variables or field shape differ)",
                self.bands.len(),
                entries.len()
            ));
        }
        let nl = self.lanes();
        if l < nl {
            return berr(format!("lane capacity {l} < {nl} lanes"));
        }
        // The model's own points: (band k, field cell) → (row, col) cell
        // names, or None for a ghost read (a value with no storage position).
        let vshape: HashMap<&String, &Vec<i64>> = states.iter().map(|(u, s)| (u, s)).collect();
        let mut want: HashMap<(usize, usize), Option<(String, String)>> = HashMap::new();
        for (k, en) in entries.iter().enumerate() {
            let sh = &self.bands[k].2;
            let vs = vshape[&en.v];
            foreach_point(&en.band, &mut |cell, cc, _, cidx| {
                let full: Vec<i64> = cell.iter().chain(cc).copied().collect();
                let ghost = cidx.iter().zip(vs.iter()).any(|(c, n)| *c < 1 || c > n);
                want.insert(
                    (k, flat(&full, sh)),
                    if ghost {
                        None
                    } else {
                        Some((cellname(&en.u, cell), cellname(&en.v, cidx)))
                    },
                );
                Ok(())
            })?;
        }
        let targets = self.targets(l);
        let pairs = self.contributions(l);
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        for &(k, f, pos) in &pairs {
            let Some(t) = want.get(&(k, f)) else { continue }; // outside the region: zero
            let Some(t) = t else {
                return berr(format!(
                    "band {} field cell {f} is a ghost read; the plan \
                                     gives it a storage position",
                    k + 1
                ));
            };
            if targets[pos].as_ref() != Some(t) {
                return berr(format!(
                    "band {} field cell {f} is {} ← {} here; the plan \
                                     puts it at {:?}",
                    k + 1,
                    t.0,
                    t.1,
                    targets[pos]
                ));
            }
            if !seen.insert((k, f)) {
                return berr(format!("band {} field cell {f} is scattered twice", k + 1));
            }
        }
        let mut dropped: Vec<_> = want
            .iter()
            .filter(|(kf, t)| t.is_some() && !seen.contains(kf))
            .collect();
        dropped.sort();
        if let Some(((k, f), Some((r, c)))) = dropped.first() {
            return berr(format!(
                "the plan drops band {} field cell {f} ({r} ← {c})",
                k + 1
            ));
        }
        Ok(BoundScatter {
            lanes: l,
            len: self.out_len(l),
            field_lens: self.bands.iter().map(|b| ncells(&b.2)).collect(),
            pairs,
        })
    }

    /// The (row, column) state-cell names each output position holds at lane
    /// capacity `l` (`None`: padding lane or out-of-range band slot).
    pub fn targets(&self, l: usize) -> Vec<Option<(String, String)>> {
        let mut out = vec![None; self.out_len(l)];
        let nl = self.lanes();
        match &self.kind {
            PlanKind::LaneBlocks {
                cells, block_vars, ..
            } => {
                let n = block_vars.len();
                for c in 0..n {
                    for r in 0..n {
                        for ln in 0..nl {
                            let cell = unflat(ln, cells);
                            out[(r + n * c) * l + ln] = Some((
                                cellname(&block_vars[r], &cell),
                                cellname(&block_vars[c], &cell),
                            ));
                        }
                    }
                }
            }
            PlanKind::LaneSparseBlocks {
                cells,
                block_vars,
                lu_pattern,
                ..
            } => {
                for (q, (r, c)) in lu_pattern.iter().enumerate() {
                    for ln in 0..nl {
                        let cell = unflat(ln, cells);
                        out[q * l + ln] = Some((
                            cellname(&block_vars[*r], &cell),
                            cellname(&block_vars[*c], &cell),
                        ));
                    }
                }
            }
            PlanKind::ColumnBanded {
                cells,
                axis_dim,
                vars,
                kl,
                ku,
                ..
            } => {
                let nvar = vars.len();
                let nn = cells[*axis_dim] as usize;
                let w = (kl + ku + 1) as usize;
                for c in 0..nvar {
                    for r in 0..nvar {
                        for j in 0..w {
                            for lev in 0..nn {
                                let col = lev as i64 + j as i64 - kl;
                                if col < 0 || col >= nn as i64 {
                                    continue;
                                }
                                for ln in 0..nl {
                                    let rc =
                                        unflat(Self::column_cell(cells, *axis_dim, lev, ln), cells);
                                    let mut cc = rc.clone();
                                    cc[*axis_dim] = col + 1;
                                    out[(((c * nvar + r) * w + j) * nn + lev) * l + ln] =
                                        Some((cellname(&vars[r], &rc), cellname(&vars[c], &cc)));
                                }
                            }
                        }
                    }
                }
            }
            PlanKind::Csr { rowptr, colind, .. } => {
                let names = state_cells(&self.states);
                for r in 0..rowptr.len() - 1 {
                    for q in rowptr[r]..rowptr[r + 1] {
                        out[q] = Some((names[r].clone(), names[colind[q]].clone()));
                    }
                }
            }
        }
        out
    }
}

/// A [`ScatterPlan`] checked against one model at a lane capacity (see
/// [`ScatterPlan::bind`]), its gather precomputed: fill with
/// [`BoundScatter::scatter`]. Plain data, `Send + Sync`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundScatter {
    pub lanes: usize,
    /// Output length.
    pub len: usize,
    field_lens: Vec<usize>,
    /// (band k, field cell, output position), in accumulation order.
    pairs: Vec<(usize, usize, usize)>,
}

impl BoundScatter {
    /// Scatter band fields (`fields[k][cell]`, see [`crate::band_fields`])
    /// into `out`: zeroed, then `+=` in plan order.
    pub fn scatter(&self, fields: &[Vec<f64>], out: &mut [f64]) -> R<()> {
        if fields.len() != self.field_lens.len()
            || fields
                .iter()
                .zip(&self.field_lens)
                .any(|(f, n)| f.len() != *n)
        {
            return Err(BandError(
                "scatter: band fields do not match the plan's bands".into(),
            ));
        }
        if out.len() != self.len {
            return Err(BandError(format!(
                "scatter: output has length {}; the plan needs {}",
                out.len(),
                self.len
            )));
        }
        out.iter_mut().for_each(|x| *x = 0.0);
        for &(k, f, pos) in &self.pairs {
            out[pos] += fields[k][f];
        }
        Ok(())
    }
}

/// Every state cell name, in canonical order (the rows of a `Csr` plan).
fn state_cells(states: &[(String, Vec<i64>)]) -> Vec<String> {
    states
        .iter()
        .flat_map(|(u, s)| (0..ncells(s)).map(move |f| cellname(u, &unflat(f, s))))
        .collect()
}

fn lane_cells(layout: &Layout, states: &[(String, Vec<i64>)]) -> R<Vec<i64>> {
    let mut shapes: Vec<&Vec<i64>> = Vec::new();
    for (_, s) in states {
        if !shapes.contains(&s) {
            shapes.push(s);
        }
    }
    if shapes.len() != 1 {
        let mut ss: Vec<String> = shapes.iter().map(|s| format!("{s:?}")).collect();
        ss.sort();
        return err(
            layout,
            format!(
                "every state must share one cell shape; got {}",
                ss.join(", ")
            ),
        );
    }
    Ok(shapes[0].clone())
}

fn block_order(
    layout: &Layout,
    entries: &[JacEntry],
    ctx: &Ctx,
    states: &[(String, Vec<i64>)],
) -> R<(BlockLu, Vec<String>)> {
    // A lane template reads one field cell per lane; a contracted band's
    // field has a cell per contracted point too, which it would drop.
    for (k, en) in entries.iter().enumerate() {
        if !en.band.contracted.is_empty() {
            return err(
                layout,
                format!(
                    "band {} ({} ← {}) has contracted dimensions",
                    k + 1,
                    en.u,
                    en.v
                ),
            );
        }
    }
    let st = detect_structure(entries, ctx);
    if st != "block_diagonal" {
        return err(
            layout,
            format!("needs a :block_diagonal Jacobian (per-cell coupling only); this one is :{st}"),
        );
    }
    let fp = block_diagonal_lu(entries);
    let mut extra: Vec<String> = states
        .iter()
        .map(|(u, _)| u.clone())
        .filter(|u| !fp.block_vars.contains(u))
        .collect();
    extra.sort();
    let mut bv = fp.block_vars.clone();
    bv.extend(extra);
    Ok((fp, bv))
}

fn column_banded(
    layout: &Layout,
    axis: &str,
    sv: &SysView,
    entries: &[JacEntry],
    states: &[(String, Vec<i64>)],
) -> R<PlanKind> {
    let scalars: Vec<&str> = states
        .iter()
        .filter(|(_, s)| s.is_empty())
        .map(|(u, _)| u.as_str())
        .collect();
    if !scalars.is_empty() {
        return err(
            layout,
            format!(
                "every state must be an array over `{axis}`; {} is scalar",
                scalars.join(", ")
            ),
        );
    }
    let mut dimnames: Vec<Vec<String>> = Vec::new();
    for (u, _) in states {
        let d = sv.variables[u].shape.clone().unwrap_or_default();
        if !dimnames.contains(&d) {
            dimnames.push(d);
        }
    }
    if dimnames.len() != 1 {
        return err(
            layout,
            format!("every state must share one shape; got {dimnames:?}"),
        );
    }
    let Some(ax) = dimnames[0].iter().position(|x| x == axis) else {
        return err(
            layout,
            format!(
                "`{axis}` is not a dimension of the states {:?}",
                dimnames[0]
            ),
        );
    };
    let cells = states[0].1.clone();
    let nn = cells[ax];
    let vars: Vec<String> = states.iter().map(|(u, _)| u.clone()).collect();
    let vidx: HashMap<&String, usize> = vars.iter().enumerate().map(|(i, u)| (u, i)).collect();
    let nvar = vars.len();
    let mut pts: Vec<(usize, usize, usize, i64, i64)> = Vec::new(); // (k, r, c, level, d)
    for (k, en) in entries.iter().enumerate() {
        let b = &en.band;
        if !b.contracted.is_empty() {
            return err(
                layout,
                format!(
                    "band {} ({} ← {}) has contracted dimensions",
                    k + 1,
                    en.u,
                    en.v
                ),
            );
        }
        for (m, (c, r)) in b.cidx.iter().zip(&b.ridx).enumerate() {
            let cls = offset_class(c, r);
            let ok = if m == ax {
                cls == "diag" || cls == "affine"
            } else {
                cls == "diag"
            };
            if !ok {
                return err(
                    layout,
                    format!(
                        "band {} ({} ← {}) couples cells {}",
                        k + 1,
                        en.u,
                        en.v,
                        if m == ax {
                            format!("at a non-literal offset along `{axis}`")
                        } else {
                            format!("across columns (dimension {} is not the axis)", m + 1)
                        }
                    ),
                );
            }
        }
        let (lo, hi) = b.rows[ax];
        for lev in lo..=hi {
            let mut bind: HashMap<String, f64> = HashMap::new();
            if let RIdx::Name(nm) = &b.ridx[ax] {
                bind.insert(nm.clone(), lev as f64);
            }
            let col = eval_cidx(&b.cidx[ax], &bind)?;
            if col < 1 || col > nn {
                continue; // ghost read: no column
            }
            pts.push((k, vidx[&en.u], vidx[&en.v], lev - 1, col - lev));
        }
    }
    let kl = pts.iter().map(|p| -p.4).max().unwrap_or(0).max(0);
    let ku = pts.iter().map(|p| p.4).max().unwrap_or(0).max(0);
    let w = kl + ku + 1;
    let mut items: Vec<(usize, usize, usize)> = pts
        .iter()
        .map(|&(k, r, c, lev, d)| {
            (
                k,
                lev as usize,
                ((((c * nvar + r) as i64) * w + (d + kl)) * nn + lev) as usize,
            )
        })
        .collect();
    items.sort_by_key(|&(k, _, off)| (off, k));
    Ok(PlanKind::ColumnBanded {
        cells,
        axis: axis.to_string(),
        axis_dim: ax,
        vars,
        kl,
        ku,
        items,
    })
}

fn csr(
    entries: &[JacEntry],
    states: &[(String, Vec<i64>)],
    bands: &[(String, String, Vec<i64>)],
) -> R<PlanKind> {
    let mut base: HashMap<&String, usize> = HashMap::new();
    let mut n = 0usize;
    for (u, s) in states {
        base.insert(u, n);
        n += ncells(s);
    }
    let shape: HashMap<&String, &Vec<i64>> = states.iter().map(|(u, s)| (u, s)).collect();
    let mut pts: Vec<(usize, usize, usize, usize)> = Vec::new(); // (row, col, k, field cell)
    for (k, en) in entries.iter().enumerate() {
        let b = &en.band;
        let vshape = shape[&en.v];
        foreach_point(b, &mut |cell, cc, _, cidx| {
            if cidx.iter().zip(vshape.iter()).any(|(c, s)| *c < 1 || c > s) {
                return Ok(()); // ghost read: no column
            }
            let full: Vec<i64> = cell.iter().chain(cc).copied().collect();
            pts.push((
                base[&en.u] + flat(cell, shape[&en.u]),
                base[&en.v] + flat(cidx, vshape),
                k,
                flat(&full, &bands[k].2),
            ));
            Ok(())
        })?;
    }
    let nz: Vec<(usize, usize)> = pts
        .iter()
        .map(|p| (p.0, p.1))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let pos: BTreeMap<(usize, usize), usize> =
        nz.iter().enumerate().map(|(i, rc)| (*rc, i)).collect();
    let mut rowptr = vec![0usize; n + 1];
    for (r, _) in &nz {
        rowptr[r + 1] += 1;
    }
    for i in 1..=n {
        rowptr[i] += rowptr[i - 1];
    }
    let mut items: Vec<(usize, usize, usize)> = pts
        .iter()
        .map(|&(r, c, k, f)| (k, f, pos[&(r, c)]))
        .collect();
    items.sort_by_key(|&(k, f, p)| (p, k, f));
    Ok(PlanKind::Csr {
        rowptr,
        colind: nz.iter().map(|rc| rc.1).collect(),
        items,
    })
}
