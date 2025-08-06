from typing import Iterable, Tuple, Dict, List
import numpy as np
from shapely.geometry import (
    LineString, MultiLineString, Polygon, MultiPolygon, base
)
from shapely.ops import unary_union, polygonize
from shapely import affinity
from collections import Counter, defaultdict

def _polygonize_lines(geom):
    return list(polygonize(geom))

def _short_side_widths(polys: Iterable[Polygon]) -> np.ndarray:
    vals = []
    for p in polys:
        minx, miny, maxx, maxy = p.bounds
        vals.append(min(maxx - minx, maxy - miny))
    return np.asarray(vals, dtype=float)

def _robust_width(values: np.ndarray) -> float:
    v = values[~np.isnan(values)]
    if v.size == 0: return 0.0
    q75, q25 = np.percentile(v, [75, 25])
    iqr = max(q75 - q25, 1e-9)
    binw = 2 * iqr / (len(v) ** (1 / 3))
    if binw <= 0: return float(np.median(v))
    bins = int(max(5, min(80, np.ceil((v.max() - v.min()) / binw))))
    hist, edges = np.histogram(v, bins=bins)
    mode_idx = int(np.argmax(hist))
    mode = 0.5 * (edges[mode_idx] + edges[mode_idx + 1])
    if hist[mode_idx] < max(3, 0.02 * len(v)):  # fallback to trimmed median
        lo, hi = np.percentile(v, [10, 90])
        v2 = v[(v >= lo) & (v <= hi)]
        return float(np.median(v2)) if v2.size else float(np.median(v))
    return float(mode)

def _row_pitch_from_target(polys: Iterable[Polygon]) -> float:
    ys = np.array([p.centroid.y for p in polys])
    xs = np.array([p.centroid.x for p in polys])
    if len(ys) < 2: return 4.0
    window = 15.0
    dys = []
    for i in range(len(ys)):
        mask = np.abs(xs - xs[i]) < window
        if np.count_nonzero(mask) <= 1: continue
        dy = np.min(np.abs(ys[mask] - ys[i]) + 1e-9)
        if dy > 1e-3: dys.append(dy)
    return float(np.median(dys)) if dys else 4.0

def _quantile_edge(polys: Iterable[Polygon], axis: str, q: float) -> float:
    vals = [p.bounds[0] if axis == "x" else p.bounds[1] for p in polys]
    return float(np.quantile(vals, q)) if vals else 0.0

def align_by_width_edge_rows_area(
    combiner_lines: MultiLineString | LineString,
    target_polys: Iterable[Polygon] | MultiPolygon,
    source_geoms: Iterable[base.BaseGeometry],
    row_shift_range: int = 3,
    edge_quantile: float = 0.02,
    simplify_tol: float = 0.02,
) -> Tuple[Dict[str, float], MultiLineString, base.BaseGeometry]:
    """Width->edge->rows with AREA scoring for row shift."""
    aoi = unary_union(list(polygonize(combiner_lines)))

    tgt: List[Polygon] = [p for p in (target_polys.geoms if isinstance(target_polys, MultiPolygon) else target_polys)
                          if p.intersects(aoi)]
    if not tgt: raise ValueError("No target polygons intersect AOI.")

    src_orig = list(source_geoms)
    src_polys = []
    for g in src_orig:
        if g.geom_type in ("Polygon", "MultiPolygon"):
            src_polys.append(g)
        elif g.geom_type in ("LineString", "MultiLineString"):
            src_polys.extend(_polygonize_lines(g))
        else:
            src_polys.extend(_polygonize_lines(g.boundary))
    if not src_polys:
        raise ValueError("No source polygons could be derived from input.")

    # 1) robust scale
    wt = _robust_width(_short_side_widths(tgt))
    ws = _robust_width(_short_side_widths(src_polys))
    s = (wt / ws) if ws > 0 else 1.0

    src_union_all = unary_union(src_orig)
    cx, cy = src_union_all.centroid.x, src_union_all.centroid.y
    src_scaled = [affinity.scale(g, xfact=s, yfact=s, origin=(cx, cy)) for g in src_orig]

    src_scaled_polys = []
    for g in src_scaled:
        if g.geom_type in ("Polygon", "MultiPolygon"):
            src_scaled_polys.append(g)
        else:
            src_scaled_polys.extend(_polygonize_lines(g))

    # 2) robust edges
    minx_t = _quantile_edge(tgt, axis="x", q=edge_quantile)
    miny_t = _quantile_edge(tgt, axis="y", q=edge_quantile)
    minx_s = _quantile_edge(src_scaled_polys, axis="x", q=edge_quantile)
    miny_s = _quantile_edge(src_scaled_polys, axis="y", q=edge_quantile)
    dx_edge = minx_t - minx_s
    dy_base = miny_t - miny_s

    # 3) pitch + row area overlap scoring
    pitch = _row_pitch_from_target(tgt)
    ys_t = np.array([p.centroid.y for p in tgt])
    y0 = float(np.median(ys_t))

    # target row unions
    tgt_by_row = defaultdict(list)
    for p in tgt:
        i = int(np.round((p.centroid.y - y0) / pitch))
        tgt_by_row[i].append(p)
    tgt_row_union = {i: unary_union([g.simplify(simplify_tol, preserve_topology=True) for g in geoms]).buffer(0)
                     for i, geoms in tgt_by_row.items()}

    # source rows before dy (simplify once)
    src_by_row = defaultdict(list)
    for p in src_scaled_polys:
        i = int(np.round((p.centroid.y - y0) / pitch))
        src_by_row[i].append(p.simplify(simplify_tol, preserve_topology=True))

    candidates = []
    for k in range(-row_shift_range, row_shift_range + 1):
        dy_k = dy_base + k * pitch
        total_area = 0.0
        total_union = 0.0
        for i in set(tgt_row_union.keys()) & set(src_by_row.keys()):
            src_row = unary_union(src_by_row[i])
            src_row = affinity.translate(src_row, xoff=dx_edge, yoff=dy_k)
            A_inter = src_row.intersection(tgt_row_union[i]).area
            A_union = src_row.union(tgt_row_union[i]).area
            total_area += A_inter
            total_union += A_union
        iou = (total_area / total_union) if total_union > 0 else 0.0
        candidates.append((total_area, iou, -abs(k), k))

    candidates.sort(reverse=True)
    best_k = candidates[0][-1]
    dy = dy_base + best_k * pitch

    # 4) refine dx via per-row medians (optional but helpful)
    xs_t = np.array([p.centroid.x for p in tgt])
    xs_s = np.array([p.centroid.x for p in src_scaled_polys])
    idx_t = np.round((ys_t - y0) / pitch).astype(int)
    idx_s_final = np.round((np.array([p.centroid.y for p in src_scaled_polys]) + dy - y0) / pitch).astype(int)
    rows_t = defaultdict(list); rows_s = defaultdict(list)
    for x, i in zip(xs_t, idx_t): rows_t[i].append(x)
    for x, i in zip(xs_s, idx_s_final): rows_s[i].append(x)
    diffs = [np.median(rows_t[r]) - np.median(rows_s[r]) for r in rows_t.keys() & rows_s.keys()]
    if diffs:
        dx_edge = float(np.median(diffs))

    # apply
    src_scaled_trans = [affinity.translate(g, xoff=dx_edge, yoff=dy) for g in src_scaled]
    parts = []
    for g in src_scaled_trans:
        b = g.boundary
        if b.geom_type == "LineString":
            parts.append(b)
        elif b.geom_type == "MultiLineString":
            parts.extend(b.geoms)
    out_mls = MultiLineString([ln.coords for ln in parts])

    # origin-based affine
    a = s; b = 0.0; d = 0.0; e = s
    xoff = dx_edge + (1.0 - s) * cx
    yoff = dy      + (1.0 - s) * cy

    params = dict(
        scale=float(s), dx=float(dx_edge), dy=float(dy),
        pitch=float(pitch), row_shift=int(best_k),
        origin_cx=float(cx), origin_cy=float(cy),
        affine=[a, b, d, e, xoff, yoff],
        notes="Row shift chosen by total per-row intersection area; tie-break by IoU then |k|."
    )
    return params, out_mls, aoi
