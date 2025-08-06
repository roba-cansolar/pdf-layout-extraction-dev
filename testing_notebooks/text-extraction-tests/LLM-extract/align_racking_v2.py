from typing import Iterable, Tuple, Dict
import numpy as np
from shapely.geometry import (
    LineString, MultiLineString, Polygon, MultiPolygon, base
)
from shapely.ops import unary_union, polygonize
from shapely import affinity

def _polygonize_lines(geom):
    # Accept LineString or MultiLineString and polygonize to faces
    return list(polygonize(geom))

def _short_side_width(polys: Iterable[Polygon]) -> float:
    vals = []
    for p in polys:
        minx, miny, maxx, maxy = p.bounds
        vals.append(min(maxx - minx, maxy - miny))
    return float(np.median(vals)) if vals else 0.0

def _row_pitch_from_target(polys: Iterable[Polygon]) -> float:
    ys = np.array([p.centroid.y for p in polys])
    xs = np.array([p.centroid.x for p in polys])
    if len(ys) < 2:
        return 4.0
    window = 15.0
    dys = []
    for i in range(len(ys)):
        mask = np.abs(xs - xs[i]) < window
        if np.count_nonzero(mask) <= 1:
            continue
        dy = np.min(np.abs(ys[mask] - ys[i]) + 1e-9)
        if dy > 1e-3:
            dys.append(dy)
    return float(np.median(dys)) if dys else 4.0

def align_by_width_edge_rows(
    combiner_lines: MultiLineString | LineString,
    target_polys: Iterable[Polygon] | MultiPolygon,
    source_geoms: Iterable[base.BaseGeometry],
    row_shift_range: int = 3,
) -> Tuple[Dict[str, float], MultiLineString, base.BaseGeometry]:
    """
    1) scale source so its rack width (short side) matches target width (median)
    2) snap scaled source to target left & bottom edges (bbox minx, miny)
    3) adjust dy by integer multiples of target pitch in [-row_shift_range, +row_shift_range]
    Returns (params, transformed MultiLineString, AOI polygon)
    """
    # AOI from combiner lines
    aoi = unary_union(list(polygonize(combiner_lines)))

    # Filter target to AOI; collect polygons
    tgt = [p for p in (target_polys.geoms if isinstance(target_polys, MultiPolygon) else target_polys)
           if p.intersects(aoi)]

    # Normalize source: polygons if already, else polygonize lines
    src_polys = []
    src_orig = list(source_geoms)
    for g in src_orig:
        if g.geom_type in ("Polygon", "MultiPolygon"):
            src_polys.append(g)
        elif g.geom_type in ("LineString", "MultiLineString"):
            src_polys.extend(_polygonize_lines(g))
        else:
            # last resort: try boundary->polygonize
            src_polys.extend(_polygonize_lines(g.boundary))

    # --- 1) Scale from short-side width
    wt = _short_side_width(tgt)
    ws = _short_side_width(src_polys)
    s  = wt / ws if ws > 0 else 1.0

    # scale about centroid of all source geometry
    src_union_all = unary_union(src_orig)
    cx, cy = src_union_all.centroid.x, src_union_all.centroid.y
    src_scaled = [affinity.scale(g, xfact=s, yfact=s, origin=(cx, cy)) for g in src_orig]

    # polygonized version of scaled source for bbox/pitch ops
    src_scaled_polys = []
    for g in src_scaled:
        if g.geom_type in ("Polygon", "MultiPolygon"):
            src_scaled_polys.append(g)
        else:
            src_scaled_polys.extend(_polygonize_lines(g))

    # --- 2) Snap edges (left + bottom)
    tgt_union = unary_union(tgt)
    src_union = unary_union(src_scaled_polys)
    minx_t, miny_t, _, _ = tgt_union.bounds
    minx_s, miny_s, _, _ = src_union.bounds
    dx_edge = minx_t - minx_s
    dy_base = miny_t - miny_s

    # --- 3) Row shift by integer pitch steps
    pitch = _row_pitch_from_target(tgt)
    ys_t = np.array([p.centroid.y for p in tgt])
    ys_s = np.array([p.centroid.y for p in src_scaled_polys])
    y0 = float(np.median(ys_t))

    idx_t = np.round((ys_t - y0) / pitch).astype(int)
    from collections import Counter
    ct = Counter(idx_t)
    best_k, best_overlap = 0, -1
    for k in range(-row_shift_range, row_shift_range + 1):
        idx_s = np.round((ys_s + dy_base + k * pitch - y0) / pitch).astype(int)
        cs = Counter(idx_s)
        overlap = sum(min(ct[r], cs.get(r, 0)) for r in ct.keys())
        if overlap > best_overlap:
            best_overlap, best_k = overlap, k

    dy = dy_base + best_k * pitch

    # Apply final transform to original source geoms and return as MultiLineString
    src_scaled_trans = [affinity.translate(g, xoff=dx_edge, yoff=dy) for g in src_scaled]
    line_parts = []
    for g in src_scaled_trans:
        gb = g.boundary
        if gb.geom_type == "LineString":
            line_parts.append(gb)
        elif gb.geom_type == "MultiLineString":
            line_parts.extend(gb.geoms)
    out_mls = MultiLineString([ln.coords for ln in line_parts])

    params = dict(scale=float(s), dx=float(dx_edge), dy=float(dy),
                  pitch=float(pitch), row_shift=int(best_k))
    return params, out_mls, aoi
