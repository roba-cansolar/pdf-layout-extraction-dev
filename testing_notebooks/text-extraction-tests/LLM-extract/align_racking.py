"""
Old method, but generalized for MultiLineString IO.

- Accepts Shapely:
    combiner_lines      : MultiLineString (AOI scaffold)
    target_polys        : list[Polygon] | MultiPolygon | GeometryCollection
    source_lines        : MultiLineString (or LineString)

- Returns:
    params              : dict(scale, dx, dy, cost, row_pitch, buffer_dist)
    source_lines_xform  : MultiLineString (transformed)
    aoi_buffer_poly     : Polygon/MultiPolygon used for fitting

Only translation + uniform scale (no rotation, aspect preserved).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, Any

import math
import numpy as np
from shapely.geometry import (
    LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection, base
)
from shapely.ops import unary_union, polygonize
from shapely import affinity

try:
    from scipy.spatial import cKDTree as KDTree  # optional speedup
    _HAS_SCIPY = True
except Exception:
    KDTree = None
    _HAS_SCIPY = False


# -------------------- Options --------------------

@dataclass
class Options:
    # AOI buffer expressed in multiples of the estimated row pitch
    aoi_buffer_in_row_pitch: float = 1.5
    # Drop target polys with nearest neighbor farther than this (in row-pitch units)
    neighbor_max_in_row_pitch: float = 1.2

    # Convert source lines to area for the XOR objective.
    # If >0, this is an ABSOLUTE buffer in map units.
    # If None, we use 0.15 * pitch as the half-width.
    source_line_buffer_abs: float | None = None

    # Optimization windows/steps (coarse→fine two-pass)
    range_dxy: float = 3.0
    range_s: float = 0.02
    coarse_step_xy: float = 0.5
    coarse_step_s: float = 0.002
    fine_step_xy: float = 0.25
    fine_step_s: float = 0.001
    iters: int = 4                     # passes per stage

    # Optional simplification used only during optimization
    simplify_tolerance: float | None = 0.02


# -------------------- Utilities --------------------

def _as_polygon_iter(geoms) -> Iterable[Polygon]:
    """Yield polygons from many possible containers."""
    if geoms is None:
        return []
    if isinstance(geoms, (Polygon,)):
        return [geoms]
    if isinstance(geoms, (MultiPolygon, GeometryCollection)):
        return [g for g in geoms.geoms if isinstance(g, Polygon)]
    # assume iterable
    return [g for g in geoms if isinstance(g, Polygon)]

def _bbox_params(g: base.BaseGeometry):
    minx, miny, maxx, maxy = g.bounds
    w, h = (maxx - minx), (maxy - miny)
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    return w, h, (cx, cy)

def _estimate_row_pitch(polys: Iterable[Polygon]) -> float:
    """Median nearest-neighbor Δy among polygons in roughly the same x (robust to outliers)."""
    cxs = []
    cys = []
    for p in polys:
        c = p.centroid
        cxs.append(c.x)
        cys.append(c.y)
    cxs = np.asarray(cxs); cys = np.asarray(cys)
    if len(cxs) < 2:
        return 4.0

    window = 15.0
    dys = []
    for i in range(len(cxs)):
        mask = np.abs(cxs - cxs[i]) < window
        if np.count_nonzero(mask) <= 1:
            continue
        dy = np.min(np.abs(cys[mask] - cys[i]) + 1e-9)
        if dy > 1e-3:
            dys.append(dy)
    return float(np.median(dys)) if dys else 4.0

def _nearest_neighbor_dists(points: np.ndarray) -> np.ndarray:
    """Euclidean NN distances (exclude self). Uses KDTree if available."""
    n = len(points)
    if n < 2:
        return np.full(n, np.inf)
    if _HAS_SCIPY:
        tree = KDTree(points)
        d, _ = tree.query(points, k=2)
        return d[:, 1]
    # grid-hash fallback
    bin_size = 10.0
    bins = {}
    for i, (x, y) in enumerate(points):
        key = (int(x // bin_size), int(y // bin_size))
        bins.setdefault(key, []).append(i)
    nn = np.full(n, np.inf)
    for i, (xi, yi) in enumerate(points):
        bx, by = int(xi // bin_size), int(yi // bin_size)
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand.extend(bins.get((bx + dx, by + dy), []))
        best = np.inf
        for j in cand:
            if i == j: 
                continue
            xj, yj = points[j]
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            if d2 < best: 
                best = d2
        nn[i] = math.sqrt(best) if best < np.inf else np.inf
    return nn

def _line_to_area(lines: base.BaseGeometry, buf: float) -> base.BaseGeometry:
    """Buffer lines into polygons (idempotent for LineString/MultiLineString)."""
    # small positive buffer; dissolve to single geometry
    return unary_union(lines).buffer(buf)

def _coord_descent(cost_fn, s0, dx0, dy0, *,
                   iters: int, rng_xy: float,
                   step_xy: float, rng_s: float, step_s: float):
    """Deterministic coordinate descent: scan dx, then dy, then s; repeat."""
    best = (s0, dx0, dy0)
    best_c = cost_fn(*best)

    def scan(which, s, dx, dy):
        nonlocal best, best_c
        if which == "dx":
            vals = np.arange(dx - rng_xy, dx + rng_xy + step_xy, step_xy)
            for v in vals:
                c = cost_fn(s, v, dy)
                if c < best_c:
                    best, best_c = (s, v, dy), c
        elif which == "dy":
            vals = np.arange(dy - rng_xy, dy + rng_xy + step_xy, step_xy)
            for v in vals:
                c = cost_fn(s, dx, v)
                if c < best_c:
                    best, best_c = (s, dx, v), c
        else:  # "s"
            lo = max(0.95, s - rng_s)
            hi = min(1.05, s + rng_s)
            vals = np.arange(lo, hi + step_s, step_s)
            for v in vals:
                c = cost_fn(v, dx, dy)
                if c < best_c:
                    best, best_c = (v, dx, dy), c

    for _ in range(iters):
        scan("dx", *best)
        scan("dy", *best)
        scan("s",  *best)
    return best, best_c


# -------------------- Main function --------------------

def align_racking_affine_lines(
    combiner_lines: MultiLineString | LineString,
    target_polys: Iterable[Polygon] | MultiPolygon | GeometryCollection,
    source_lines: MultiLineString | LineString,
    opts: Options = Options(),
) -> Tuple[Dict[str, float], MultiLineString, base.BaseGeometry]:
    """
    Old robust method:
      1) buffer AOI (in row pitches) built from combiner lines (polygonize->buffer)
      2) filter target polygons to drop isolated strays
      3) objective: XOR area between target union and (buffered source lines),
         inside AOI buffer; normalize by AOI area
      4) coarse→fine coordinate descent over {dx, dy, scale} (no rotation)
    """

    # 1) AOI polygon from lines
    aoi_area = unary_union(list(polygonize(combiner_lines)))  # may be multi-polygons

    # 2) Prep target polys
    target_list = list(_as_polygon_iter(target_polys))
    if len(target_list) == 0:
        raise ValueError("No target polygons supplied.")

    # Pitch & AOI buffer
    row_pitch = _estimate_row_pitch(target_list)
    buffer_dist = opts.aoi_buffer_in_row_pitch * row_pitch
    aoi_buf = aoi_area.buffer(buffer_dist)

    # Filter targets (within buffer & not isolated)
    centers = np.array([[p.centroid.x, p.centroid.y] for p in target_list])
    nn = _nearest_neighbor_dists(centers)
    keep = []
    for i, p in enumerate(target_list):
        if not p.intersects(aoi_buf):
            continue
        if np.isfinite(nn[i]) and nn[i] <= opts.neighbor_max_in_row_pitch * row_pitch + 1e-6:
            keep.append(p)
    if not keep:
        keep = [p for p in target_list if p.intersects(aoi_buf)]

    # Optional simplify (optimization only)
    if opts.simplify_tolerance and opts.simplify_tolerance > 0:
        target_opt = [p.simplify(opts.simplify_tolerance, preserve_topology=True) for p in keep]
    else:
        target_opt = keep

    target_union = unary_union(target_opt).buffer(0)
    At = target_union.area
    Ao = aoi_buf.area

    # 3) Source lines -> area for cost, choose line buffer
    src_line_buf = (opts.source_line_buffer_abs
                    if (opts.source_line_buffer_abs is not None)
                    else max(0.05 * row_pitch, 0.15 * row_pitch))
    # NB: buffer after each tentative transform (for accuracy)

    # 4) Initial guess from bbox fit inside AOI
    #    For bbox we need areas: create a quick once-off source area
    source_area_0 = _line_to_area(source_lines, src_line_buf)
    src_clip = source_area_0.intersection(aoi_buf)
    tgt_clip = target_union.intersection(aoi_buf)

    ws, hs, (cx_s, cy_s) = _bbox_params(src_clip)
    wt, ht, (cx_t, cy_t) = _bbox_params(tgt_clip)
    s0 = min(wt / ws, ht / hs) if ws * hs > 0 else 1.0
    dx0, dy0 = cx_t - cx_s, cy_t - cy_s

    # 5) Cost function (scale around source centroid computed from lines)
    src_center = unary_union(source_lines).centroid

    def transform_lines(scale, dx, dy):
        g = affinity.scale(source_lines, xfact=scale, yfact=scale, origin=(src_center.x, src_center.y))
        g = affinity.translate(g, xoff=dx, yoff=dy)
        return g

    def cost(scale, dx, dy):
        # transform lines → buffer to area → clip to AOI → XOR with target
        g_lines = transform_lines(scale, dx, dy)
        g_area = _line_to_area(g_lines, src_line_buf).intersection(aoi_buf)
        inter = g_area.intersection(target_union).area
        xor_area = At + g_area.area - 2 * inter
        return xor_area / Ao

    # 6) Two-pass coordinate descent (coarse -> fine)
    # coarse
    (s1, dx1, dy1), _ = _coord_descent(
        cost, s0, dx0, dy0,
        iters=opts.iters,
        rng_xy=opts.range_dxy, step_xy=opts.coarse_step_xy,
        rng_s=opts.range_s,  step_s=opts.coarse_step_s
    )
    # fine
    (s_best, dx_best, dy_best), c_best = _coord_descent(
        cost, s1, dx1, dy1,
        iters=opts.iters,
        rng_xy=opts.range_dxy, step_xy=opts.fine_step_xy,
        rng_s=opts.range_s,  step_s=opts.fine_step_s
    )

    # 7) Apply final transform to source lines (return as MultiLineString)
    g_final = transform_lines(s_best, dx_best, dy_best)
    if isinstance(g_final, LineString):
        out_lines = MultiLineString([g_final.coords])
    elif isinstance(g_final, MultiLineString):
        out_lines = g_final
    else:
        # If the geometry type somehow changes (e.g., collection), coerce back to MultiLineString
        parts = []
        for geom in getattr(g_final, "geoms", []):
            if isinstance(geom, LineString):
                parts.append(geom)
        out_lines = MultiLineString([ln.coords for ln in parts]) if parts else g_final

    params = dict(
        scale=float(s_best),
        dx=float(dx_best),
        dy=float(dy_best),
        cost=float(c_best),
        row_pitch=float(row_pitch),
        buffer_dist=float(buffer_dist),
        line_buffer=float(src_line_buf),
    )
    return params, out_lines, aoi_buf
