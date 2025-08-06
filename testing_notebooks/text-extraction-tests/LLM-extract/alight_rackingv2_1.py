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
    if not vals:
        return 0.0
    rounded = np.round(vals, 6)
    uniq, counts = np.unique(rounded, return_counts=True)
    mode_val = uniq[np.argmax(counts)]
    return float(mode_val)

def _row_pitch_from_target(polys: Iterable[Polygon]) -> float:
    """Calculate the most common row pitch (distance between adjacent row centerlines)"""
    if not polys:
        return 4.0  # fallback value
    
    poly_list = list(polys)
    if len(poly_list) < 2:
        return 4.0  # fallback value
    
    # Get all centroid Y coordinates
    y_coords = np.array([p.centroid.y for p in poly_list])
    
    # Group polygons into rows by clustering similar Y coordinates
    # Sort Y coordinates to identify distinct rows
    y_sorted = np.sort(y_coords)
    
    # Find gaps between Y coordinates to identify distinct rows
    # A gap larger than half the minimum spacing indicates a new row
    y_diffs = np.diff(y_sorted)
    if len(y_diffs) == 0:
        return 4.0
    
    # Use median of small differences as threshold for grouping within same row
    small_diffs = y_diffs[y_diffs < np.percentile(y_diffs, 75)]  # Use lower 75% of differences
    if len(small_diffs) > 0:
        within_row_threshold = np.median(small_diffs) * 2  # Allow some variation within row
    else:
        within_row_threshold = np.min(y_diffs) / 2
    
    # Group Y coordinates into rows
    row_centers = []
    current_row_ys = [y_sorted[0]]
    
    for i in range(1, len(y_sorted)):
        if y_sorted[i] - y_sorted[i-1] <= within_row_threshold:
            # Same row
            current_row_ys.append(y_sorted[i])
        else:
            # New row - record center of current row and start new one
            row_centers.append(np.mean(current_row_ys))
            current_row_ys = [y_sorted[i]]
    
    # Add the last row
    row_centers.append(np.mean(current_row_ys))
    
    if len(row_centers) < 2:
        return 4.0  # Need at least 2 rows to calculate pitch
    
    # Calculate distances between adjacent row centers
    row_centers = np.array(row_centers)
    row_pitches = np.diff(row_centers)
    
    if len(row_pitches) == 0:
        return 4.0
    
    # override row_piteched for now, use ydiff, cut off small numbers
    row_pitches = y_diffs[y_diffs > 0.0001]
    
    # Find the most common pitch (mode)
    # Round to avoid floating point precision issues
    rounded_pitches = np.round(row_pitches, 4)
    unique_pitches, counts = np.unique(rounded_pitches, return_counts=True)
    most_common_pitch = unique_pitches[np.argmax(counts)]
    most_common_pitch = np.median(row_pitches)
    
    return float(most_common_pitch)

def _extract_vertical_gaps(polys: Iterable[Polygon], x_position: float, tolerance: float = 0.5) -> np.ndarray:
    """Extract gaps along a vertical line by analyzing polygon intersections"""
    if not polys:
        return np.array([])
    
    # Create vertical line at specified x position
    poly_list = list(polys)
    if not poly_list:
        return np.array([])
    
    # Get overall Y bounds
    all_bounds = [p.bounds for p in poly_list]
    min_y = min(bounds[1] for bounds in all_bounds)
    max_y = max(bounds[3] for bounds in all_bounds)
    
    # Create vertical line spanning the full height
    from shapely.geometry import LineString
    vertical_line = LineString([(x_position, min_y - 1), (x_position, max_y + 1)])
    
    # Find intersections with polygons and collect Y ranges
    y_ranges = []
    for poly in poly_list:
        # Check if vertical line intersects polygon (with tolerance)
        buffered_poly = poly.buffer(tolerance)
        if vertical_line.intersects(buffered_poly):
            _, miny, _, maxy = poly.bounds
            y_ranges.append((miny, maxy))
    
    if len(y_ranges) < 2:
        return np.array([])
    
    # Sort by Y position
    y_ranges.sort(key=lambda x: x[0])
    
    # Calculate gaps between adjacent polygons
    gaps = []
    for i in range(len(y_ranges) - 1):
        gap_start = y_ranges[i][1]  # End of current polygon
        gap_end = y_ranges[i + 1][0]  # Start of next polygon
        gap_size = gap_end - gap_start
        if gap_size > 0.01:  # Only meaningful gaps
            gaps.append(gap_size)
    
    return np.array(gaps)

def _extract_row_spacings(polys: Iterable[Polygon]) -> np.ndarray:
    """Extract actual row-to-row spacings from polygon centroids"""
    if not polys:
        return np.array([])
    
    poly_list = list(polys)
    if len(poly_list) < 2:
        return np.array([])
    
    # Get Y coordinates and sort them
    y_coords = np.array([p.centroid.y for p in poly_list])
    y_sorted = np.sort(y_coords)
    
    # Calculate all consecutive differences
    spacings = np.diff(y_sorted)
    
    # Filter out tiny differences (same row) and keep meaningful spacings
    meaningful_spacings = spacings[spacings > 0.5]  # Adjust threshold as needed
    
    return meaningful_spacings

def _calculate_spacing_score(target_spacings: np.ndarray, source_spacings: np.ndarray) -> float:
    """Calculate a comprehensive score for how well spacing distributions match"""
    if len(target_spacings) == 0 or len(source_spacings) == 0:
        return float('inf')
    
    # Primary metric: mean spacing match
    target_mean = np.mean(target_spacings)
    source_mean = np.mean(source_spacings)
    mean_error = abs(target_mean - source_mean) / max(target_mean, 1e-6)
    
    # Secondary metric: distribution shape (standard deviation)
    target_std = np.std(target_spacings) if len(target_spacings) > 1 else 0
    source_std = np.std(source_spacings) if len(source_spacings) > 1 else 0
    std_error = abs(target_std - source_std) / max(target_mean, 1e-6)
    
    # Tertiary metric: range similarity
    target_range = np.max(target_spacings) - np.min(target_spacings) if len(target_spacings) > 1 else 0
    source_range = np.max(source_spacings) - np.min(source_spacings) if len(source_spacings) > 1 else 0
    range_error = abs(target_range - source_range) / max(target_mean, 1e-6)
    
    # Weighted combination
    total_score = 4.0 * mean_error + 1.0 * std_error + 0.5 * range_error
    return total_score

def _optimize_scale_by_spacings(
    target_polys: Iterable[Polygon],
    source_polys: Iterable[Polygon], 
    initial_scale: float,
    target_cx: float,
    target_cy: float
) -> float:
    """Fine-tune scale by directly matching row spacing distributions with multi-stage optimization"""
    
    poly_list_tgt = list(target_polys)
    poly_list_src = list(source_polys)
    
    if len(poly_list_tgt) < 3 or len(poly_list_src) < 3:
        return initial_scale
    
    # Extract target row spacings once
    target_spacings = _extract_row_spacings(poly_list_tgt)
    if len(target_spacings) == 0:
        return initial_scale
    
    def evaluate_scale(test_scale: float) -> float:
        """Helper function to evaluate a scale factor"""
        scaled_src_polys = [affinity.scale(p, xfact=test_scale, yfact=test_scale, 
                                         origin=(target_cx, target_cy)) for p in poly_list_src]
        source_spacings = _extract_row_spacings(scaled_src_polys)
        return _calculate_spacing_score(target_spacings, source_spacings)
    
    best_scale = initial_scale
    best_score = evaluate_scale(initial_scale)
    
    # Stage 1: Coarse search with wide range
    print("Stage 1: Coarse optimization...")
    scale_range_coarse = np.linspace(initial_scale * 0.7, initial_scale * 1.3, 61)
    
    for test_scale in scale_range_coarse:
        score = evaluate_scale(test_scale)
        if score < best_score:
            best_score = score
            best_scale = test_scale
    
    # Stage 2: Fine search around best candidate
    print(f"Stage 2: Fine optimization around {best_scale:.4f}...")
    search_window = abs(best_scale * 0.05)  # ±5% around best candidate
    scale_range_fine = np.linspace(best_scale - search_window, best_scale + search_window, 101)
    
    for test_scale in scale_range_fine:
        if test_scale <= 0:  # Skip invalid scales
            continue
        score = evaluate_scale(test_scale)
        if score < best_score:
            best_score = score
            best_scale = test_scale
    
    # Stage 3: Ultra-fine search for precision
    print(f"Stage 3: Ultra-fine optimization around {best_scale:.4f}...")
    search_window_ultra = abs(best_scale * 0.01)  # ±1% around best candidate
    scale_range_ultra = np.linspace(best_scale - search_window_ultra, best_scale + search_window_ultra, 201)
    
    for test_scale in scale_range_ultra:
        if test_scale <= 0:  # Skip invalid scales
            continue
        score = evaluate_scale(test_scale)
        if score < best_score:
            best_score = score
            best_scale = test_scale
    
    # Stage 4: Golden section search for final precision
    print("Stage 4: Golden section search for final precision...")
    
    # Golden ratio for optimal search
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi
    
    # Set search bounds around current best
    search_width = abs(best_scale * 0.005)  # ±0.5% 
    tol = abs(best_scale * 0.0001)  # 0.01% tolerance
    
    a = max(best_scale - search_width, initial_scale * 0.1)  # Lower bound
    b = best_scale + search_width  # Upper bound
    
    # Initial points
    c = a + resphi * (b - a)
    d = a + (1 - resphi) * (b - a)
    fc = evaluate_scale(c)
    fd = evaluate_scale(d)
    
    iteration = 0
    while abs(b - a) > tol and iteration < 20:
        iteration += 1
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + resphi * (b - a)
            fc = evaluate_scale(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + (1 - resphi) * (b - a)
            fd = evaluate_scale(d)
    
    # Final best scale
    final_scale = (a + b) / 2
    final_score = evaluate_scale(final_scale)
    
    if final_score < best_score:
        best_scale = final_scale
        best_score = final_score
    
    print(f"Scale optimization complete: {initial_scale:.6f} -> {best_scale:.6f} (score: {best_score:.6f})")
    return best_scale

def align_by_pitch_edge_rows(
    combiner_lines: MultiLineString | LineString,
    target_polys: Iterable[Polygon] | MultiPolygon,
    source_geoms: Iterable[base.BaseGeometry],
    row_shift_range: int = 3,
) -> Tuple[Dict[str, float], MultiLineString, base.BaseGeometry]:
    """
    1) scale source so its row pitch matches target row pitch (most common)
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

    # --- 1) Scale from row pitch (first pass)
    pitch_tgt = _row_pitch_from_target(tgt)
    pitch_src = _row_pitch_from_target(src_polys)
    print("target pitch, source pitch", pitch_tgt, pitch_src)
    s_initial = pitch_tgt / pitch_src if pitch_src > 0 else 1.0
    
    print("initial pitch-based scale", s_initial)
    
    # Get target centroid for scaling operations
    tgt_union = unary_union(tgt)
    target_cx, target_cy = tgt_union.centroid.x, tgt_union.centroid.y
    
    # --- 1.5) Fine-tune scale using row spacing analysis
    print("Optimizing scale using row spacing analysis...")
    s_optimized = _optimize_scale_by_spacings(tgt, src_polys, s_initial, target_cx, target_cy)
    
    # Use the optimized scale
    s = s_optimized
    src_scaled = [affinity.scale(g, xfact=s, yfact=s, origin=(target_cx, target_cy)) for g in src_orig]

    # polygonized version of scaled source for bbox/pitch ops
    src_scaled_polys = []
    for g in src_scaled:
        if g.geom_type in ("Polygon", "MultiPolygon"):
            src_scaled_polys.append(g)
        else:
            src_scaled_polys.extend(_polygonize_lines(g))

    # --- 2) Snap edges (left + bottom)
    # Note: tgt_union already calculated above
    src_union = unary_union(src_scaled_polys)
    minx_t, miny_t, _, _ = tgt_union.bounds
    minx_s, miny_s, _, _ = src_union.bounds
    dx_edge = minx_t - minx_s
    dy_base = miny_t - miny_s

    # --- 3) Row shift by integer pitch steps
    # Use target pitch for row alignment (source should now match after scaling)
    pitch = pitch_tgt
    print("using target pitch for alignment", pitch)
    ys_t = np.array([p.centroid.y for p in tgt])
    ys_s = np.array([p.centroid.y for p in src_scaled_polys])
    y0 = float(np.median(ys_t))

    idx_t = np.round((ys_t - y0) / pitch).astype(int)
    from collections import Counter
    ct = Counter(idx_t)
    best_k, best_overlap = 0, -1
    for k in range(-row_shift_range, row_shift_range + 1):
        print("k", k)
        idx_s = np.round((ys_s + dy_base + k * pitch - y0) / pitch).astype(int)
        cs = Counter(idx_s)
        overlap = sum(min(ct[r], cs.get(r, 0)) for r in ct.keys())
        # Area-based overlap calculation
        """   overlap = 0.0
        for t_poly in tgt:
            for s_poly in src_scaled_polys:
                inter = t_poly.intersection(s_poly)
                if not inter.is_empty:
                    overlap += inter.area"""
        print("overlap", overlap)
        if overlap > best_overlap:
            best_overlap, best_k = overlap, k
    print("best_overlap", best_overlap)
    print("best_k", best_k)
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

    # --- build a valid origin-based affine for shapely.affine_transform
    # we already have: s (scale), dx, dy, and (target_cx, target_cy) from the target centroid used for scaling

    a = s
    b = 0.0
    d = 0.0
    e = s
    xoff = dx_edge + (1.0 - s) * target_cx
    yoff = dy + (1.0 - s) * target_cy

    affine_origin_based = [a, b, d, e, xoff, yoff]

    params = dict(
        scale=float(s),
        dx=float(dx_edge),
        dy=float(dy),
        pitch=float(pitch),
        row_shift=int(best_k),
        origin_cx=float(target_cx),
        origin_cy=float(target_cy),
        affine=affine_origin_based,      # <<-- add this
    )

    return params, out_mls, aoi
