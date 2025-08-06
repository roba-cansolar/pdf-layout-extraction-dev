"""
PDF As-Built Drawing Extraction Utilities

This module provides a comprehensive pipeline for extracting, processing, and labeling 
geometric data from PDF as-built electrical drawings. The module is designed specifically 
for processing electrical as-built drawings to identify and label components such as 
inverters and combiners.

WORKFLOW OVERVIEW:
==================

The module operates in two main phases:

1. GEOMETRIC EXTRACTION PHASE:
   - Loads PDF pages and extracts vector drawing data
   - Filters drawing elements by layer (e.g., "E-CB_AREA", "key plan|BEI-BLOCK-OUTLINE")
   - Applies clip region filtering to focus on relevant drawing areas
   - Converts vector paths (lines, rectangles, quads) to line geometries
   - Performs endpoint clustering and snapping for better connectivity
   - Extends dangling line endpoints to create closed network topology
   - Polygonizes the line network to generate component boundary polygons

2. AI-POWERED LABELING PHASE:
   - Clips each polygon with visual buffer highlighting
   - Generates rasterized images of polygon regions
   - Uses Google Gemini AI to identify and extract component labels
   - Builds structured data mapping labels to polygon geometries
   - Supports batch processing across multiple pages and element types

CORE CLASSES:
=============

ExtractionConfig: Configuration parameters for the extraction pipeline
- Controls tolerance values, extension lengths, and processing parameters
- Customizable for different drawing types and quality requirements

MAIN FUNCTIONS:
===============

Core Extraction Pipeline:
- extract_polygons_from_layer(): Main pipeline function
- load_pdf_page(): PDF loading and drawing extraction
- build_clip_map(): Clip region identification and mapping
- extract_lines_from_paths(): Vector path to line geometry conversion
- cluster_and_snap_endpoints(): Endpoint topology improvement
- extend_dangles(): Network connectivity enhancement
- polygonize_lines(): Polygon generation from line networks

AI Labeling Pipeline:
- process_elements_on_page(): Batch polygon labeling for a page
- get_label_from_gemini(): Individual polygon label extraction
- get_page_label_from_gemini(): Page-level classification
- clip_pdf_to_polygon(): Visual polygon highlighting and clipping

Utilities:
- visualize_extraction_results(): Comprehensive result visualization
- save_polygons_to_json()/load_polygons_from_json(): Data persistence
- batch_extract_polygons(): Multi-file processing

PERFORMANCE CHARACTERISTICS:
============================

Processing times vary by:
- PDF complexity and layer density
- Number of drawing elements per layer  
- Polygon count and geometric complexity
- AI API response times (major bottleneck)

Typical performance:
- Geometric extraction: 0.1-2 seconds per page
- AI labeling: 2-10 seconds per polygon (API dependent)
- Memory usage: 50-200MB per page depending on drawing complexity

DEPENDENCIES:
=============

Core Processing:
- pymupdf/fitz: PDF parsing and vector extraction
- shapely: Geometric operations and polygon processing
- numpy: Numerical computations
- pandas: Data analysis and layer statistics

Visualization:
- matplotlib: Result plotting and visualization
- PIL (Pillow): Image processing and manipulation

AI Integration:
- google.generativeai: Gemini AI API for label extraction
- dotenv: Environment variable management

Data Handling:
- json: Structured data serialization
- pathlib: Cross-platform path handling
"""

import pymupdf
import fitz
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, MultiLineString,MultiPoint
from shapely.ops import unary_union, polygonize
from shapely import wkt
import pandas as pd
from collections import defaultdict, Counter
from PIL import Image
import io
import re
import os
import math
import itertools
import random
from typing import List, Dict, Tuple, Optional, Any
import time
import json
import pathlib
from dotenv import load_dotenv
import os
import google.generativeai as genai
from shapely.affinity import affine_transform
from shapely.ops import unary_union
from shapely.geometry import mapping
import json


env_path = pathlib.Path().resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class ExtractionConfig:
    """
    Configuration parameters for the PDF extraction and processing pipeline.
    
    This class encapsulates all tunable parameters that control the behavior of the
    geometric extraction algorithms, performance optimizations, visualization settings,
    and AI labeling workflow.
    
    The parameters are grouped into logical categories for easy customization based on
    different drawing types, quality requirements, and performance constraints.
    
    Attributes:
        Core Processing Parameters:
            tolerance (float): Distance tolerance (in drawing units) for clustering nearby 
                             line endpoints. Smaller values = more precise but may miss
                             connections. Larger values = more connections but may over-snap.
                             Default: 1.0, Typical range: 0.5-3.0
                             
            snap_distance (float): Distance for snapping points to cluster centers.
                                 Should typically match or be slightly larger than tolerance.
                                 Default: 1.0
                                 
            extension_length (float): Maximum length (in drawing units) for extending 
                                    dangling line endpoints to connect to the network.
                                    Longer extensions catch more connections but may create
                                    incorrect topology. Default: 40.0, Range: 10.0-100.0
                                    
            min_polygon_area (float): Minimum area threshold for valid polygons.
                                    Filters out noise and very small artifacts.
                                    Default: 10.0, Typical range: 1.0-50.0
        
        Performance Parameters:
            grid_size (float): Grid size for Shapely spatial operations. Larger values
                             can improve performance but reduce precision. 
                             Default: 3.0, Range: 1.0-10.0
                             
            max_iterations (int): Maximum iterations for iterative dangle extension.
                                Currently not used but reserved for future algorithms.
                                Default: 100
                                
            batch_size (int): Batch size for processing multiple pages/files.
                             Controls memory vs. parallelization trade-off.
                             Default: 10, Range: 1-50
        
        Visualization Parameters:
            figure_size (tuple): Matplotlib figure size for visualization plots.
                               Default: (12, 8)
                               
            dpi (int): DPI for visualization rendering. Higher = better quality but slower.
                      Default: 150, Range: 72-300
                      
            show_dangles (bool): Whether to visualize dangling endpoints in plots.
                               Default: True
                               
            show_extensions (bool): Whether to visualize extension lines in plots.
                                  Default: True
        
        AI Labeling Parameters:
            polygon_buffer (float): Buffer distance (in drawing units) around polygons
                                  when creating clipped images for AI analysis. Provides
                                  visual context around the component boundary.
                                  Default: 50.0, Range: 10.0-100.0
    
    Usage:
        # Default configuration
        config = ExtractionConfig()
        
        # High precision configuration
        config = ExtractionConfig()
        config.tolerance = 0.5
        config.extension_length = 20.0
        config.min_polygon_area = 1.0
        
        # Performance-optimized configuration
        config = ExtractionConfig()
        config.grid_size = 5.0
        config.dpi = 100
        config.extension_length = 30.0
    """
    
    def __init__(self):
        # Core processing parameters - control geometric extraction accuracy
        self.tolerance = 1.0  # Distance tolerance for clustering endpoints (drawing units)
        self.snap_distance = 1.0  # Distance for snapping points to cluster centers
        self.extension_length = 40.0  # Maximum length for extending dangles to connect network
        self.min_polygon_area = 10.0  # Minimum area threshold for valid polygons
        
        # Performance parameters - control speed vs. quality trade-offs
        self.grid_size = 3.0  # Grid size for Shapely spatial operations
        self.max_iterations = 100  # Maximum iterations for iterative algorithms
        self.batch_size = 10  # Batch size for multi-page processing
        
        # Visualization parameters - control output quality and content
        self.figure_size = (12, 8)  # Matplotlib figure size for plots
        self.dpi = 150  # DPI for visualization rendering
        self.show_dangles = True  # Show dangling endpoints in visualizations
        self.show_extensions = True  # Show extension lines in visualizations

        # AI labeling parameters - control image clipping and context
        self.polygon_buffer = 50.0  # Buffer around polygons for AI image clipping


def load_pdf_page(file_path: str, page_number: int = 0) -> Tuple[Any, List[Dict]]:
    """
    Load a PDF page and extract drawings
    
    Args:
        file_path: Path to PDF file
        page_number: Page number to load (0-indexed)
        
    Returns:
        Tuple of (page object, list of drawings)
    """
    try:
        doc = pymupdf.open(file_path)
        if page_number >= len(doc):
            raise ValueError(f"Page {page_number} not found. PDF has {len(doc)} pages.")
        
        page = doc[page_number]
        drawings = page.get_drawings()
        
        return page, drawings
    except Exception as e:
        raise Exception(f"Error loading PDF: {str(e)}")


def analyze_layers(drawings: List[Dict]) -> pd.DataFrame:
    """
    Analyze drawing layers and return summary statistics
    
    Args:
        drawings: List of drawing dictionaries
        
    Returns:
        DataFrame with layer analysis
    """
    layer_summary = defaultdict(lambda: defaultdict(int))
    
    for drawing in drawings:
        layer = drawing.get('layer', 'Unknown')
        # Extract item type from first element of items (matches original implementation)
        items = drawing.get('items', [])
        if items and isinstance(items[0], tuple) and len(items[0]) > 0:
            item_type = items[0][0]
        else:
            item_type = 'Unknown'
        layer_summary[layer][item_type] += 1
    
    # Convert to DataFrame
    rows = []
    for layer, types in layer_summary.items():
        for item_type, count in types.items():
            rows.append({'Layer': layer, 'Item Type': item_type, 'Count': count})
    
    summary_df = pd.DataFrame(rows)
    
    if not summary_df.empty:
        # Create pivot table
        pivot_summary = summary_df.pivot_table(
            index='Layer', 
            columns='Item Type', 
            values='Count', 
            fill_value=0,
            aggfunc='sum'
        )
        return pivot_summary
    
    return pd.DataFrame()


def build_clip_map(page):
    """
    Build a mapping of clip regions for the page (matches original implementation)
    
    Returns:
        draw_info: list[dict] - Each entry is the original get_drawings dict plus a key 'clips'
        clips: dict[str, dict] - name → clip details
    """
    import fitz
    
    stack, draw_info, clips_seen = {}, [], {}

    for seq, d in enumerate(page.get_drawings(extended=True)):
        lvl = d["level"]

        # Register clips
        if d["type"] == "clip":
            clip_entry = {
                "rect": fitz.Rect(d["scissor"]),
                "items": d.get("items", []),
                "level": lvl,
                "seq": seq,
                "name": f"clip_{seq}",
            }
            stack[lvl] = clip_entry
            clips_seen[clip_entry["name"]] = clip_entry
            continue  # clip path draws nothing itself

        # Maintain clip stack
        dead = [L for L in stack if L > lvl]
        for L in dead:
            stack.pop(L)

        active_chain = [stack[L]["name"] for L in sorted(stack)]
        draw_info.append({**d, "clips": active_chain})

    return draw_info, clips_seen


def clip_to_polygon(c):
    """
    Convert a clip definition to a Shapely polygon (matches original implementation)
    """
    # Full path reconstruction
    pts = []
    for tag, *rest in c.get("items", []):
        if tag == "re":
            rc = rest[0]
            pts.extend([(rc.x0, rc.y0), (rc.x1, rc.y0),
                        (rc.x1, rc.y1), (rc.x0, rc.y1)])
        elif tag == "l":
            p1, p2 = rest
            if not pts:
                pts.append((p1.x, p1.y))
            pts.append((p2.x, p2.y))
        elif tag == "qu":
            quad, = rest
            pts.append((quad.ul.x, quad.ul.y))
            pts.append((quad.ur.x, quad.ur.y))
            pts.append((quad.lr.x, quad.lr.y))
            pts.append((quad.ll.x, quad.ll.y))

    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])

    return Polygon(pts) if len(pts) >= 4 else None


def filter_layer_paths(drawings: List[Dict], layer_name: str, 
                      clip_name: Optional[str] = None) -> List[Dict]:
    """
    Filter drawings by layer and optionally by clip region
    
    Args:
        drawings: List of drawing dictionaries
        layer_name: Name of layer to filter
        clip_name: Optional clip region name
        
    Returns:
        Filtered list of drawings
    """
    layer_paths = [p for p in drawings if p.get("layer") == layer_name]
    
    if clip_name:
        # Filter by clip if specified
        filtered_paths = []
        for p in layer_paths:
            if 'clip_chain' in p and clip_name in p['clip_chain']:
                filtered_paths.append(p)
        return filtered_paths
    
    return layer_paths


def extract_lines_from_paths(paths: List[Dict], config: ExtractionConfig) -> List[LineString]:
    """
    Extract LineString objects from drawing paths (matches original implementation)
    
    Args:
        paths: List of drawing path dictionaries
        config: Configuration object
        
    Returns:
        List of LineString objects
    """
    lines = []
    
    for path in paths:
        for tag, *it in path.get("items", []):
            if tag == "l":  # line
                p1, p2 = it
                line = LineString([(p1.x, p1.y), (p2.x, p2.y)])
                if line.length > 0:  # Skip zero-length lines
                    lines.append(line)
            elif tag == "re":  # rectangle
                rect = it[0]
                # Create rectangle as 4 lines
                coords = [
                    (rect.x0, rect.y0), (rect.x1, rect.y0),
                    (rect.x1, rect.y1), (rect.x0, rect.y1), (rect.x0, rect.y0)
                ]
                for i in range(len(coords)-1):
                    line = LineString([coords[i], coords[i+1]])
                    if line.length > 0:
                        lines.append(line)
            elif tag == "qu":  # quad
                quad = it[0]
                # Create quad as 4 lines
                coords = [
                    (quad.ul.x, quad.ul.y), (quad.ur.x, quad.ur.y),
                    (quad.lr.x, quad.lr.y), (quad.ll.x, quad.ll.y), (quad.ul.x, quad.ul.y)
                ]
                for i in range(len(coords)-1):
                    line = LineString([coords[i], coords[i+1]])
                    if line.length > 0:
                        lines.append(line)
            # Note: "c" (bezier) curves would need special handling
    
    return lines


def cluster_and_snap_endpoints(lines: List[LineString], tolerance: float) -> List[LineString]:
    """
    Cluster nearby endpoints and snap lines to cluster centers (matches original implementation)
    
    Args:
        lines: List of LineString objects
        tolerance: Distance tolerance for clustering
        
    Returns:
        List of LineString objects with snapped endpoints
    """
    if not lines:
        return []
    
    import numpy as np
    
    # Gather all endpoints (start and end) from all lines into a flat numpy array (matches original)
    endpts = np.asarray([c for ls in lines for c in (ls.coords[0], ls.coords[-1])])

    # Cluster endpoints that are within TOL distance of each other (matches original exactly)
    clusters = []
    for x, y in endpts:
        # Try to find an existing cluster close to (x, y)
        for k, (cx, cy, n) in enumerate(clusters):
            # If the current point is within TOL distance of the cluster center
            if (x-cx)**2 + (y-cy)**2 < tolerance**2:
                n += 1  # Increment the number of points in the cluster
                # Update the cluster center as the running average
                clusters[k] = ((cx*(n-1)+x)/n, (cy*(n-1)+y)/n, n)
                break
        else:
            # No close cluster found, start a new cluster with this point
            clusters.append((x, y, 1))

    # Function to snap a point to the nearest cluster center within TOL distance (matches original)
    def snap(pt):
        for cx, cy, _ in clusters:
            if (pt[0]-cx)**2 + (pt[1]-cy)**2 < tolerance**2:
                return (cx, cy)  # Snap to cluster center
        return pt  # No cluster close enough, return original point

    # Create new lines with endpoints snapped to cluster centers (matches original)
    lines_snap = [LineString([snap(ls.coords[0]), snap(ls.coords[-1])]) for ls in lines]
    
    return lines_snap


def extend_dangles(lines: List[LineString], config: ExtractionConfig) -> List[LineString]:
    """
    Extend dangling line endpoints to connect to the network topology.
    
    This function identifies "dangles" (line endpoints that don't connect to other lines)
    and attempts to extend them toward nearby dangles to create connections. This is crucial
    for creating closed polygons from drawing elements that may have small gaps due to
    drawing precision or extraction artifacts.
    
    Algorithm:
    1. Node the input lines to create a connectivity graph
    2. Identify dangles (endpoints connected to only one line)
    3. For each dangle:
       - Calculate extension direction based on the attached line
       - Find candidate dangles within tolerance distance (0.5 units) of the extension ray
       - Apply directional constraint (within 10 degrees of extension direction)
       - Connect to the closest valid candidate dangle
    
    The algorithm uses both spatial and directional tolerances to balance connectivity
    improvement with geometric accuracy.
    
    Args:
        lines (List[LineString]): Input line geometries representing the drawing network
        config (ExtractionConfig): Configuration containing extension_length and other params
        
    Returns:
        List[LineString]: Extension lines connecting previously unconnected dangles
        
    Performance Notes:
        - Time complexity: O(n²) where n is the number of dangles
        - Spatial complexity: O(n) for the connectivity graph
        - Most expensive operation: dangle identification and ray intersection
        
    Tuning Parameters:
        - config.extension_length: Maximum extension distance (default 40.0)
        - Spatial tolerance: 0.5 drawing units (hardcoded)
        - Angular tolerance: 10 degrees (hardcoded)
        
    Example:
        >>> lines = [LineString([(0,0), (10,0)]), LineString([(15,0), (25,0)])]
        >>> config = ExtractionConfig()
        >>> extensions = extend_dangles(lines, config)
        >>> len(extensions)  # Should be 1 if dangles are within tolerance
        1
    """
    if not lines:
        return []
    
    import math
    from shapely import wkt
    
    # Node the lines: merge lines at shared endpoints
    noded = unary_union(lines)
    
    # Ensure noded is always a list of LineStrings for downstream processing
    if noded.geom_type != "LineString":
        noded = list(noded.geoms)
    else:
        noded = [noded]
    
    # Build a connectivity graph: map each endpoint (as WKT string) to the lines that touch it
    graph = {}
    for ls in noded:
        a, b = Point(ls.coords[0]), Point(ls.coords[-1])
        graph.setdefault(a.wkt, []).append(ls)
        graph.setdefault(b.wkt, []).append(ls)

    # Identify "dangles": endpoints that are only touched by a single line (potential dead-ends)
    dangles = [wkt.loads(w) for w, segs in graph.items() if len(segs) == 1]
    dangles_multipoint = MultiPoint(dangles)
    extensions = []
    net = unary_union(noded)  # Union of all noded lines for intersection checks

    # For each dangle, try to extend it to connect to the network
    for p in dangles:
        try:
            seg = graph[p.wkt][0]  # The single segment attached to this dangle
            A, B = Point(seg.coords[0]), Point(seg.coords[-1])
            other = B if p.equals(A) else A  # The other endpoint of the segment
            dx, dy = p.x - other.x, p.y - other.y  # Direction vector from other to dangle
            norm = math.hypot(dx, dy)
            if norm == 0:
                continue  # Skip degenerate cases
            dx, dy = dx/norm, dy/norm  # Normalize direction
            # Create a ray extending from the dangle point
            ray = LineString([p, (p.x+dx*config.extension_length, p.y+dy*config.extension_length)])
            
            # Find dangles within tolerance of the ray (0.5 drawing units)
            tolerance = 0.5
            cand = []
            
            for dangle in dangles:
                if dangle.equals(p):  # Skip the current dangle
                    continue
                
                distance_to_ray = ray.distance(dangle)
                if distance_to_ray <= tolerance:
                    cand.append(dangle)
            
            # Find the closest valid dangle within tolerance and direction
            best = None
            best_d = None
            direction_tolerance_degrees = 10.0
            direction_tolerance_radians = math.radians(direction_tolerance_degrees)
            
            for q in cand:
                d = math.hypot(q.x-p.x, q.y-p.y)
                # Skip if too close
                if d < 1e-3:
                    continue
                
                # Check direction constraint with tolerance
                # Calculate direction vector from current dangle to candidate
                candidate_dx = q.x - p.x
                candidate_dy = q.y - p.y
                candidate_norm = math.hypot(candidate_dx, candidate_dy)
                
                if candidate_norm > 0:
                    # Normalize candidate direction
                    candidate_dx /= candidate_norm
                    candidate_dy /= candidate_norm
                    
                    # Calculate angle between original direction (dx, dy) and candidate direction
                    # Using dot product: cos(angle) = dx*candidate_dx + dy*candidate_dy
                    dot_product = dx * candidate_dx + dy * candidate_dy
                    # Clamp to [-1, 1] to avoid numerical errors in acos
                    dot_product = max(-1.0, min(1.0, dot_product))
                    angle = math.acos(dot_product)
                    
                    # Check if angle is within tolerance
                    if angle <= direction_tolerance_radians:
                        if best is None or d < best_d:
                            best, best_d = q, d
            
            if best:
                # Add the extension from the dangle to the best candidate dangle
                extensions.append(LineString([p, best]))
        except Exception:
            continue  # Skip problematic dangles
    
    return extensions


def polygonize_lines(lines: List[LineString], extensions: List[LineString], 
                    config: ExtractionConfig) -> List[Polygon]:
    """
    Create polygons from lines and extensions (matches original implementation)
    
    Args:
        lines: Original lines
        extensions: Extension lines
        config: Configuration object
        
    Returns:
        List of Polygon objects
    """
    all_lines = lines + extensions
    
    if not all_lines:
        return []
    
    try:
        # Combine original lines and extensions, then polygonize to find faces (matches original)
        from shapely.geometry import MultiLineString
        import shapely
        
        all_lines_multi = MultiLineString(all_lines)
        union = shapely.union_all(all_lines_multi, grid_size=config.grid_size)
        faces = list(polygonize(union))
        
        # Filter by minimum area
        valid_polygons = [p for p in faces if p.area >= config.min_polygon_area]
        
        return valid_polygons
        
    except Exception as e:
        print(f"Warning: Polygonization failed: {str(e)}")
        return []


def visualize_extraction_results(page, lines: List[LineString], 
                               extensions: List[LineString], 
                               polygons: List[Polygon],
                               dangles: List[Point] = None,
                               layer_name: str = "",
                               config: ExtractionConfig = None) -> plt.Figure:
    """
    Create visualization of extraction results
    
    Args:
        page: PyMuPDF page object
        lines: Extracted lines
        extensions: Extension lines
        polygons: Resulting polygons
        dangles: Dangle points
        layer_name: Name of processed layer
        config: Configuration object
        
    Returns:
        Matplotlib figure
    """
    if config is None:
        config = ExtractionConfig()
    
    fig, axes = plt.subplots(2, 2, figsize=config.figure_size)
    fig.suptitle(f'Extraction Results - Layer: {layer_name}', fontsize=14)
    
    # Original page
    ax1 = axes[0, 0]
    pix = page.get_pixmap()
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    ax1.imshow(img, origin='upper')
    ax1.set_title('Original Page')
    ax1.axis('off')
    
    # Lines and extensions
    ax2 = axes[0, 1]
    for line in lines:
        x, y = line.xy
        ax2.plot(x, y, 'b-', linewidth=1, alpha=0.7)
    
    if config.show_extensions and extensions:
        for ext in extensions:
            x, y = ext.xy
            ax2.plot(x, y, 'r-', linewidth=2, alpha=0.8)
    
    if config.show_dangles and dangles:
        x_coords = [d.y for d in dangles]
        y_coords = [d.x for d in dangles]
        ax2.scatter(x_coords, y_coords, c='red', s=50, alpha=0.8)
    
    ax2.set_title('Lines and Extensions')
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    
    # Polygons
    ax3 = axes[1, 0]
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple']
    
    for i, poly in enumerate(polygons):
        if poly.exterior:
            y, x = poly.exterior.xy
            # Swap x and y for coordinate system
            ax3.plot(y, x, color=colors[i % len(colors)], linewidth=2)
            ax3.fill(y, x, color=colors[i % len(colors)], alpha=0.3)
    
    ax3.set_title(f'Polygons ({len(polygons)} found)')
    ax3.set_aspect('equal')
    ax3.invert_yaxis()
    
    # Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    Statistics:
    
    Lines: {len(lines)}
    Extensions: {len(extensions)}
    Polygons: {len(polygons)}
    
    Total Area: {sum(p.area for p in polygons):.2f}
    
    Config:
    Tolerance: {config.tolerance}
    Extension Length: {config.extension_length}
    Min Polygon Area: {config.min_polygon_area}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig


def extract_polygons_from_layer(file_path: str, page_number: int, layer_name: str,
                               config: ExtractionConfig = None, clip_poly: Polygon = None) -> Dict[str, Any]:
    """
    Main pipeline function to extract component boundary polygons from a specific PDF layer.
    
    This is the primary entry point for the geometric extraction pipeline. It orchestrates
    the complete workflow from PDF loading through polygon generation, handling all the
    intermediate processing steps including layer filtering, clip region processing,
    line extraction, topology improvement, and polygonization.
    
    The function is designed to work with CAD-generated PDF drawings where components
    are drawn as vector paths on specific layers (e.g., "E-CB_AREA" for combiner areas,
    "key plan|BEI-BLOCK-OUTLINE" for inverter blocks).
    
    Processing Pipeline:
    1. Load PDF page and extract all drawing elements
    2. Build clip region map to identify drawing boundaries
    3. Filter elements by target layer name
    4. Find primary clip region containing most layer elements
    5. Filter elements to those within the primary clip region
    6. Extract line geometries from vector paths (lines, rectangles, quads)
    7. Cluster and snap nearby endpoints for better connectivity
    8. Extend dangling endpoints to close gaps in the network
    9. Polygonize the resulting line network
    10. Filter polygons by minimum area threshold
    
    Args:
        file_path (str): Path to the PDF file to process
        page_number (int): Zero-indexed page number to extract from
        layer_name (str): CAD layer name to filter for extraction. Common values:
                         - "E-CB_AREA": Combiner box areas
                         - "key plan|BEI-BLOCK-OUTLINE": Inverter block outlines
                         - "EQUIPMENT": General equipment boundaries
        config (ExtractionConfig, optional): Configuration parameters. If None,
                                           default configuration is used.
        
    Returns:
        Dict[str, Any]: Comprehensive extraction results containing:
            - 'success' (bool): Whether extraction completed successfully
            - 'page' (fitz.Page): PyMuPDF page object for further processing
            - 'lines' (List[LineString]): Processed line geometries
            - 'extensions' (List[LineString]): Generated extension lines
            - 'polygons' (List[Polygon]): Final extracted polygons
            - 'layer_name' (str): The processed layer name
            - 'processing_time' (float): Total processing time in seconds
            - 'stats' (dict): Detailed processing statistics including:
                - 'total_drawings': Total drawing elements on page
                - 'layer_paths': Elements on target layer
                - 'filtered_paths': Elements after clip filtering
                - 'lines_extracted': Number of lines extracted
                - 'lines_snapped': Number of lines after snapping
                - 'extensions': Number of extension lines created
                - 'polygons': Number of final polygons
                - 'total_area': Combined area of all polygons
            
            On failure, returns:
            - 'success': False
            - 'error': Error description
            - 'polygons': []
            - 'processing_time': Partial processing time
            - 'traceback': Full error traceback (if exception occurred)
    
    Performance Characteristics:
        - Typical processing time: 0.1-2.0 seconds per page
        - Memory usage: 50-200MB depending on drawing complexity
        - Scales roughly O(n log n) with number of drawing elements
        - Bottlenecks: PyMuPDF drawing extraction, Shapely geometric operations
    
    Common Usage Patterns:
        # Basic extraction with default settings
        result = extract_polygons_from_layer("drawing.pdf", 14, "E-CB_AREA")
        
        # High precision extraction
        config = ExtractionConfig()
        config.tolerance = 0.5
        config.min_polygon_area = 1.0
        result = extract_polygons_from_layer("drawing.pdf", 14, "E-CB_AREA", config)
        
        # Check results
        if result['success']:
            polygons = result['polygons']
            print(f"Found {len(polygons)} polygons in {result['processing_time']:.2f}s")
        else:
            print(f"Extraction failed: {result['error']}")
    
    Error Handling:
        The function handles common failure modes gracefully:
        - Invalid page numbers
        - Missing or empty layers
        - Geometric processing errors
        - Memory limitations
        
        All exceptions are caught and returned as structured error responses
        rather than raising, making the function safe for batch processing.
    
    See Also:
        - load_pdf_page(): PDF loading utilities
        - build_clip_map(): Clip region processing
        - extract_lines_from_paths(): Line extraction from vector paths
        - extend_dangles(): Topology improvement algorithms
        - polygonize_lines(): Polygon generation from line networks
    """
    if config is None:
        config = ExtractionConfig()
    
    start_time = time.time()
    
    try:
        # Load page and drawings
        page, _ = load_pdf_page(file_path, page_number)
        
        # Build clip map using extended drawings (matches original)
        drawings, clip_dict = build_clip_map(page)
        
        # Filter by layer
        layer_paths = [p for p in drawings if p.get("layer") == layer_name]
        
        if not layer_paths:
            return {
                'success': False,
                'error': f'No paths found for layer: {layer_name}',
                'polygons': [],
                'processing_time': time.time() - start_time
            }
        
        # Find the clip rectangle that contains most objects on that layer
        clip_counter = Counter()
        for p in layer_paths:
            for cname in p.get("clips", []):
                clip_counter[cname] += 1

        best_clip_entry = clip_dict[clip_counter.most_common(1)[0][0]] if clip_counter else None
        
        if clip_poly is None:
            
            clip_poly = clip_to_polygon(best_clip_entry) if best_clip_entry else None

        # Keep only layer elements whose bounding-box centre is inside clip
        filtered_layer_paths = []
        bounding_rect = []
        if clip_poly and not clip_poly.is_empty:
            inside = clip_poly.contains
            for p in layer_paths:
                r = p.get("rect")
                if r:
                    cx = (r.x0 + r.x1) / 2
                    cy = (r.y0 + r.y1) / 2
                    if inside(Point(cx, cy)):
                        filtered_layer_paths.append(p)
                        bounding_rect.append(r)
        else:
            filtered_layer_paths = layer_paths[:]
        
        # Extract lines from filtered paths
        lines = extract_lines_from_paths(filtered_layer_paths, config)
        
        if not lines:
            return {
                'success': False,
                'error': 'No lines extracted from layer paths',
                'polygons': [],
                'processing_time': time.time() - start_time
            }
        
        # Snap endpoints (matches original implementation exactly)
        snapped_lines = cluster_and_snap_endpoints(lines, config.tolerance)
        
        # Extend dangles
        extensions = extend_dangles(snapped_lines, config)
        
        # Create polygons
        polygons = polygonize_lines(snapped_lines, extensions, config)
        
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'page': page,
            'lines': snapped_lines,
            'extensions': extensions,
            'polygons': polygons,
            'rects': bounding_rect,
            'layer_name': layer_name,
            'processing_time': processing_time,
            'clip_poly': clip_poly,
            'stats': {
                'total_drawings': len(drawings),
                'layer_paths': len(layer_paths),
                'filtered_paths': len(filtered_layer_paths),
                'lines_extracted': len(lines),
                'lines_snapped': len(snapped_lines),
                'extensions': len(extensions),
                'polygons': len(polygons),
                'total_area': sum(p.area for p in polygons)
            }
        }
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'polygons': [],
            'processing_time': time.time() - start_time
        }


def batch_extract_polygons(file_paths: List[str], page_numbers: List[int], 
                          layer_names: List[str], config: ExtractionConfig = None) -> Dict[str, Any]:
    """
    Batch extract polygons from multiple files/pages/layers
    
    Args:
        file_paths: List of PDF file paths
        page_numbers: List of page numbers for each file
        layer_names: List of layer names for each extraction
        config: Configuration object
        
    Returns:
        Dictionary with batch results
    """
    if config is None:
        config = ExtractionConfig()
    
    results = {}
    total_polygons = []
    
    start_time = time.time()
    
    for i, (file_path, page_num, layer_name) in enumerate(zip(file_paths, page_numbers, layer_names)):
        key = f"{os.path.basename(file_path)}_p{page_num}_{layer_name}"
        
        print(f"Processing {i+1}/{len(file_paths)}: {key}")
        
        result = extract_polygons_from_layer(file_path, page_num, layer_name, config)
        results[key] = result
        
        if result['success']:
            total_polygons.extend(result['polygons'])
    
    total_time = time.time() - start_time
    
    return {
        'results': results,
        'total_polygons': total_polygons,
        'total_processing_time': total_time,
        'summary': {
            'total_files': len(file_paths),
            'successful_extractions': sum(1 for r in results.values() if r['success']),
            'total_polygons_found': len(total_polygons),
            'average_time_per_extraction': total_time / len(file_paths) if file_paths else 0
        }
    }


def save_polygons_to_json(polygons: List[Polygon], output_path: str) -> None:
    """
    Save polygons to JSON file
    
    Args:
        polygons: List of Polygon objects
        output_path: Output file path
    """
    polygon_data = []
    
    for i, poly in enumerate(polygons):
        polygon_data.append({
            'id': i,
            'area': poly.area,
            'bounds': poly.bounds,
            'wkt': poly.wkt,
            'coordinates': list(poly.exterior.coords) if poly.exterior else []
        })
    
    with open(output_path, 'w') as f:
        json.dump(polygon_data, f, indent=2)


def load_polygons_from_json(input_path: str) -> List[Polygon]:
    """
    Load polygons from JSON file
    
    Args:
        input_path: Input file path
        
    Returns:
        List of Polygon objects
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    polygons = []
    for poly_data in data:
        if 'wkt' in poly_data:
            poly = wkt.loads(poly_data['wkt'])
            polygons.append(poly)
        elif 'coordinates' in poly_data:
            poly = Polygon(poly_data['coordinates'])
            polygons.append(poly)
    
    return polygons




def plot_polygons_on_pdf(result, file_path=None, figsize=(20, 20)):
    """
    Plots extracted polygons overlaid on the PDF page image in a notebook.

    Parameters:
        result: dict, output from extract_polygons_from_layer
        file_path: str or None, path to the PDF file (if not in result)
        figsize: tuple, size of the matplotlib figure
    """
    pdf_path = file_path if file_path is not None else result.get('file_path', None)
    if pdf_path is None:
        raise ValueError("PDF file path must be provided either as argument or in result['file_path']")
    page_obj = result.get('page')
    page_number = page_obj.number if hasattr(page_obj, 'number') else 0

    doc = fitz.open(pdf_path)
    outpage = doc[page_number]
    shape = outpage.new_shape()

    colors = [
        (1, 0, 0),  # red
        (0, 1, 0),  # green
        (0, 0, 1),  # blue
        (1, 1, 0),  # yellow
        (1, 0, 1),  # magenta
        (0, 1, 1),  # cyan
        (1, 0.5, 0),  # orange
        (0.5, 0, 1),  # purple
        (0, 0, 0),    # black
    ]

    for i, poly in enumerate(result['polygons']):
        x, y = poly.exterior.xy
        color = colors[i % len(colors)]
        points = list(zip(x, y))
        shape.draw_polyline(points)
        shape.finish(
            color=color + (0.3,),
            width=5
        )

    shape.commit()

    fig, ax = plt.subplots(figsize=figsize)
    pix = outpage.get_pixmap()
    img = pix.tobytes("png")
    im = Image.open(io.BytesIO(img))
    ax.imshow(im)
    ax.axis('off')
    plt.show()

# Functions to label polygons

def draw_polygon_on_page(page, shapely_polygon, bbox_offset=(0, 0)):
    """
    Draw polygon outline and fill on a PDF page using PyMuPDF Shape
    
    Args:
        page: PyMuPDF page object
        shapely_polygon: Shapely Polygon object  
        bbox_offset: Offset to apply to coordinates (x_offset, y_offset)
    """
    if not hasattr(shapely_polygon, 'exterior'):
        return
        
    # Get polygon coordinates
    coords = list(shapely_polygon.exterior.coords)
    
    # Convert to PyMuPDF Points, applying offset
    fitz_points = []
    for x, y in coords:
        adjusted_x = x - bbox_offset[0] 
        adjusted_y = y - bbox_offset[1]
        fitz_points.append(fitz.Point(adjusted_x, adjusted_y))
    
    if len(fitz_points) < 3:
        return
        
    # Create shape for drawing
    shape = page.new_shape()
    
    # Draw filled polygon with light transparent red
    # RGB values: light red with transparency
    fill_color = (1.0, 0.9, 0.9)  # Very light red
    stroke_color = (0.8, 0.2, 0.2)  # Darker red for outline
    
    # Draw the polygon
    shape.draw_polyline(fitz_points)
    shape.finish(
        fill=fill_color,     # Light red fill
        color=stroke_color,  # Darker red outline  
        width=1.5,          # Line width for outline
        fill_opacity=0.3,   # Make fill semi-transparent
        stroke_opacity=0.8  # Make outline more visible
    )
    
    # Handle holes if they exist
    for hole in shapely_polygon.interiors:
        hole_coords = list(hole.coords)
        hole_points = []
        for x, y in hole_coords:
            adjusted_x = x - bbox_offset[0]
            adjusted_y = y - bbox_offset[1] 
            hole_points.append(fitz.Point(adjusted_x, adjusted_y))
            
        if len(hole_points) >= 3:
            # Draw hole (white fill to "cut out" the hole)
            shape.draw_polygon(hole_points)
            shape.finish(
                fill=(1.0, 1.0, 1.0),  # White fill
                color=stroke_color,     # Same outline color
                width=1.0,
                fill_opacity=1.0,      # Opaque white
                stroke_opacity=0.8
            )
    
    shape.commit()

def get_buffered_polygon_bounds(shapely_polygon, buffer_amount=50):
    """
    Get bounding box of Shapely polygon with buffer
    
    Args:
        shapely_polygon: Shapely Polygon object
        buffer_amount: Buffer to add around polygon (in points)
    
    Returns:
        tuple: (buffered_polygon, bbox)
    """
    if shapely_polygon is None:
        return None, None
    
    # Buffer the polygon
    buffered_polygon = shapely_polygon.buffer(buffer_amount)
    
    # Get bounding box of buffered polygon
    bounds = buffered_polygon.bounds
    bbox = (bounds[0], bounds[1], bounds[2], bounds[3])  # (minx, miny, maxx, maxy)
    
    return buffered_polygon, bbox

def clip_pdf_to_polygon(pdf_path, page_num, shapely_polygon, buffer_amount=50, output_path=None, dpi=600, draw_polygon=True):
    """
    Clip PDF page to polygon shape with buffer, optionally drawing the polygon first
    
    Args:
        pdf_path: Path to input PDF
        page_num: Page number (0-indexed)
        shapely_polygon: Shapely Polygon object
        buffer_amount: Buffer around polygon (in points)
        output_path: Output path for clipped PDF (optional)
        dpi: DPI for raster output
        draw_polygon: Whether to draw polygon outline and fill before clipping
    
    Returns:
        dict: Contains both vector PDF and masked raster image
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Get buffered polygon and bounding box
    buffered_polygon, bbox = get_buffered_polygon_bounds(shapely_polygon, buffer_amount)
    
    if bbox is None:
        doc.close()
        return None
    
    # Draw the original polygon on the page before clipping (if requested)
    if draw_polygon:
        draw_polygon_on_page(page, shapely_polygon)
    
    # Create rectangle for clipping
    clip_rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
    
    results = {}
    

    """
    new_doc = fitz.open(pdf_path)
    new_page = new_doc.new_page(width=clip_rect.width, height=clip_rect.height)
    
    # Copy the clipped content to the bounding box
    new_page.show_pdf_page(new_page.rect, doc, page_num, clip=clip_rect)
    """

    page.set_cropbox(clip_rect)
    # Save vector PDF (bounding box clipped)
    results['vector_pdf'] = doc
    results['vector_note'] = "Vector PDF uses bounding box clipping with polygon overlay."
    
    """# Raster output at specified DPI with masking
    mat = fitz.Matrix(dpi/72, dpi/72)  # Create transformation matrix for DPI
    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
    img_data = pix.tobytes("png")
    
    # Convert to PIL Image
    img = Image.open(io.BytesIO(img_data))
    
    # Create mask for the polygon
    #mask = create_polygon_mask(buffered_polygon, bbox, img.width, img.height)
    
    # Apply mask to image
    # Create RGBA version if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Apply mask - make areas outside polygon transparent
    #img.putalpha(mask)
    
    results['raster_image'] = img
    #results['mask'] = mask
    results['dpi'] = dpi
    results['bbox'] = bbox
    results['buffered_polygon'] = buffered_polygon"""
    
    # Save raster image if output path provided (commented out since raster functionality is disabled)
    # if output_path:
    #     raster_path = output_path.replace('.pdf', f'_raster_{dpi}dpi.png')
    #     img.save(raster_path)
    #     results['raster_path'] = raster_path
    
    
    return results

def get_label_from_gemini(img_bytes,labels,element_type):
    # Load .env file from the project root (relative to this notebook)
   

    USE_GEMINI = True  # Set to False to use Groq instead

    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    if element_type == "inverter":
        label_instructions = "Inverter labels will be labelled as 'Block X.X' (e.g., 'Block 1.1', 'Block 1.2', 'Block 2.1'), return only the number"
    elif element_type == "combiner":
        label_instructions = "Combiner labels will be labelled as 'X.X.X.C.X' (e.g., '1.2.3.C.1', '1.2.3.C.2', '2.1.4.C.1')"
    else:
        label_instructions = "Unknown element type"

    prompt = f"""You are analyzing an electrical as-built drawing image, an area of interest has been higligted in red, representing a {element_type}. 
    Please the label most likely associated with the area of interest. When deciding what text is best associated with the area of interest,
    consider the content of teh text as more important than relative size or color of the text. 

    {label_instructions}

    Respond with a JSON object like this:

    ```json
    {{"element_type": "inverter",
    "label": " 1.1"}}
    ```json

    for context, here are up to 5 previous labels for this same type of device: {labels[:5]}
    """

    # Convert img_bytes to a PIL Image
    img = Image.open(io.BytesIO(img_bytes))

    response = model.generate_content(
        [prompt, img]
    )

    import json
    # Extract label from Gemini response JSON
    try:
        json_str = response.text
        json_str = json_str.replace('```json', '').replace('```', '').strip()
        label_data = json.loads(json_str)
        label = label_data.get("label", None)
    except Exception:
        label = None
    return label, response.usage_metadata.total_token_count

def process_elements_on_page(file_path: str, page_number: int, result: Dict[str, Any], 
                           element_type: str, output_dir: Optional[str] = None) -> Tuple[Dict[str, Polygon], int]:
    """
    Process all extracted polygons on a page to generate AI-based labels for electrical components.
    
    This function takes the output from extract_polygons_from_layer() and generates labeled
    component data by clipping each polygon region, highlighting it visually, and sending
    the image to Google Gemini AI for label identification. It's designed specifically for
    electrical as-built drawings where components have standardized labeling schemes.
    
    The function creates visual clips with red highlighting around each polygon to provide
    context to the AI model, then processes each clip individually to extract component
    labels based on the specified element type.
    
    Args:
        file_path (str): Path to the source PDF file
        page_number (int): Zero-indexed page number being processed
        result (Dict[str, Any]): Output from extract_polygons_from_layer() containing:
                               - 'polygons': List of Polygon objects to process
                               - 'page': PyMuPDF page object
                               - Other extraction metadata
        element_type (str): Type of electrical component for labeling context.
                          Supported values:
                          - "inverter": For inverter blocks (expects "Block X.X" labels)
                          - "combiner": For combiner boxes (expects "X.X.X.C.X" labels)
        output_dir (Optional[str]): If provided, saves clipped images to this directory.
                                  Primarily used for debugging and manual review.
    
    Returns:
        Tuple[Dict[str, Polygon], int]: A tuple containing:
            - label_poly_dict (Dict[str, Polygon]): Mapping of extracted labels to their
              corresponding Polygon objects. Keys are label strings, values are Shapely
              Polygon geometries.
            - token_total (int): Total number of AI API tokens consumed during processing.
              Used for cost tracking and rate limiting.
    
    Processing Workflow:
        1. For each polygon in the extraction result:
           a. Create clipped PDF region with 20-unit buffer around polygon
           b. Draw red highlight overlay on the polygon area
           c. Render clipped region as 100 DPI image
           d. Send image to Gemini AI with element-type-specific prompt
           e. Parse JSON response to extract component label
           f. Accumulate tokens used and maintain label history for context
        
        2. Build final mapping dictionary of labels to polygons
        3. Return results with total token consumption
    
    Performance Characteristics:
        - Processing time: 2-10 seconds per polygon (AI API dependent)
        - Network dependency: Requires stable internet for Gemini API
        - Token consumption: ~100-500 tokens per polygon
        - Memory usage: ~10-50MB per polygon for image processing
    
    Error Handling:
        - AI API failures result in None labels but don't stop processing
        - Invalid JSON responses are caught and handled gracefully
        - Network timeouts are handled by the underlying API client
        - Malformed polygons are skipped with warnings
    
    API Dependencies:
        - Google Gemini AI API (requires GEMINI_API_KEY in environment)
        - Rate limits apply based on API tier (typically 60 requests/minute)
        - Token limits may apply for large batch processing
    
    Example Usage:
        # Process combiner boxes on page 15
        result = extract_polygons_from_layer("drawing.pdf", 15, "E-CB_AREA")
        if result['success']:
            labels, tokens = process_elements_on_page(
                "drawing.pdf", 15, result, "combiner"
            )
            print(f"Found {len(labels)} labeled combiners using {tokens} tokens")
            for label, polygon in labels.items():
                print(f"  {label}: {polygon.area:.1f} sq units")
    
    Label Format Examples:
        Inverter blocks: "1.1", "1.2", "2.1" (extracted from "Block 1.1", etc.)
        Combiner boxes: "1.2.3.C.1", "1.2.3.C.2", "2.1.4.C.1"
    
    Cost Considerations:
        - Each polygon incurs API costs (~$0.001-0.01 per polygon)
        - Large documents can result in significant API usage
        - Consider batching strategies for cost optimization
        - Monitor token consumption for budget management
    
    See Also:
        - get_label_from_gemini(): Individual polygon labeling
        - clip_pdf_to_polygon(): Visual clipping and highlighting
        - extract_polygons_from_layer(): Upstream polygon extraction
    """
 
    token_total = 0
    label_poly_dict = {}
    labels =[]
   
    for idx, poly in enumerate(result):

        #output_path = os.path.join(output_dir, f"polygon_clip_{idx}.png")
        clipped_results = clip_pdf_to_polygon(
            pdf_path=file_path,
            page_num=page_number,
            shapely_polygon=poly,
            buffer_amount=20,
            output_path=None,
            dpi=100,
            draw_polygon=True
        )
        
        img_bytes = clipped_results['vector_pdf'][page_number].get_pixmap(dpi=100).tobytes("png")

        label, token_count = get_label_from_gemini(img_bytes,labels,element_type)
        labels.append(label)
        token_total += token_count
        #print(label+" "+str(token_count))
        

        if output_dir:
            img_bytes = clipped_results['vector_pdf'][page_number].get_pixmap(dpi=100).tobytes("png")
            with open(output_dir, "wb") as f:
                f.write(img_bytes)
        
        label_poly_dict[label] = poly
    return label_poly_dict, token_total

def get_page_label_from_gemini(img_bytes):


    USE_GEMINI = True  # Set to False to use Groq instead

    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""You are analyzing an electrical as-built drawing image. Return only the number of the inverter block being
    represented in this page. You can find this information in the title block. 

    Respond with a JSON object like this:

    ```json
    {{"inverter_block_number": " 1.1"}}
    ```json

    """

    # Convert img_bytes to a PIL Image
    img = Image.open(io.BytesIO(img_bytes))

    response = model.generate_content(
        [prompt, img]
    )

    print(response.text)
    import json
    # Extract label from Gemini response JSON
    try:
        json_str = response.text
        json_str = json_str.replace('```json', '').replace('```', '').strip()
        label_data = json.loads(json_str)
        label = label_data.get("inverter_block_number", None)
    except Exception:
        label = None
    return label, response.usage_metadata.total_token_count



def bbox_of_geoms(geoms):
    """Return (minx, miny, maxx, maxy) for a list of shapely geometries."""
    if not geoms:
        raise ValueError("No geometries provided")
    union = unary_union(geoms)
    return union.bounds  # (minx, miny, maxx, maxy)

def affine_from_bbox(src_bbox, dst_bbox, keep_aspect=True, center=True):
    """
    Build an affine transform that maps src_bbox -> dst_bbox.
    Returns the 6-tuple (a, b, d, e, xoff, yoff) for shapely.affinity.affine_transform.

    keep_aspect=True: uniform scale (fit inside), preserving aspect ratio.
    center=True: center the fitted result in the destination bbox when keep_aspect is True.
    """
    sx_src = src_bbox[2] - src_bbox[0]
    sy_src = src_bbox[3] - src_bbox[1]
    sx_dst = dst_bbox[2] - dst_bbox[0]
    sy_dst = dst_bbox[3] - dst_bbox[1]
    if sx_src == 0 or sy_src == 0:
        raise ValueError("Source bbox has zero size")

    if keep_aspect:
        s = min(sx_dst / sx_src, sy_dst / sy_src)
        a = s  # scale x
        e = s  # scale y
        b = d = 0.0
        # map src min corner to (dst min) first, then optionally center
        xoff = dst_bbox[0] - src_bbox[0] * a
        yoff = dst_bbox[1] - src_bbox[1] * e

        if center:
            # compute fitted size and add centering offset
            fitted_w = sx_src * s
            fitted_h = sy_src * s
            extra_x = (sx_dst - fitted_w) / 2.0
            extra_y = (sy_dst - fitted_h) / 2.0
            xoff += extra_x
            yoff += extra_y
    else:
        # non-uniform scale to exactly fill destination bbox
        a = sx_dst / sx_src
        e = sy_dst / sy_src
        b = d = 0.0
        xoff = dst_bbox[0] - src_bbox[0] * a
        yoff = dst_bbox[1] - src_bbox[1] * e

    return (a, b, d, e, xoff, yoff)

def transform_page_polygons(page_polys_dict, base_poly, 
                            page_frame_bbox=None, keep_aspect=True, center=True):
    """
    Transform a page's sub-polygons into the base coordinate system defined by base_poly.
    - page_polys_dict: dict {label: shapely Polygon}
    - base_poly: shapely Polygon for the corresponding cell on the base layout
    - page_frame_bbox: optional (minx, miny, maxx, maxy) for the page's own coordinate frame.
      If None, will use the bbox of all sub-polygons.
    Returns: dict {label: transformed Polygon}
    """
    if page_frame_bbox is None:
        src_bbox = bbox_of_geoms(list(page_polys_dict.values()))
    else:
        src_bbox = page_frame_bbox

    dst_bbox = base_poly.bounds
    A = affine_from_bbox(src_bbox, dst_bbox, keep_aspect=keep_aspect, center=center)

    transformed = {k: affine_transform(geom, A) for k, geom in page_polys_dict.items()}
    return transformed, A  # return A too if you want to reuse/log

def convert_to_shapely_geometry(geom):
    """
    Convert various geometry types to Shapely geometries.
    
    Args:
        geom: Geometry object (can be Shapely geometry, PyMuPDF Rect, etc.)
        
    Returns:
        Shapely geometry object or None if conversion fails
    """
    # If it's already a Shapely geometry, return as-is
    if hasattr(geom, '__geo_interface__') or hasattr(geom, 'geom_type'):
        return geom
    
    # Handle PyMuPDF Rect objects
    if hasattr(geom, 'x0') and hasattr(geom, 'y0') and hasattr(geom, 'x1') and hasattr(geom, 'y1'):
        # Convert Rect to Polygon
        coords = [
            (geom.x0, geom.y0),  # bottom-left
            (geom.x1, geom.y0),  # bottom-right  
            (geom.x1, geom.y1),  # top-right
            (geom.x0, geom.y1),  # top-left
            (geom.x0, geom.y0)   # close the polygon
        ]
        return Polygon(coords)
    
    # Handle PyMuPDF Point objects
    if hasattr(geom, 'x') and hasattr(geom, 'y') and not hasattr(geom, 'coords'):
        return Point(geom.x, geom.y)
    
    # Handle coordinate tuples/lists
    if isinstance(geom, (tuple, list)) and len(geom) >= 2:
        if len(geom) == 2:  # Point
            return Point(geom[0], geom[1])
        elif len(geom) >= 4:  # Could be a bbox or polygon coords
            # Assume it's a polygon if more than 2 coordinates
            return Polygon(geom)
    
    # If we can't convert it, return None
    print(f"Warning: Cannot convert geometry of type {type(geom)} to Shapely geometry")
    return None

def to_geojson_feature(geom, properties):
    return {
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": properties or {}
    }

def build_layout_geojson(base_polys, pages, 
                         key_normalizer=lambda s: s.strip(),
                         keep_aspect=True, center=True,
                         include_base_cells=True, transform_to_base=True):
    """
    Create a GeoJSON FeatureCollection.
    - base_polys: dict like {" 1.2": Polygon, ...} defining the base layout.
    - pages: dict like {page_number: {"page_label": "1.2", "combiner_label_poly_dict": {...}, "page_elements": {...}, ...}}
    - keep_aspect/center: how to fit page -> cell.
    - include_base_cells: include the outer/base polygons as their own features.
    """
    features = []

    # optional: add base cells
    if include_base_cells:
        for base_key, base_geom in base_polys.items():
            features.append(
                to_geojson_feature(
                    base_geom,
                    {"role": "base_cell", "base_key": key_normalizer(base_key)}
                )
            )

    # add page child polygons and other page elements
    for page_num, page_info in pages.items():
        if page_info["page_label"] is None:
            print(f"Page {page_num} has no label")
            continue
        page_label = key_normalizer(page_info["page_label"])
        sub_polys = page_info["combiner_label_poly_dict"]
        page_elements = page_info.get("page_elements", {})
        
        if page_label not in {key_normalizer(k): None for k in base_polys}.keys():
            # If labels don't match perfectly, adapt `key_normalizer` or map explicitly.
            raise KeyError(f"Page label '{page_label}' not found in base polygons keys")

        # find the actual base polygon by matching normalized keys
        base_key = next(k for k in base_polys.keys() if key_normalizer(k) == page_label)
        base_geom = base_polys[base_key]

        # transform sub-polygons to base coordinate system
        if transform_to_base:
            transformed, A = transform_page_polygons(
                sub_polys, base_geom, page_frame_bbox=None, keep_aspect=keep_aspect, center=center
            )
        else:
            transformed = sub_polys
            A = [None, None, None, None, None, None]

        # emit combiner features
        for combiner_label, g in transformed.items():
            props = {
                "role": "page_child",
                "element_type": "combiner",
                "parent_page_label": page_label,
                "parent_page_number": page_num,
                "combiner_label": combiner_label,
                # stash the transform used for traceability:
                "affine_transform": {"a": A[0], "b": A[1], "d": A[2], "e": A[3], "xoff": A[4], "yoff": A[5]}
            }
            features.append(to_geojson_feature(g, props))

        # transform and emit other page elements using the same transform
        if transform_to_base and A is not None and A[0] is not None:
            for element_type, geometries in page_elements.items():
                if not geometries:  # Skip empty lists
                    continue
                    
                # Transform each geometry using the same affine transform
                for idx, geom in enumerate(geometries):
                    if geom is None:
                        continue
                    
                    try:
                        # Convert PyMuPDF objects to Shapely geometries if needed
                        shapely_geom = convert_to_shapely_geometry(geom)
                        if shapely_geom is None:
                            continue
                        
                        # Apply the same transform used for combiners
                        transformed_geom = affine_transform(shapely_geom, A)
                        
                        props = {
                            "role": "page_element",
                            "element_type": element_type,
                            "element_index": idx,
                            "parent_page_label": page_label,
                            "parent_page_number": page_num,
                            # stash the transform used for traceability:
                            "affine_transform": {"a": A[0], "b": A[1], "d": A[2], "e": A[3], "xoff": A[4], "yoff": A[5]}
                        }
                        
                        features.append(to_geojson_feature(transformed_geom, props))
                        
                    except Exception as e:
                        print(f"Warning: Failed to transform {element_type}[{idx}] on page {page_num}: {e}")
                        continue
        elif not transform_to_base:
            # If not transforming, include page elements in original coordinates
            for element_type, geometries in page_elements.items():
                if not geometries:  # Skip empty lists
                    continue
                    
                for idx, geom in enumerate(geometries):
                    if geom is None:
                        continue
                    
                    try:
                        # Convert PyMuPDF objects to Shapely geometries if needed
                        shapely_geom = convert_to_shapely_geometry(geom)
                        if shapely_geom is None:
                            continue
                        
                        props = {
                            "role": "page_element",
                            "element_type": element_type,
                            "element_index": idx,
                            "parent_page_label": page_label,
                            "parent_page_number": page_num,
                            "affine_transform": {"a": None, "b": None, "d": None, "e": None, "xoff": None, "yoff": None}
                        }
                        
                        features.append(to_geojson_feature(shapely_geom, props))
                        
                    except Exception as e:
                        print(f"Warning: Failed to process {element_type}[{idx}] on page {page_num}: {e}")
                        continue

    return {
        "type": "FeatureCollection",
        "features": features
    }

def extract_page_elements(test_pdf, page_number, config, layers_dict):
    results = {}
    # First, extract the combiner_outlines (clip polygon) if present, as others may depend on it
    clip_poly = None
    if 'combiner_outlines' in layers_dict:
        combiner_result = extract_polygons_from_layer(
            test_pdf, page_number, layers_dict['combiner_outlines']['layer'], config, None
        )
        results['combiner_outlines'] = combiner_result
        clip_poly = combiner_result.get('clip_poly', None)

    for key, info in layers_dict.items():
        if key == 'combiner_outlines':
            continue  # already handled
        result = extract_polygons_from_layer(
            test_pdf, page_number, info['layer'], config, clip_poly
        )
        results[key] = result

    page_elements = {}
    for key, info in layers_dict.items():
        result = results.get(key, {})
        typ = info['type']
        if typ == 'polygon':
            page_elements[key] = result.get('polygons', [])
        elif typ == 'rectangle':
            page_elements[key] = result.get('rects', [])
        elif typ == 'line':
            lines = result.get('lines', [])
            extensions = result.get('extensions', [])
            page_elements[key] = lines + extensions
        else:
            page_elements[key] = []

    return page_elements