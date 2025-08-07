"""
Transform debugging helper functions for visualization
"""

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiLineString, LineString, MultiPolygon
from shapely.affinity import affine_transform
from shapely.ops import unary_union
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm

def plot_transform_steps(source_geom, target_geom, affine_params, intermediate_steps=None):
    """
    Plot the transformation process step by step
    
    Args:
        source_geom: Source geometry (Polygon, MultiLineString, etc.)
        target_geom: Target geometry  
        affine_params: List of 6 affine parameters [a, b, d, e, xoff, yoff]
        intermediate_steps: Optional list of intermediate geometries
    """
    
    num_plots = 3 if not intermediate_steps else len(intermediate_steps) + 2
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    # Plot 1: Source geometry
    ax = axes[0]
    plot_geometry(ax, source_geom, color='blue', label='Source', alpha=0.5)
    ax.set_title("Source Geometry")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot intermediate steps
    if intermediate_steps:
        for i, step_geom in enumerate(intermediate_steps):
            ax = axes[i+1]
            plot_geometry(ax, source_geom, color='blue', alpha=0.2, label='Original')
            plot_geometry(ax, step_geom, color='orange', alpha=0.7, 
                         label=f'Step {i+1}')
            ax.set_title(f"Intermediate Step {i+1}")
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    # Final plot: Overlay
    ax = axes[-1]
    plot_geometry(ax, source_geom, color='blue', alpha=0.3, label='Source')
    if target_geom:
        plot_geometry(ax, target_geom, color='red', alpha=0.3, label='Target')
    
    # Apply transform to source
    if affine_params and not any(np.isnan(affine_params)):
        transformed = affine_transform(source_geom, affine_params)
        plot_geometry(ax, transformed, color='green', alpha=0.7, label='Transformed')
    
    ax.set_title("Transform Overlay")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Hide unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_geometry(ax, geom, color='blue', alpha=0.5, label=None, linewidth=1):
    """Helper to plot various geometry types"""
    
    if geom is None:
        return
        
    if hasattr(geom, 'geom_type'):
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            ax.plot(x, y, color=color, alpha=alpha, label=label, linewidth=linewidth)
            ax.fill(x, y, color=color, alpha=alpha*0.3)
            
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth)
                ax.fill(x, y, color=color, alpha=alpha*0.3)
            if label:
                ax.plot([], [], color=color, label=label)  # For legend
                
        elif geom.geom_type == 'LineString':
            x, y = geom.xy
            ax.plot(x, y, color=color, alpha=alpha, label=label, linewidth=linewidth)
            
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                x, y = line.xy
                ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth)
            if label:
                ax.plot([], [], color=color, label=label)  # For legend
                
        elif geom.geom_type == 'Point':
            ax.scatter(geom.x, geom.y, color=color, alpha=alpha, label=label, s=50)
            
        elif geom.geom_type == 'MultiPoint':
            xs = [pt.x for pt in geom.geoms]
            ys = [pt.y for pt in geom.geoms]
            ax.scatter(xs, ys, color=color, alpha=alpha, label=label, s=50)

def plot_affine_decomposition(affine_params):
    """
    Decompose and visualize affine transformation parameters
    
    Args:
        affine_params: [a, b, d, e, xoff, yoff]
    """
    if len(affine_params) != 6:
        return None
        
    a, b, d, e, xoff, yoff = affine_params
    
    # Calculate scale, rotation, and shear
    scale_x = np.sqrt(a**2 + d**2)
    scale_y = np.sqrt(b**2 + e**2)
    rotation = np.arctan2(d, a) * 180 / np.pi
    shear = np.arctan2(b, e) * 180 / np.pi - rotation
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot 1: Unit square transformation
    ax = axes[0, 0]
    unit_square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    transformed_square = affine_transform(unit_square, affine_params)
    
    plot_geometry(ax, unit_square, color='blue', alpha=0.5, label='Unit Square')
    plot_geometry(ax, transformed_square, color='red', alpha=0.5, label='Transformed')
    
    ax.set_title("Unit Square Transformation")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    
    # Plot 2: Parameter values
    ax = axes[0, 1]
    ax.axis('off')
    
    param_text = f"""
    Affine Parameters:
    a = {a:.4f}    b = {b:.4f}
    d = {d:.4f}    e = {e:.4f}
    xoff = {xoff:.4f}
    yoff = {yoff:.4f}
    
    Decomposition:
    Scale X: {scale_x:.4f}
    Scale Y: {scale_y:.4f}
    Rotation: {rotation:.2f}°
    Shear: {shear:.2f}°
    Translation: ({xoff:.2f}, {yoff:.2f})
    """
    
    ax.text(0.1, 0.5, param_text, fontsize=10, family='monospace',
            verticalalignment='center')
    ax.set_title("Transformation Parameters")
    
    # Plot 3: Vector field
    ax = axes[1, 0]
    x = np.linspace(-1, 2, 10)
    y = np.linspace(-1, 2, 10)
    X, Y = np.meshgrid(x, y)
    
    # Apply transformation to grid points
    U = a * X + b * Y + xoff - X
    V = d * X + e * Y + yoff - Y
    
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, alpha=0.5)
    ax.set_title("Transformation Vector Field")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    
    # Plot 4: Error metrics (if available)
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title("Transformation Quality Metrics")
    
    # Add quality metrics if we can calculate them
    quality_text = """
    Quality Metrics:
    
    Orthogonality: Check if a*b + d*e ≈ 0
    Current: {:.6f}
    
    Uniform Scale: Check if scale_x ≈ scale_y
    Ratio: {:.4f}
    
    Determinant: {}
    (Positive = preserves orientation)
    """.format(
        a*b + d*e,
        scale_x/scale_y if scale_y != 0 else float('inf'),
        a*e - b*d
    )
    
    ax.text(0.1, 0.5, quality_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    return fig

def plot_alignment_quality(source_polys, target_polys, transformed_polys):
    """
    Visualize alignment quality between transformed source and target
    
    Args:
        source_polys: List of source polygons
        target_polys: List of target polygons  
        transformed_polys: List of transformed source polygons
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Overlap visualization
    ax = axes[0]
    
    # Plot targets in gray
    for poly in target_polys[:100]:  # Limit for performance
        if hasattr(poly, 'exterior'):
            x, y = poly.exterior.xy
            ax.fill(x, y, color='gray', alpha=0.3)
            ax.plot(x, y, color='gray', linewidth=0.5)
    
    # Plot transformed in color based on overlap
    for poly in transformed_polys[:100]:
        if hasattr(poly, 'exterior'):
            # Calculate overlap with targets
            overlap_area = 0
            for target in target_polys:
                if hasattr(target, 'intersection'):
                    overlap_area += poly.intersection(target).area
            
            overlap_ratio = overlap_area / poly.area if poly.area > 0 else 0
            color = cm.RdYlGn(overlap_ratio)  # Red to green colormap
            
            x, y = poly.exterior.xy
            ax.fill(x, y, color=color, alpha=0.5)
            ax.plot(x, y, color='black', linewidth=0.5)
    
    ax.set_title("Overlap Quality (Red=Bad, Green=Good)")
    ax.set_aspect('equal')
    
    # Plot 2: Distance heatmap
    ax = axes[1]
    
    # Calculate distances between transformed and target centroids
    distances = []
    positions = []
    
    for t_poly in transformed_polys[:100]:
        if hasattr(t_poly, 'centroid'):
            centroid = t_poly.centroid
            min_dist = float('inf')
            
            for target in target_polys:
                if hasattr(target, 'centroid'):
                    dist = centroid.distance(target.centroid)
                    min_dist = min(min_dist, dist)
            
            if min_dist < float('inf'):
                distances.append(min_dist)
                positions.append((centroid.x, centroid.y))
    
    if distances:
        xs, ys = zip(*positions)
        scatter = ax.scatter(xs, ys, c=distances, cmap='RdYlGn_r', s=20, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Distance to Nearest Target')
    
    ax.set_title("Centroid Distance Heatmap")
    ax.set_aspect('equal')
    
    # Plot 3: Statistics
    ax = axes[2]
    ax.axis('off')
    
    # Calculate statistics
    total_source = len(source_polys)
    total_target = len(target_polys)
    total_transformed = len(transformed_polys)
    
    # Calculate overlap statistics
    overlap_ratios = []
    for poly in transformed_polys[:100]:
        if hasattr(poly, 'area') and poly.area > 0:
            overlap_area = 0
            for target in target_polys:
                if hasattr(target, 'intersection'):
                    overlap_area += poly.intersection(target).area
            overlap_ratios.append(overlap_area / poly.area)
    
    stats_text = f"""
    Alignment Statistics:
    
    Source Polygons: {total_source}
    Target Polygons: {total_target}
    Transformed: {total_transformed}
    
    Overlap Quality:
    Mean: {np.mean(overlap_ratios):.2%} if overlap_ratios else 0
    Std: {np.std(overlap_ratios):.2%} if overlap_ratios else 0
    Min: {np.min(overlap_ratios):.2%} if overlap_ratios else 0
    Max: {np.max(overlap_ratios):.2%} if overlap_ratios else 0
    
    Distance Stats:
    Mean: {np.mean(distances):.2f} if distances else 0
    Std: {np.std(distances):.2f} if distances else 0
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    ax.set_title("Alignment Statistics")
    
    plt.tight_layout()
    return fig

def plot_racking_pitch_analysis(polygons):
    """
    Analyze and visualize racking pitch (spacing between rows)
    
    Args:
        polygons: List of rack polygons
    """
    if not polygons:
        return None
        
    # Extract centroid Y coordinates
    y_coords = []
    for poly in polygons:
        if hasattr(poly, 'centroid'):
            y_coords.append(poly.centroid.y)
    
    if len(y_coords) < 2:
        return None
        
    y_coords = np.array(sorted(y_coords))
    
    # Calculate differences (pitch)
    y_diffs = np.diff(y_coords)
    
    # Filter out very small differences (same row)
    threshold = np.percentile(y_diffs, 25) if len(y_diffs) > 4 else np.min(y_diffs) * 2
    row_pitches = y_diffs[y_diffs > threshold]
    
    if len(row_pitches) == 0:
        return None
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Y-coordinate distribution
    ax = axes[0, 0]
    ax.hist(y_coords, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Y Coordinate")
    ax.set_ylabel("Count")
    ax.set_title("Y-Coordinate Distribution")
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Pitch histogram
    ax = axes[0, 1]
    ax.hist(row_pitches, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(row_pitches), color='red', linestyle='--', 
              label=f'Mean: {np.mean(row_pitches):.2f}')
    ax.axvline(np.median(row_pitches), color='green', linestyle='--',
              label=f'Median: {np.median(row_pitches):.2f}')
    ax.set_xlabel("Row Pitch")
    ax.set_ylabel("Count")
    ax.set_title("Row Pitch Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Pitch variation along array
    ax = axes[1, 0]
    ax.plot(range(len(row_pitches)), row_pitches, 'b.-', alpha=0.7)
    ax.axhline(np.mean(row_pitches), color='red', linestyle='--', alpha=0.5)
    ax.fill_between(range(len(row_pitches)), 
                    np.mean(row_pitches) - np.std(row_pitches),
                    np.mean(row_pitches) + np.std(row_pitches),
                    color='red', alpha=0.2)
    ax.set_xlabel("Row Index")
    ax.set_ylabel("Pitch")
    ax.set_title("Pitch Variation Along Array")
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Find most common pitch
    rounded_pitches = np.round(row_pitches, 2)
    unique_pitches, counts = np.unique(rounded_pitches, return_counts=True)
    most_common_pitch = unique_pitches[np.argmax(counts)]
    
    stats_text = f"""
    Pitch Statistics:
    
    Total Rows Detected: {len(y_coords)}
    Row Gaps Analyzed: {len(row_pitches)}
    
    Mean Pitch: {np.mean(row_pitches):.4f}
    Median Pitch: {np.median(row_pitches):.4f}
    Std Dev: {np.std(row_pitches):.4f}
    Min Pitch: {np.min(row_pitches):.4f}
    Max Pitch: {np.max(row_pitches):.4f}
    
    Most Common: {most_common_pitch:.4f}
    (appears {np.max(counts)} times)
    
    Coefficient of Variation: {np.std(row_pitches)/np.mean(row_pitches):.2%}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    ax.set_title("Pitch Analysis Summary")
    
    plt.tight_layout()
    return fig