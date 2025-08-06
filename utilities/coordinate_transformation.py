"""
Coordinate Transformation Utilities for PDF Layout Extraction

This module provides utilities for transforming coordinates from PDF space
to a shared site coordinate system using affine transformations.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from shapely.geometry import (
    MultiLineString, Polygon, MultiPolygon, Point,
    LineString, GeometryCollection
)
from shapely.affinity import affine_transform
from shapely.ops import unary_union, polygonize
import shapely


def align_multilinestrings_by_convex_hull(
    source_mls: MultiLineString,
    target_mls: MultiLineString,
    scale_mode: str = 'area'
) -> Tuple[List[float], MultiLineString]:
    """
    Calculate affine transform (isotropic scale + translation) to align source_mls to target_mls
    based on their convex hulls.
    
    Args:
        source_mls: Source MultiLineString to transform
        target_mls: Target MultiLineString to align to
        scale_mode: 'area' (scale by sqrt of area ratio) or 'fit' (fit entirely inside target)
    
    Returns:
        Tuple of (affine_params, transformed_source_mls)
        affine_params: [a, b, d, e, xoff, yoff] for use with affine_transform
    """
    # Compute convex hulls
    source_hull = source_mls.convex_hull
    target_hull = target_mls.convex_hull

    # Centroids
    source_centroid = source_hull.centroid
    target_centroid = target_hull.centroid

    if scale_mode == 'area':
        # Areas
        source_area = source_hull.area
        target_area = target_hull.area

        # Isotropic scale to match areas
        scale = np.sqrt(target_area / source_area) if source_area > 0 else 1.0

        # Translation to align centroids after scaling
        dx = target_centroid.x - (source_centroid.x * scale)
        dy = target_centroid.y - (source_centroid.y * scale)

    elif scale_mode == 'fit':
        # Get bounds: (minx, miny, maxx, maxy)
        s_minx, s_miny, s_maxx, s_maxy = source_hull.bounds
        t_minx, t_miny, t_maxx, t_maxy = target_hull.bounds

        s_width = s_maxx - s_minx
        s_height = s_maxy - s_miny
        t_width = t_maxx - t_minx
        t_height = t_maxy - t_miny

        # Avoid division by zero
        if s_width == 0 or s_height == 0:
            scale = 1.0
        else:
            scale_x = t_width / s_width
            scale_y = t_height / s_height
            scale = min(scale_x, scale_y)

        # Align centers of bounding boxes
        s_center_x = (s_minx + s_maxx) / 2
        s_center_y = (s_miny + s_maxy) / 2
        t_center_x = (t_minx + t_maxx) / 2
        t_center_y = (t_miny + t_maxy) / 2

        scaled_source_center_x = s_center_x * scale
        scaled_source_center_y = s_center_y * scale

        dx = t_center_x - scaled_source_center_x
        dy = t_center_y - scaled_source_center_y

    else:
        raise ValueError("scale_mode must be 'area' or 'fit'")

    # Affine params: [a, b, d, e, xoff, yoff]
    affine_params = [scale, 0, 0, scale, dx, dy]
    transformed_source = affine_transform(source_mls, affine_params)

    return affine_params, transformed_source


def compose_affine(
    transform1: List[float],
    transform2: List[float]
) -> List[float]:
    """
    Compose two affine transformations.
    
    Args:
        transform1: First affine transform [a, b, d, e, xoff, yoff]
        transform2: Second affine transform [a, b, d, e, xoff, yoff]
    
    Returns:
        Combined affine transform parameters
    """
    a1, b1, d1, e1, xoff1, yoff1 = transform1
    a2, b2, d2, e2, xoff2, yoff2 = transform2
    
    return [
        a1 * a2 + b1 * d2,
        a1 * b2 + b1 * e2,
        d1 * a2 + e1 * d2,
        d1 * b2 + e1 * e2,
        xoff1 * a2 + yoff1 * d2 + xoff2,
        xoff1 * b2 + yoff1 * e2 + yoff2,
    ]


def transform_geometry(
    geometry: Union[Polygon, MultiPolygon, LineString, MultiLineString, Point],
    affine_params: List[float]
) -> Union[Polygon, MultiPolygon, LineString, MultiLineString, Point]:
    """
    Apply affine transformation to any geometry type.
    
    Args:
        geometry: Shapely geometry to transform
        affine_params: Affine transform parameters [a, b, d, e, xoff, yoff]
    
    Returns:
        Transformed geometry of the same type
    """
    return affine_transform(geometry, affine_params)


def transform_geometry_dict(
    geometry_dict: Dict[str, Any],
    affine_params: List[float]
) -> Dict[str, Any]:
    """
    Transform all geometries in a dictionary.
    
    Args:
        geometry_dict: Dictionary containing geometries
        affine_params: Affine transform parameters
    
    Returns:
        Dictionary with transformed geometries
    """
    transformed = {}
    
    for key, value in geometry_dict.items():
        if isinstance(value, dict):
            # Recursively transform nested dictionaries
            transformed[key] = transform_geometry_dict(value, affine_params)
        elif isinstance(value, list):
            # Transform lists of geometries
            transformed[key] = [
                transform_geometry(item, affine_params)
                if hasattr(item, 'geom_type') else item
                for item in value
            ]
        elif hasattr(value, 'geom_type'):
            # Transform single geometry
            transformed[key] = transform_geometry(value, affine_params)
        else:
            # Keep non-geometry values as-is
            transformed[key] = value
    
    return transformed


class CoordinateTransformer:
    """
    Manages coordinate transformations from PDF space to site space.
    """
    
    def __init__(self):
        self.site_layout_mls = None
        self.inverter_layout_mls = None
        self.pdf_to_inverter_params = {}  # Dict[page_num, affine_params]
        self.inverter_to_site_params = None
        self.pdf_to_site_params = {}  # Dict[page_num, affine_params]
    
    def set_site_layout(self, site_lines: Union[List[LineString], MultiLineString]):
        """Set the site layout reference (M-PLAN-TRACKER OUTLINE)."""
        if isinstance(site_lines, list):
            self.site_layout_mls = MultiLineString(site_lines)
        else:
            self.site_layout_mls = site_lines
    
    def set_inverter_layout(self, inverter_polygons: Dict[str, Polygon]):
        """Set the inverter layout from labeled polygons."""
        lines = []
        for poly in inverter_polygons.values():
            lines.append(list(poly.exterior.coords))
        self.inverter_layout_mls = MultiLineString(lines)
    
    def compute_inverter_to_site_transform(self, scale_mode: str = 'fit'):
        """
        Compute transformation from inverter space to site space.
        
        Args:
            scale_mode: 'area' or 'fit'
        
        Returns:
            Affine transform parameters
        """
        if self.site_layout_mls is None or self.inverter_layout_mls is None:
            raise ValueError("Both site and inverter layouts must be set")
        
        params, _ = align_multilinestrings_by_convex_hull(
            self.inverter_layout_mls,
            self.site_layout_mls,
            scale_mode=scale_mode
        )
        
        self.inverter_to_site_params = params
        return params
    
    def compute_page_to_inverter_transform(
        self,
        page_num: int,
        page_combiner_polygons: List[Polygon],
        inverter_polygon: Polygon,
        scale_mode: str = 'fit'
    ) -> List[float]:
        """
        Compute transformation from page space to inverter space.
        
        Args:
            page_num: Page number
            page_combiner_polygons: List of combiner polygons on the page
            inverter_polygon: Target inverter polygon
            scale_mode: 'area' or 'fit'
        
        Returns:
            Affine transform parameters
        """
        if not page_combiner_polygons:
            return [1, 0, 0, 1, 0, 0]  # Identity transform
        
        # Create MultiLineString from combiner polygons
        lines = [list(poly.exterior.coords) for poly in page_combiner_polygons]
        combiner_mls = MultiLineString(lines)
        
        # Create MultiLineString from inverter polygon
        inverter_mls = MultiLineString([list(inverter_polygon.exterior.coords)])
        
        # Compute transformation
        params, _ = align_multilinestrings_by_convex_hull(
            combiner_mls,
            inverter_mls,
            scale_mode=scale_mode
        )
        
        self.pdf_to_inverter_params[page_num] = params
        return params
    
    def compute_page_to_site_transform(self, page_num: int) -> List[float]:
        """
        Compute direct transformation from page space to site space.
        
        Args:
            page_num: Page number
        
        Returns:
            Affine transform parameters
        """
        if page_num not in self.pdf_to_inverter_params:
            raise ValueError(f"No PDF to inverter transform for page {page_num}")
        
        if self.inverter_to_site_params is None:
            raise ValueError("Inverter to site transform not computed")
        
        # Compose transformations: PDF -> Inverter -> Site
        params = compose_affine(
            self.pdf_to_inverter_params[page_num],
            self.inverter_to_site_params
        )
        
        self.pdf_to_site_params[page_num] = params
        return params
    
    def transform_to_site(
        self,
        geometry: Union[Polygon, MultiPolygon, LineString, MultiLineString, Point],
        page_num: int
    ) -> Union[Polygon, MultiPolygon, LineString, MultiLineString, Point]:
        """
        Transform geometry from page space to site space.
        
        Args:
            geometry: Geometry to transform
            page_num: Page number the geometry comes from
        
        Returns:
            Transformed geometry in site coordinates
        """
        if page_num not in self.pdf_to_site_params:
            self.compute_page_to_site_transform(page_num)
        
        return transform_geometry(geometry, self.pdf_to_site_params[page_num])
    
    def transform_page_data_to_site(
        self,
        page_data: Dict[str, Any],
        page_num: int
    ) -> Dict[str, Any]:
        """
        Transform all geometries in page data to site coordinates.
        
        Args:
            page_data: Dictionary containing page elements
            page_num: Page number
        
        Returns:
            Transformed page data
        """
        if page_num not in self.pdf_to_site_params:
            self.compute_page_to_site_transform(page_num)
        
        return transform_geometry_dict(
            page_data,
            self.pdf_to_site_params[page_num]
        )