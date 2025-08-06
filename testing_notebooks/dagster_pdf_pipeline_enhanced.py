"""
Enhanced Dagster PDF Processing Pipeline with Coordinate Transformation

This module extends the original pipeline with coordinate transformation capabilities
to translate all extracted elements into a shared site coordinate system defined by
the "Xpanels - Northstar|M-PLAN-TRACKER OUTLINE" layer.

Key features:
1. Site layout extraction from overlay layer
2. Two-stage coordinate transformation (PDF -> Inverter -> Site)
3. Fine-tuned rack alignment
4. Complete GeoJSON export in unified coordinate system
5. User-configurable parameters
"""

import os
import sys
import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass

import dagster
from dagster import (
    asset, 
    AssetIn, 
    Config, 
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    Definitions
)
from dagster import ConfigurableResource
import numpy as np
import fitz
from shapely.geometry import Polygon, Point, MultiLineString, LineString, mapping
from shapely.ops import polygonize
import shapely
import google.generativeai as genai
from dotenv import load_dotenv

# Add utilities to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.pdf_extraction_utils import (
    ExtractionConfig,
    load_pdf_page,
    extract_polygons_from_layer,
    process_elements_on_page,
    get_label_from_gemini,
    get_page_label_from_gemini,
    build_layout_geojson,
    clip_pdf_to_polygon,
    filter_layer_paths,
    extract_lines_from_paths,
    extract_page_elements
)
from utilities.coordinate_transformation import (
    CoordinateTransformer,
    align_multilinestrings_by_convex_hull,
    compose_affine,
    transform_geometry,
    transform_geometry_dict
)

# Load environment variables
load_dotenv()

# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================

class EnhancedPipelineConfig(Config):
    """Enhanced configuration for PDF processing with coordinate transformation"""
    
    # PDF and page settings
    pdf_path: str = "C:/Users/roba7/Documents/Programming projects/as-built-processing/pdf_layout_extraction_dev/docs/full_pdf/NorthStar As Built - Rev 2 2016-11-15.pdf"
    page_start: int = 14
    page_end: int = 15  # Set to 15 to process just page 14
    overview_page: int = 14  # Page with inverter layout overview
    site_layout_page: int = 1  # Page with site layout (M-PLAN-TRACKER OUTLINE)
    
    # Layer definitions
    site_layer: str = "Xpanels - Northstar|M-PLAN-TRACKER OUTLINE"
    inverter_layer: str = "key plan|BEI-BLOCK-OUTLINE"
    combiner_layer: str = "E-CB_AREA"
    rack_outline_layer: str = "Xpanels - Northstar|E-PLAN-MCB"
    
    # Coordinate transformation settings
    scale_mode: str = "fit"  # 'fit' or 'area'
    enable_rack_alignment: bool = True
    rack_alignment_iterations: int = 4
    
    # Performance settings
    max_workers: int = 8
    batch_size: int = 10
    cache_enabled: bool = True
    cache_dir: str = "./cache"
    
    # Extraction settings
    extension_length: float = 50.0
    simplify_tolerance: float = 0.02
    
    # API settings
    api_rate_limit: int = 60
    api_timeout: int = 30
    api_retry_count: int = 3
    
    # Output settings
    output_dir: str = "./output"
    output_format: str = "geojson"  # 'geojson' or 'json'
    include_debug_layers: bool = False  # Include intermediate transformation results

# ============================================================================
# RESOURCE DEFINITIONS (Reuse from original)
# ============================================================================

class GeminiAPIPool(ConfigurableResource):
    """Resource pool for managing Gemini API connections"""
    
    api_key: str = os.getenv('GEMINI_API_KEY')
    
    def get_model(self):
        """Get configured Gemini model"""
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel("gemini-2.5-flash-lite")

class CacheManager(ConfigurableResource):
    """Manages caching for extraction results"""
    
    cache_dir: str = "./cache"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

# ============================================================================
# NEW ASSETS FOR COORDINATE TRANSFORMATION
# ============================================================================

@asset(
    description="Extract site layout from M-PLAN-TRACKER OUTLINE layer"
)
def site_layout(
    context: AssetExecutionContext,
    config: EnhancedPipelineConfig
) -> Dict[str, Any]:
    """Extract the site layout that defines the target coordinate system"""
    
    start_time = time.time()
    
    # Load the page with site layout
    page, drawings = load_pdf_page(config.pdf_path, config.site_layout_page)
    
    # Extract lines from the site layer
    layer_paths = filter_layer_paths(drawings, config.site_layer)
    
    extraction_config = ExtractionConfig()
    extraction_config.extension_length = config.extension_length
    extraction_config.simplify_tolerance = config.simplify_tolerance
    
    lines = extract_lines_from_paths(layer_paths, extraction_config)
    
    # Create MultiLineString
    mls = MultiLineString([list(line.coords) for line in lines])
    
    # Also create polygons for visualization
    polygons = list(polygonize(lines))
    
    elapsed = time.time() - start_time
    context.log.info(f"Extracted site layout with {len(lines)} lines in {elapsed:.2f}s")
    
    context.add_output_metadata({
        "num_lines": len(lines),
        "num_polygons": len(polygons),
        "processing_time": elapsed
    })
    
    return {
        "lines": lines,
        "multilinestring": mls,
        "polygons": polygons,
        "convex_hull": mls.convex_hull
    }

@asset(
    description="Extract and label inverter blocks from overview page"
)
def inverter_layout(
    context: AssetExecutionContext,
    config: EnhancedPipelineConfig
) -> Dict[str, Any]:
    """Extract inverter layout with labels"""
    
    start_time = time.time()
    
    extraction_config = ExtractionConfig()
    extraction_config.extension_length = 20
    
    # Extract polygons from inverter layer
    result = extract_polygons_from_layer(
        config.pdf_path, 
        config.overview_page, 
        config.inverter_layer, 
        extraction_config
    )
    
    if not result['success']:
        raise Exception(f"Failed to extract inverter layout: {result['error']}")
    
    # Process labels if API key is available
    try:
        inverter_label_poly_dict, token_total = process_elements_on_page(
            config.pdf_path,
            config.overview_page,
            result['polygons'],
            "inverter",
            None
        )
    except Exception as e:
        context.log.warning(f"Could not extract labels: {e}. Using default labels.")
        # Use default labels if API fails
        inverter_label_poly_dict = {
            f"INV_{i+1}": poly for i, poly in enumerate(result['polygons'])
        }
        token_total = 0
    
    # Create MultiLineString for transformation
    lines = []
    for poly in inverter_label_poly_dict.values():
        lines.append(list(poly.exterior.coords))
    mls = MultiLineString(lines)
    
    # Also keep the raw extraction results for fine-tuning
    raw_mls = MultiLineString(
        [list(line.coords) for line in result['lines']] +
        [list(line.coords) for line in result['extensions']]
    )
    
    elapsed = time.time() - start_time
    context.log.info(f"Extracted {len(inverter_label_poly_dict)} labeled inverters in {elapsed:.2f}s")
    
    context.add_output_metadata({
        "num_inverters": len(inverter_label_poly_dict),
        "processing_time": elapsed,
        "tokens_used": token_total
    })
    
    return {
        "labeled_polygons": inverter_label_poly_dict,
        "multilinestring": mls,
        "raw_multilinestring": raw_mls,
        "polygons": result['polygons']
    }

@asset(
    description="Initialize coordinate transformer with site and inverter layouts"
)
def coordinate_transformer(
    context: AssetExecutionContext,
    config: EnhancedPipelineConfig,
    site_layout: Dict[str, Any],
    inverter_layout: Dict[str, Any]
) -> CoordinateTransformer:
    """Set up the coordinate transformation system"""
    
    transformer = CoordinateTransformer()
    
    # Set site layout
    transformer.set_site_layout(site_layout["multilinestring"])
    
    # Set inverter layout
    transformer.set_inverter_layout(inverter_layout["labeled_polygons"])
    
    # Compute inverter to site transformation
    params = transformer.compute_inverter_to_site_transform(scale_mode=config.scale_mode)
    
    context.log.info(f"Computed inverter to site transform: scale={params[0]:.4f}, dx={params[4]:.2f}, dy={params[5]:.2f}")
    
    context.add_output_metadata({
        "scale": params[0],
        "translation_x": params[4],
        "translation_y": params[5]
    })
    
    return transformer

@asset(
    description="Extract and transform page elements to site coordinates"
)
def transformed_page_elements(
    context: AssetExecutionContext,
    config: EnhancedPipelineConfig,
    coordinate_transformer: CoordinateTransformer,
    inverter_layout: Dict[str, Any]
) -> Dict[int, Dict[str, Any]]:
    """Extract page elements and transform them to site coordinates"""
    
    start_time = time.time()
    
    # Define layers to extract
    layers_dict = {
        'rack_outlines': {'layer': config.rack_outline_layer, 'type': 'polygon'},
        'combiner_boxes': {'layer': "E-CB", 'type': 'rectangle'},
        'combiner_outlines': {'layer': config.combiner_layer, 'type': 'polygon'},
        'ug_conduit': {'layer': "E-UG_CONDUIT", 'type': 'line'},
        'inverter_boxed': {'layer': "Xpanels - Northstar|E-Equipment", 'type': 'rectangle'}
    }
    
    extraction_config = ExtractionConfig()
    extraction_config.extension_length = config.extension_length
    
    all_transformed_data = {}
    page_numbers = list(range(config.page_start, config.page_end))
    
    for page_num in page_numbers:
        context.log.info(f"Processing page {page_num}")
        
        # Extract page elements
        page_data = {}
        for layer_name, layer_info in layers_dict.items():
            result = extract_polygons_from_layer(
                config.pdf_path,
                page_num,
                layer_info['layer'],
                extraction_config
            )
            if result['success']:
                page_data[layer_name] = result['polygons']
        
        # Get page label
        try:
            doc = fitz.open(config.pdf_path)
            page = doc[page_num]
            img_bytes = page.get_pixmap(dpi=300).tobytes("png")
            doc.close()
            page_label, _ = get_page_label_from_gemini(img_bytes)
        except Exception as e:
            context.log.warning(f"Could not extract page label for page {page_num}: {e}")
            page_label = f"Page_{page_num}"
        
        if page_label and page_label in inverter_layout["labeled_polygons"]:
            # Get the target inverter polygon
            inverter_poly = inverter_layout["labeled_polygons"][page_label]
            
            # Compute page to inverter transformation using combiner outlines
            if 'combiner_outlines' in page_data and page_data['combiner_outlines']:
                coordinate_transformer.compute_page_to_inverter_transform(
                    page_num,
                    page_data['combiner_outlines'],
                    inverter_poly,
                    scale_mode=config.scale_mode
                )
                
                # Transform all page elements to site coordinates
                transformed_data = {}
                for element_type, polygons in page_data.items():
                    transformed_data[element_type] = [
                        coordinate_transformer.transform_to_site(poly, page_num)
                        for poly in polygons
                    ]
                
                # Also transform combiner labels if needed
                if 'combiner_outlines' in page_data:
                    # Process combiner labels
                    try:
                        combiner_labels, _ = process_elements_on_page(
                            config.pdf_path,
                            page_num,
                            page_data['combiner_outlines'],
                            "combiner",
                            None
                        )
                    except Exception as e:
                        context.log.warning(f"Could not extract combiner labels: {e}")
                        combiner_labels = {
                            f"CB_{i+1}": poly for i, poly in enumerate(page_data['combiner_outlines'])
                        }
                    
                    # Transform labeled combiners
                    transformed_combiner_labels = {}
                    for label, poly in combiner_labels.items():
                        transformed_poly = coordinate_transformer.transform_to_site(poly, page_num)
                        transformed_combiner_labels[label] = transformed_poly
                    
                    transformed_data['combiner_labels'] = transformed_combiner_labels
                
                all_transformed_data[page_num] = {
                    'page_label': page_label,
                    'elements': transformed_data,
                    'transform_params': coordinate_transformer.pdf_to_site_params[page_num]
                }
    
    elapsed = time.time() - start_time
    context.log.info(f"Processed {len(all_transformed_data)} pages in {elapsed:.2f}s")
    
    context.add_output_metadata({
        "num_pages": len(all_transformed_data),
        "processing_time": elapsed
    })
    
    return all_transformed_data

@asset(
    description="Apply fine-tuned rack alignment if enabled"
)
def aligned_rack_elements(
    context: AssetExecutionContext,
    config: EnhancedPipelineConfig,
    transformed_page_elements: Dict[int, Dict[str, Any]],
    site_layout: Dict[str, Any]
) -> Dict[int, Dict[str, Any]]:
    """Apply fine-tuned rack alignment to transformed elements"""
    
    if not config.enable_rack_alignment:
        context.log.info("Rack alignment disabled, returning transformed elements as-is")
        return transformed_page_elements
    
    start_time = time.time()
    
    # Import alignment functions (optional, only if available)
    try:
        from align_racking import align_racking_affine_lines, Options
        
        opts = Options(
            aoi_buffer_in_row_pitch=1.5,
            neighbor_max_in_row_pitch=1.2,
            source_line_buffer_abs=None,
            simplify_tolerance=config.simplify_tolerance,
            iters=config.rack_alignment_iterations
        )
        
        aligned_data = {}
        
        for page_num, page_data in transformed_page_elements.items():
            if 'rack_outlines' in page_data['elements'] and 'combiner_outlines' in page_data['elements']:
                # Convert to MultiLineString for alignment
                rack_mls = MultiLineString([
                    list(poly.exterior.coords) 
                    for poly in page_data['elements']['rack_outlines']
                ])
                
                combiner_mls = MultiLineString([
                    list(poly.exterior.coords)
                    for poly in page_data['elements']['combiner_outlines']
                ])
                
                # Apply alignment
                params, aligned_racks, aoi = align_racking_affine_lines(
                    combiner_mls,
                    site_layout['polygons'],
                    rack_mls,
                    opts
                )
                
                # Convert back to polygons
                aligned_rack_polygons = list(polygonize(aligned_racks))
                
                # Update the page data
                page_data_copy = dict(page_data)
                page_data_copy['elements']['rack_outlines_aligned'] = aligned_rack_polygons
                page_data_copy['alignment_params'] = params
                
                aligned_data[page_num] = page_data_copy
            else:
                aligned_data[page_num] = page_data
        
        elapsed = time.time() - start_time
        context.log.info(f"Applied rack alignment in {elapsed:.2f}s")
        
    except ImportError:
        context.log.warning("Rack alignment module not available, skipping alignment")
        aligned_data = transformed_page_elements
        elapsed = 0
    
    context.add_output_metadata({
        "alignment_applied": config.enable_rack_alignment,
        "processing_time": elapsed
    })
    
    return aligned_data

@asset(
    description="Generate final GeoJSON with all transformed elements"
)
def final_geojson(
    context: AssetExecutionContext,
    config: EnhancedPipelineConfig,
    site_layout: Dict[str, Any],
    inverter_layout: Dict[str, Any],
    aligned_rack_elements: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate comprehensive GeoJSON output with all elements in site coordinates"""
    
    start_time = time.time()
    
    features = []
    
    # Add site layout as reference
    if config.include_debug_layers:
        features.append({
            "type": "Feature",
            "geometry": mapping(site_layout["convex_hull"]),
            "properties": {
                "layer": "site_layout",
                "type": "reference",
                "description": "Site layout convex hull (M-PLAN-TRACKER OUTLINE)"
            }
        })
        
        # Add site polygons
        for i, poly in enumerate(site_layout["polygons"]):
            features.append({
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "layer": "site_layout",
                    "type": "polygon",
                    "index": i
                }
            })
    
    # Add inverter blocks
    for label, poly in inverter_layout["labeled_polygons"].items():
        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "layer": "inverter",
                "type": "block",
                "label": label
            }
        })
    
    # Add all transformed page elements
    for page_num, page_data in aligned_rack_elements.items():
        page_label = page_data.get('page_label', f"Page_{page_num}")
        
        for element_type, elements in page_data['elements'].items():
            if element_type == 'combiner_labels':
                # Handle labeled combiners
                for label, poly in elements.items():
                    features.append({
                        "type": "Feature",
                        "geometry": mapping(poly),
                        "properties": {
                            "layer": "combiner",
                            "type": "labeled",
                            "label": label,
                            "page": page_num,
                            "inverter_block": page_label
                        }
                    })
            elif isinstance(elements, list):
                # Handle lists of geometries
                for i, geom in enumerate(elements):
                    features.append({
                        "type": "Feature",
                        "geometry": mapping(geom),
                        "properties": {
                            "layer": element_type,
                            "type": "geometry",
                            "index": i,
                            "page": page_num,
                            "inverter_block": page_label
                        }
                    })
    
    # Create final GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "source_pdf": config.pdf_path,
            "pages_processed": list(aligned_rack_elements.keys()),
            "coordinate_system": "site",
            "scale_mode": config.scale_mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Save to file
    output_path = Path(config.output_dir) / "transformed_layout.geojson"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    elapsed = time.time() - start_time
    
    context.log.info(f"Generated GeoJSON with {len(features)} features in {elapsed:.2f}s")
    context.log.info(f"Output saved to: {output_path}")
    
    context.add_output_metadata({
        "num_features": len(features),
        "output_path": str(output_path),
        "processing_time": elapsed
    })
    
    return geojson

# ============================================================================
# PIPELINE SUMMARY ASSET
# ============================================================================

@asset(
    description="Generate pipeline execution summary"
)
def pipeline_summary(
    context: AssetExecutionContext,
    config: EnhancedPipelineConfig,
    final_geojson: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate comprehensive summary of the pipeline execution"""
    
    summary = {
        "configuration": {
            "pdf_path": config.pdf_path,
            "pages_processed": f"{config.page_start} to {config.page_end}",
            "site_layer": config.site_layer,
            "inverter_layer": config.inverter_layer,
            "scale_mode": config.scale_mode,
            "rack_alignment_enabled": config.enable_rack_alignment
        },
        "results": {
            "total_features": len(final_geojson["features"]),
            "feature_breakdown": {}
        },
        "performance": {
            "max_workers": config.max_workers,
            "batch_size": config.batch_size,
            "cache_enabled": config.cache_enabled
        }
    }
    
    # Count features by type
    for feature in final_geojson["features"]:
        layer = feature["properties"].get("layer", "unknown")
        if layer not in summary["results"]["feature_breakdown"]:
            summary["results"]["feature_breakdown"][layer] = 0
        summary["results"]["feature_breakdown"][layer] += 1
    
    context.log.info("Pipeline execution complete")
    context.log.info(f"Summary: {json.dumps(summary, indent=2)}")
    
    # Save summary
    summary_path = Path(config.output_dir) / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    context.add_output_metadata({
        "summary_path": str(summary_path)
    })
    
    return summary

# ============================================================================
# DAGSTER DEFINITIONS
# ============================================================================

defs = Definitions(
    assets=[
        site_layout,
        inverter_layout,
        coordinate_transformer,
        transformed_page_elements,
        aligned_rack_elements,
        final_geojson,
        pipeline_summary
    ],
    resources={
        "cache_manager": CacheManager(),
        "config": EnhancedPipelineConfig()
    }
)

if __name__ == "__main__":
    print("Enhanced Dagster PDF Processing Pipeline")
    print("=========================================")
    print("Features:")
    print("1. Site layout extraction from M-PLAN-TRACKER OUTLINE")
    print("2. Two-stage coordinate transformation (PDF -> Inverter -> Site)")
    print("3. Parallel page processing with caching")
    print("4. Optional fine-tuned rack alignment")
    print("5. Comprehensive GeoJSON export in unified coordinates")
    print("")
    print("User-configurable parameters:")
    print("- PDF path and page range")
    print("- Layer names for extraction")
    print("- Coordinate transformation mode ('fit' or 'area')")
    print("- Rack alignment settings")
    print("- Performance tuning (workers, batch size, caching)")
    print("- Output format and location")
    print("")
    print("To run the pipeline:")
    print("  dagster dev -f dagster_pdf_pipeline_enhanced.py")
    print("")
    print("To configure: Edit EnhancedPipelineConfig class or use Dagster UI")