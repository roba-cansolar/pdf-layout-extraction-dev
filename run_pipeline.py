"""
Simple Pipeline Runner for PDF to GeoJSON Extraction

This script provides an easy way to run the PDF extraction pipeline
with user-configurable parameters.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from testing_notebooks.dagster_pdf_pipeline_enhanced import (
    EnhancedPipelineConfig,
    site_layout,
    inverter_layout,
    coordinate_transformer,
    transformed_page_elements,
    aligned_rack_elements,
    final_geojson,
    pipeline_summary,
    GeminiAPIPool,
    CacheManager
)
from utilities.pdf_extraction_utils import ExtractionConfig
from utilities.coordinate_transformation import CoordinateTransformer

class PipelineRunner:
    """Simplified runner for the PDF extraction pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline runner with configuration
        
        Args:
            config: Dictionary containing pipeline configuration
        """
        self.config = self._validate_config(config)
        self.gemini_pool = None
        self.cache_manager = None
        self.results = {}
    
    def _validate_config(self, config: Dict[str, Any]) -> EnhancedPipelineConfig:
        """Validate and convert configuration to EnhancedPipelineConfig"""
        
        # Check required fields
        required = ['pdf_path', 'page_start', 'page_end']
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Create config object with defaults
        pipeline_config = EnhancedPipelineConfig(
            pdf_path=config['pdf_path'],
            page_start=config['page_start'],
            page_end=config['page_end']
        )
        
        # Override with user-provided values
        for key, value in config.items():
            if hasattr(pipeline_config, key):
                setattr(pipeline_config, key, value)
        
        return pipeline_config
    
    def setup_resources(self):
        """Initialize required resources"""
        
        print("Setting up resources...")
        
        # Initialize Gemini API
        if os.getenv('GEMINI_API_KEY'):
            self.gemini_pool = GeminiAPIPool()
            self.gemini_pool.setup_for_execution(None)
        else:
            print("Warning: GEMINI_API_KEY not set. Label extraction will be skipped.")
        
        # Initialize cache manager
        self.cache_manager = CacheManager(cache_dir=self.config.cache_dir)
        self.cache_manager.setup_for_execution(None)
        
        print("Resources initialized.")
    
    def run_extraction(self):
        """Run the main extraction pipeline"""
        
        print("\n" + "="*60)
        print("PDF TO GEOJSON EXTRACTION PIPELINE")
        print("="*60)
        
        # Display configuration
        print("\nConfiguration:")
        print(f"  PDF: {self.config.pdf_path}")
        print(f"  Pages: {self.config.page_start} to {self.config.page_end}")
        print(f"  Output: {self.config.output_dir}")
        print(f"  Scale Mode: {self.config.scale_mode}")
        print(f"  Rack Alignment: {'Enabled' if self.config.enable_rack_alignment else 'Disabled'}")
        
        # Setup resources
        self.setup_resources()
        
        # Step 1: Extract site layout
        print("\n[1/7] Extracting site layout...")
        self.results['site_layout'] = self._extract_site_layout()
        
        # Step 2: Extract inverter layout
        print("\n[2/7] Extracting inverter layout...")
        self.results['inverter_layout'] = self._extract_inverter_layout()
        
        # Step 3: Initialize coordinate transformer
        print("\n[3/7] Setting up coordinate transformation...")
        self.results['transformer'] = self._setup_transformer()
        
        # Step 4: Extract and transform page elements
        print("\n[4/7] Extracting page elements...")
        self.results['page_elements'] = self._extract_page_elements()
        
        # Step 5: Apply rack alignment (if enabled)
        if self.config.enable_rack_alignment:
            print("\n[5/7] Applying rack alignment...")
            self.results['aligned_elements'] = self._apply_rack_alignment()
        else:
            print("\n[5/7] Skipping rack alignment (disabled)")
            self.results['aligned_elements'] = self.results['page_elements']
        
        # Step 6: Generate GeoJSON
        print("\n[6/7] Generating GeoJSON output...")
        self.results['geojson'] = self._generate_geojson()
        
        # Step 7: Generate summary
        print("\n[7/7] Generating summary...")
        self.results['summary'] = self._generate_summary()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"\nOutput files:")
        print(f"  - {Path(self.config.output_dir) / 'transformed_layout.geojson'}")
        print(f"  - {Path(self.config.output_dir) / 'pipeline_summary.json'}")
    
    def _extract_site_layout(self) -> Dict[str, Any]:
        """Extract site layout"""
        from utilities.pdf_extraction_utils import load_pdf_page, filter_layer_paths, extract_lines_from_paths
        from shapely.geometry import MultiLineString
        from shapely.ops import polygonize
        
        page, drawings = load_pdf_page(self.config.pdf_path, self.config.site_layout_page)
        layer_paths = filter_layer_paths(drawings, self.config.site_layer)
        
        extraction_config = ExtractionConfig()
        extraction_config.extension_length = self.config.extension_length
        extraction_config.simplify_tolerance = self.config.simplify_tolerance
        
        lines = extract_lines_from_paths(layer_paths, extraction_config)
        mls = MultiLineString([list(line.coords) for line in lines])
        polygons = list(polygonize(lines))
        
        print(f"  Extracted {len(lines)} lines, {len(polygons)} polygons")
        
        return {
            "lines": lines,
            "multilinestring": mls,
            "polygons": polygons,
            "convex_hull": mls.convex_hull
        }
    
    def _extract_inverter_layout(self) -> Dict[str, Any]:
        """Extract inverter layout"""
        from utilities.pdf_extraction_utils import extract_polygons_from_layer, process_elements_on_page
        from shapely.geometry import MultiLineString
        
        extraction_config = ExtractionConfig()
        extraction_config.extension_length = 20
        
        result = extract_polygons_from_layer(
            self.config.pdf_path,
            self.config.overview_page,
            self.config.inverter_layer,
            extraction_config
        )
        
        if not result['success']:
            raise Exception(f"Failed to extract inverter layout: {result['error']}")
        
        # Process labels if API is available
        if self.gemini_pool:
            inverter_label_poly_dict, token_total = process_elements_on_page(
                self.config.pdf_path,
                self.config.overview_page,
                result['polygons'],
                "inverter",
                None
            )
            print(f"  Extracted {len(inverter_label_poly_dict)} labeled inverters")
        else:
            # Use dummy labels if API not available
            inverter_label_poly_dict = {
                f"INV_{i}": poly for i, poly in enumerate(result['polygons'])
            }
            print(f"  Extracted {len(inverter_label_poly_dict)} inverters (no labels)")
        
        lines = []
        for poly in inverter_label_poly_dict.values():
            lines.append(list(poly.exterior.coords))
        mls = MultiLineString(lines)
        
        raw_mls = MultiLineString(
            [list(line.coords) for line in result['lines']] +
            [list(line.coords) for line in result['extensions']]
        )
        
        return {
            "labeled_polygons": inverter_label_poly_dict,
            "multilinestring": mls,
            "raw_multilinestring": raw_mls,
            "polygons": result['polygons']
        }
    
    def _setup_transformer(self) -> CoordinateTransformer:
        """Setup coordinate transformer"""
        transformer = CoordinateTransformer()
        transformer.set_site_layout(self.results['site_layout']["multilinestring"])
        transformer.set_inverter_layout(self.results['inverter_layout']["labeled_polygons"])
        
        params = transformer.compute_inverter_to_site_transform(scale_mode=self.config.scale_mode)
        print(f"  Transform params: scale={params[0]:.4f}, dx={params[4]:.2f}, dy={params[5]:.2f}")
        
        return transformer
    
    def _extract_page_elements(self) -> Dict[int, Dict[str, Any]]:
        """Extract and transform page elements"""
        from utilities.pdf_extraction_utils import (
            extract_polygons_from_layer, 
            get_page_label_from_gemini,
            process_elements_on_page
        )
        import fitz
        
        layers_dict = {
            'rack_outlines': {'layer': self.config.rack_outline_layer, 'type': 'polygon'},
            'combiner_outlines': {'layer': self.config.combiner_layer, 'type': 'polygon'},
        }
        
        extraction_config = ExtractionConfig()
        extraction_config.extension_length = self.config.extension_length
        
        all_transformed_data = {}
        page_numbers = list(range(self.config.page_start, self.config.page_end))
        
        for i, page_num in enumerate(page_numbers):
            print(f"  Processing page {page_num} ({i+1}/{len(page_numbers)})", end='\r')
            
            # Extract page elements
            page_data = {}
            for layer_name, layer_info in layers_dict.items():
                result = extract_polygons_from_layer(
                    self.config.pdf_path,
                    page_num,
                    layer_info['layer'],
                    extraction_config
                )
                if result['success']:
                    page_data[layer_name] = result['polygons']
            
            # Get page label
            if self.gemini_pool:
                doc = fitz.open(self.config.pdf_path)
                page = doc[page_num]
                img_bytes = page.get_pixmap(dpi=300).tobytes("png")
                doc.close()
                page_label, _ = get_page_label_from_gemini(img_bytes)
            else:
                # Use page number as label
                page_label = f"Page_{page_num}"
            
            # Find matching inverter
            inverter_poly = None
            for inv_label, poly in self.results['inverter_layout']["labeled_polygons"].items():
                if inv_label == page_label or page_label in inv_label:
                    inverter_poly = poly
                    break
            
            if not inverter_poly and self.results['inverter_layout']["labeled_polygons"]:
                # Use first inverter as fallback
                inverter_poly = list(self.results['inverter_layout']["labeled_polygons"].values())[0]
            
            if inverter_poly and 'combiner_outlines' in page_data and page_data['combiner_outlines']:
                # Compute transformation
                self.results['transformer'].compute_page_to_inverter_transform(
                    page_num,
                    page_data['combiner_outlines'],
                    inverter_poly,
                    scale_mode=self.config.scale_mode
                )
                
                # Transform elements
                transformed_data = {}
                for element_type, polygons in page_data.items():
                    transformed_data[element_type] = [
                        self.results['transformer'].transform_to_site(poly, page_num)
                        for poly in polygons
                    ]
                
                all_transformed_data[page_num] = {
                    'page_label': page_label,
                    'elements': transformed_data,
                    'transform_params': self.results['transformer'].pdf_to_site_params.get(page_num)
                }
        
        print(f"\n  Processed {len(all_transformed_data)} pages")
        return all_transformed_data
    
    def _apply_rack_alignment(self) -> Dict[int, Dict[str, Any]]:
        """Apply rack alignment (stub for now)"""
        # This would use the align_racking module if available
        return self.results['page_elements']
    
    def _generate_geojson(self) -> Dict[str, Any]:
        """Generate GeoJSON output"""
        from shapely.geometry import mapping
        import time
        
        features = []
        
        # Add inverter blocks
        for label, poly in self.results['inverter_layout']["labeled_polygons"].items():
            features.append({
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "layer": "inverter",
                    "type": "block",
                    "label": label
                }
            })
        
        # Add transformed page elements
        for page_num, page_data in self.results['aligned_elements'].items():
            page_label = page_data.get('page_label', f"Page_{page_num}")
            
            for element_type, elements in page_data['elements'].items():
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
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "source_pdf": self.config.pdf_path,
                "pages_processed": list(self.results['aligned_elements'].keys()),
                "coordinate_system": "site",
                "scale_mode": self.config.scale_mode,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Save to file
        output_path = Path(self.config.output_dir) / "transformed_layout.geojson"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"  Generated {len(features)} features")
        return geojson
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate pipeline summary"""
        summary = {
            "configuration": {
                "pdf_path": self.config.pdf_path,
                "pages_processed": f"{self.config.page_start} to {self.config.page_end}",
                "site_layer": self.config.site_layer,
                "inverter_layer": self.config.inverter_layer,
                "scale_mode": self.config.scale_mode,
                "rack_alignment_enabled": self.config.enable_rack_alignment
            },
            "results": {
                "total_features": len(self.results['geojson']["features"]),
                "pages_processed": len(self.results['aligned_elements']),
                "inverters_found": len(self.results['inverter_layout']["labeled_polygons"])
            }
        }
        
        # Save summary
        summary_path = Path(self.config.output_dir) / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file"""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            return json.load(f)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PDF to GeoJSON Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python run_pipeline.py
  
  # Run with custom configuration file
  python run_pipeline.py --config my_config.json
  
  # Run with command-line overrides
  python run_pipeline.py --pdf path/to/file.pdf --pages 1 50 --output ./results
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (JSON or YAML)'
    )
    
    parser.add_argument(
        '--pdf',
        type=str,
        help='Path to input PDF file'
    )
    
    parser.add_argument(
        '--pages',
        type=int,
        nargs=2,
        metavar=('START', 'END'),
        help='Page range to process (start end)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--scale-mode',
        choices=['fit', 'area'],
        help='Coordinate transformation scale mode'
    )
    
    parser.add_argument(
        '--no-rack-alignment',
        action='store_true',
        help='Disable rack alignment'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )
    
    args = parser.parse_args()
    
    # Load base configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            'pdf_path': "C:/Users/roba7/Documents/Programming projects/as-built-processing/pdf_layout_extraction_dev/docs/full_pdf/NorthStar As Built - Rev 2 2016-11-15.pdf",
            'page_start': 13,
            'page_end': 63,
            'output_dir': './output',
            'scale_mode': 'fit',
            'enable_rack_alignment': True,
            'cache_enabled': True
        }
    
    # Apply command-line overrides
    if args.pdf:
        config['pdf_path'] = args.pdf
    
    if args.pages:
        config['page_start'] = args.pages[0]
        config['page_end'] = args.pages[1]
    
    if args.output:
        config['output_dir'] = args.output
    
    if args.scale_mode:
        config['scale_mode'] = args.scale_mode
    
    if args.no_rack_alignment:
        config['enable_rack_alignment'] = False
    
    if args.no_cache:
        config['cache_enabled'] = False
    
    # Run pipeline
    try:
        runner = PipelineRunner(config)
        runner.run_extraction()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()