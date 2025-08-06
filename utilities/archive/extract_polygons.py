#!/usr/bin/env python3
"""
Main execution script for PDF polygon extraction pipeline

This script provides a command-line interface to extract polygons from 
PDF as-built drawings using the extraction utilities.
"""

import argparse
import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Add utilities to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_extraction_utils import (
    ExtractionConfig,
    load_pdf_page,
    analyze_layers,
    extract_polygons_from_layer,
    batch_extract_polygons,
    visualize_extraction_results,
    save_polygons_to_json,
    load_polygons_from_json
)


def main():
    parser = argparse.ArgumentParser(description='Extract polygons from PDF as-built drawings')
    
    # Input parameters
    parser.add_argument('pdf_file', help='Path to PDF file')
    parser.add_argument('--page', '-p', type=int, default=0, 
                       help='Page number to process (0-indexed, default: 0)')
    parser.add_argument('--layer', '-l', type=str, 
                       help='Layer name to extract (if not specified, will show available layers)')
    
    # Processing parameters
    parser.add_argument('--tolerance', '-t', type=float, default=1.0,
                       help='Tolerance for clustering endpoints (default: 1.0)')
    parser.add_argument('--extension-length', '-e', type=float, default=50.0,
                       help='Length for extending dangles (default: 50.0)')
    parser.add_argument('--min-area', '-a', type=float, default=10.0,
                       help='Minimum polygon area (default: 10.0)')
    
    # Output parameters
    parser.add_argument('--output', '-o', type=str, 
                       help='Output JSON file for polygons')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Show visualization of results')
    parser.add_argument('--save-plot', '-s', type=str,
                       help='Save plot to file')
    
    # Batch processing
    parser.add_argument('--batch-config', '-b', type=str,
                       help='JSON file with batch processing configuration')
    
    # Performance options
    parser.add_argument('--grid-size', '-g', type=float, default=2.0,
                       help='Grid size for shapely operations (default: 2.0)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.pdf_file):
        print(f"Error: PDF file '{args.pdf_file}' not found")
        return 1
    
    # Setup configuration
    config = ExtractionConfig()
    config.tolerance = args.tolerance
    config.extension_length = args.extension_length
    config.min_polygon_area = args.min_area
    config.grid_size = args.grid_size
    
    try:
        # Handle batch processing
        if args.batch_config:
            return process_batch(args.batch_config, config, args)
        
        # Load page and analyze layers
        print(f"Loading PDF: {args.pdf_file}, Page: {args.page}")
        page, drawings = load_pdf_page(args.pdf_file, args.page)
        
        print(f"Found {len(drawings)} drawing elements")
        
        # Analyze layers
        layer_summary = analyze_layers(drawings)
        
        if layer_summary.empty:
            print("No layers found in the drawings")
            return 1
        
        print("\nAvailable layers:")
        print("=" * 50)
        print(layer_summary)
        
        # If no layer specified, just show layers and exit
        if not args.layer:
            print("\nUse --layer to specify which layer to extract")
            return 0
        
        # Extract polygons
        print(f"\nExtracting polygons from layer: {args.layer}")
        result = extract_polygons_from_layer(args.pdf_file, args.page, args.layer, config)
        
        if not result['success']:
            print(f"Error: {result['error']}")
            return 1
        
        # Print results
        print("\nExtraction Results:")
        print("=" * 50)
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print(f"Polygons found: {len(result['polygons'])}")
        print(f"Total area: {result['stats']['total_area']:.2f}")
        
        print("\nDetailed statistics:")
        for key, value in result['stats'].items():
            print(f"  {key}: {value}")
        
        # Save polygons if requested
        if args.output:
            save_polygons_to_json(result['polygons'], args.output)
            print(f"\nPolygons saved to: {args.output}")
        
        # Visualize if requested
        if args.visualize or args.save_plot:
            print("\nGenerating visualization...")
            
            # Extract dangles for visualization (simplified)
            dangles = []  # Could be extracted from processing if needed
            
            fig = visualize_extraction_results(
                result['page'],
                result['lines'],
                result['extensions'],
                result['polygons'],
                dangles,
                args.layer,
                config
            )
            
            if args.save_plot:
                fig.savefig(args.save_plot, dpi=config.dpi, bbox_inches='tight')
                print(f"Plot saved to: {args.save_plot}")
            
            if args.visualize:
                plt.show()
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


def process_batch(batch_config_file: str, config: ExtractionConfig, args) -> int:
    """Process batch configuration file"""
    
    try:
        with open(batch_config_file, 'r') as f:
            batch_config = json.load(f)
    except Exception as e:
        print(f"Error reading batch config: {str(e)}")
        return 1
    
    # Validate batch config
    required_fields = ['files', 'pages', 'layers']
    for field in required_fields:
        if field not in batch_config:
            print(f"Error: Batch config missing required field: {field}")
            return 1
    
    files = batch_config['files']
    pages = batch_config['pages']
    layers = batch_config['layers']
    
    # Validate lengths
    if not (len(files) == len(pages) == len(layers)):
        print("Error: files, pages, and layers must have the same length")
        return 1
    
    print(f"Processing batch of {len(files)} extractions...")
    
    # Process batch
    results = batch_extract_polygons(files, pages, layers, config)
    
    print(f"\nBatch processing completed in {results['total_processing_time']:.2f} seconds")
    print(f"Successful extractions: {results['summary']['successful_extractions']}/{results['summary']['total_files']}")
    print(f"Total polygons found: {results['summary']['total_polygons_found']}")
    
    # Save batch results
    if args.output:
        output_data = {
            'config': {
                'tolerance': config.tolerance,
                'extension_length': config.extension_length,
                'min_polygon_area': config.min_polygon_area
            },
            'summary': results['summary'],
            'polygons': [poly.wkt for poly in results['total_polygons']]
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Batch results saved to: {args.output}")
    
    return 0


def create_sample_batch_config():
    """Create a sample batch configuration file"""
    
    sample_config = {
        "files": [
            "docs/full_pdf/sample1.pdf",
            "docs/full_pdf/sample2.pdf"
        ],
        "pages": [0, 1],
        "layers": ["E-CB_AREA", "E-UG_CONDUIT"],
        "description": "Sample batch configuration for processing multiple PDFs"
    }
    
    with open("sample_batch_config.json", "w") as f:
        json.dump(sample_config, f, indent=2)
    
    print("Sample batch configuration created: sample_batch_config.json")


if __name__ == "__main__":
    # If run with --create-sample, create sample config and exit
    if len(sys.argv) == 2 and sys.argv[1] == "--create-sample":
        create_sample_batch_config()
        sys.exit(0)
    
    sys.exit(main())