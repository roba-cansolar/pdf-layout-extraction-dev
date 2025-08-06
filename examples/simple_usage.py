#!/usr/bin/env python3
"""
Simple usage examples for PDF polygon extraction

This script demonstrates basic usage patterns for the extraction utilities.
"""

import os
import sys
from pathlib import Path

# Add utilities to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utilities'))

from utilities.pdf_extraction_utils import (
    ExtractionConfig,
    load_pdf_page,
    analyze_layers,
    extract_polygons_from_layer,
    batch_extract_polygons,
    save_polygons_to_json
)


def example_1_basic_extraction():
    """Example 1: Basic single page extraction"""
    print("Example 1: Basic Single Page Extraction")
    print("-" * 40)
    
    # Find a PDF file to work with
    pdf_dir = Path("../docs/full_pdf")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found. Place PDFs in docs/full_pdf/ directory.")
        return
    
    pdf_path = str(pdf_files[0])
    print(f"Working with: {os.path.basename(pdf_path)}")
    
    # Load page and analyze layers
    page, drawings = load_pdf_page(pdf_path, 0)
    layer_summary = analyze_layers(drawings)
    
    print(f"Found {len(drawings)} drawing elements")
    print(f"Available layers: {list(layer_summary.index[:3])}")  # Show first 3
    
    # Extract polygons from first layer
    if not layer_summary.empty:
        layer_name = layer_summary.index[0]
        
        config = ExtractionConfig()
        config.tolerance = 2.0
        
        result = extract_polygons_from_layer(pdf_path, 0, layer_name, config)
        
        if result['success']:
            print(f"Extracted {len(result['polygons'])} polygons from layer '{layer_name}'")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
        else:
            print(f"Extraction failed: {result['error']}")


def example_2_parameter_tuning():
    """Example 2: Parameter tuning for better results"""
    print("\nExample 2: Parameter Tuning")
    print("-" * 40)
    
    pdf_dir = Path("../docs/full_pdf")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found.")
        return
    
    pdf_path = str(pdf_files[0])
    
    # Try different tolerance settings
    tolerances = [1.0, 2.0, 5.0]
    
    # Get a layer to test with
    page, drawings = load_pdf_page(pdf_path, 0)
    layer_summary = analyze_layers(drawings)
    
    if layer_summary.empty:
        print("No layers found.")
        return
    
    layer_name = layer_summary.index[0]
    
    print(f"Testing different tolerance values on layer '{layer_name}':")
    
    for tolerance in tolerances:
        config = ExtractionConfig()
        config.tolerance = tolerance
        config.extension_length = 50.0
        config.min_polygon_area = 10.0
        
        result = extract_polygons_from_layer(pdf_path, 0, layer_name, config)
        
        if result['success']:
            polygons_found = len(result['polygons'])
            processing_time = result['processing_time']
            print(f"  Tolerance {tolerance}: {polygons_found} polygons ({processing_time:.2f}s)")
        else:
            print(f"  Tolerance {tolerance}: Failed - {result['error']}")


def example_3_batch_processing():
    """Example 3: Batch processing multiple extractions"""
    print("\nExample 3: Batch Processing")
    print("-" * 40)
    
    pdf_dir = Path("../docs/full_pdf")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if len(pdf_files) < 2:
        print("Need at least 2 PDF files for batch example.")
        return
    
    # Setup batch job - process first 2 PDFs
    files = [str(pdf_files[0]), str(pdf_files[1])]
    pages = [0, 0]  # First page of each
    
    # Get layers from each file
    layers = []
    for pdf_file in files:
        try:
            page, drawings = load_pdf_page(pdf_file, 0)
            layer_summary = analyze_layers(drawings)
            if not layer_summary.empty:
                layers.append(layer_summary.index[0])  # First layer
            else:
                layers.append("Unknown")  # Fallback
        except:
            layers.append("Unknown")
    
    print(f"Batch processing {len(files)} files:")
    for i, (f, p, l) in enumerate(zip(files, pages, layers)):
        print(f"  {i+1}. {os.path.basename(f)}, page {p}, layer '{l}'")
    
    # Configure for batch
    config = ExtractionConfig()
    config.tolerance = 2.0
    config.extension_length = 50.0
    
    # Run batch
    results = batch_extract_polygons(files, pages, layers, config)
    
    print(f"\nBatch Results:")
    print(f"  Total processing time: {results['total_processing_time']:.2f} seconds")
    print(f"  Successful extractions: {results['summary']['successful_extractions']}")
    print(f"  Total polygons found: {results['summary']['total_polygons_found']}")


def example_4_save_and_load():
    """Example 4: Saving and loading results"""
    print("\nExample 4: Save and Load Results")
    print("-" * 40)
    
    pdf_dir = Path("../docs/full_pdf")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found.")
        return
    
    pdf_path = str(pdf_files[0])
    
    # Extract polygons
    page, drawings = load_pdf_page(pdf_path, 0)
    layer_summary = analyze_layers(drawings)
    
    if layer_summary.empty:
        print("No layers found.")
        return
    
    layer_name = layer_summary.index[0]
    config = ExtractionConfig()
    
    result = extract_polygons_from_layer(pdf_path, 0, layer_name, config)
    
    if result['success']:
        # Save to JSON
        output_file = "example_polygons.json"
        save_polygons_to_json(result['polygons'], output_file)
        print(f"Saved {len(result['polygons'])} polygons to {output_file}")
        
        # Load back and verify
        from utilities.pdf_extraction_utils import load_polygons_from_json
        loaded_polygons = load_polygons_from_json(output_file)
        print(f"Loaded {len(loaded_polygons)} polygons from file")
        
        # Clean up
        os.remove(output_file)
        print("Cleaned up example file")
    else:
        print(f"Extraction failed: {result['error']}")


def main():
    """Run all examples"""
    print("PDF Polygon Extraction - Usage Examples")
    print("=" * 50)
    
    examples = [
        example_1_basic_extraction,
        example_2_parameter_tuning,
        example_3_batch_processing,
        example_4_save_and_load
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Example failed: {e}")
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNext steps:")
    print("1. Try the Streamlit app: streamlit run streamlit_app.py")
    print("2. Use command line: python utilities/extract_polygons.py --help")
    print("3. Explore the utilities in utilities/pdf_extraction_utils.py")


if __name__ == "__main__":
    main()