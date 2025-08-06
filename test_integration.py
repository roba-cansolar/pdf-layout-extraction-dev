#!/usr/bin/env python3
"""
Integration test script for PDF polygon extraction pipeline

This script tests the core functionality to ensure everything works correctly.
"""

import os
import sys
import json
from pathlib import Path

# Add utilities to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utilities'))

try:
    from utilities.pdf_extraction_utils import (
        ExtractionConfig,
        load_pdf_page,
        analyze_layers,
        extract_polygons_from_layer,
        save_polygons_to_json
    )
    print("✅ Successfully imported all utility functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_pdf_loading():
    """Test PDF loading functionality"""
    print("\n🔍 Testing PDF loading...")
    
    pdf_dir = Path("docs/full_pdf")
    if not pdf_dir.exists():
        print("⚠️  PDF directory not found, skipping PDF loading test")
        return None
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠️  No PDF files found, skipping PDF loading test")
        return None
    
    # Test with first available PDF
    test_pdf = str(pdf_files[0])
    print(f"📄 Testing with: {os.path.basename(test_pdf)}")
    
    try:
        page, drawings = load_pdf_page(test_pdf, 0)
        print(f"✅ Successfully loaded page with {len(drawings)} drawing elements")
        return test_pdf, page, drawings
    except Exception as e:
        print(f"❌ PDF loading failed: {e}")
        return None


def test_layer_analysis(drawings):
    """Test layer analysis"""
    print("\n📊 Testing layer analysis...")
    
    try:
        layer_summary = analyze_layers(drawings)
        if not layer_summary.empty:
            print(f"✅ Found {len(layer_summary)} layers")
            print("📋 Available layers:")
            for layer in layer_summary.index[:5]:  # Show first 5 layers
                total_items = layer_summary.loc[layer].sum()
                print(f"   - {layer}: {total_items} items")
            return layer_summary.index.tolist()
        else:
            print("⚠️  No layers found in drawings")
            return []
    except Exception as e:
        print(f"❌ Layer analysis failed: {e}")
        return []


def test_polygon_extraction(pdf_path, layers):
    """Test polygon extraction"""
    print("\n🔺 Testing polygon extraction...")
    
    if not layers:
        print("⚠️  No layers available for extraction test")
        return False
    
    # Test with first available layer
    test_layer = layers[0]
    print(f"🎯 Testing extraction from layer: {test_layer}")
    
    try:
        config = ExtractionConfig()
        config.tolerance = 2.0  # Use higher tolerance for better connectivity
        
        result = extract_polygons_from_layer(pdf_path, 0, test_layer, config)
        
        if result['success']:
            print(f"✅ Successfully extracted {len(result['polygons'])} polygons")
            print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
            print(f"📐 Total area: {result['stats']['total_area']:.2f}")
            return result
        else:
            print(f"❌ Extraction failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Polygon extraction failed: {e}")
        return False


def test_json_export(result):
    """Test JSON export functionality"""
    print("\n💾 Testing JSON export...")
    
    if not result or not result['success']:
        print("⚠️  No successful extraction result to export")
        return False
    
    try:
        output_file = "test_polygons.json"
        save_polygons_to_json(result['polygons'], output_file)
        
        # Verify file was created and has content
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            print(f"✅ Successfully exported {len(data)} polygons to {output_file}")
            
            # Clean up test file
            os.remove(output_file)
            return True
        else:
            print("❌ Export file was not created")
            return False
            
    except Exception as e:
        print(f"❌ JSON export failed: {e}")
        return False


def test_config_validation():
    """Test configuration parameter validation"""
    print("\n⚙️  Testing configuration...")
    
    try:
        config = ExtractionConfig()
        
        # Test default values
        assert config.tolerance > 0, "Tolerance should be positive"
        assert config.extension_length > 0, "Extension length should be positive"
        assert config.min_polygon_area >= 0, "Min area should be non-negative"
        
        # Test parameter modification
        config.tolerance = 5.0
        config.extension_length = 100.0
        config.min_polygon_area = 25.0
        
        assert config.tolerance == 5.0, "Tolerance not set correctly"
        assert config.extension_length == 100.0, "Extension length not set correctly"
        assert config.min_polygon_area == 25.0, "Min area not set correctly"
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("🚀 Starting PDF Polygon Extraction Integration Tests")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Configuration
    test_results.append(test_config_validation())
    
    # Test 2: PDF Loading
    pdf_data = test_pdf_loading()
    if pdf_data:
        pdf_path, page, drawings = pdf_data
        test_results.append(True)
        
        # Test 3: Layer Analysis
        layers = test_layer_analysis(drawings)
        test_results.append(len(layers) > 0)
        
        # Test 4: Polygon Extraction
        extraction_result = test_polygon_extraction(pdf_path, layers)
        test_results.append(bool(extraction_result))
        
        # Test 5: JSON Export
        test_results.append(test_json_export(extraction_result))
        
    else:
        # Skip dependent tests if PDF loading failed
        test_results.extend([False, False, False, False])
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(test_results)}/{len(test_results)} tests")
    
    test_names = [
        "Configuration Validation",
        "PDF Loading", 
        "Layer Analysis",
        "Polygon Extraction",
        "JSON Export"
    ]
    
    for name, result in zip(test_names, test_results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {name}")
    
    if all(test_results):
        print("\n🎉 All tests passed! The pipeline is ready to use.")
        print("\n🚀 To get started:")
        print("   1. Run: streamlit run streamlit_app.py")
        print("   2. Or use CLI: python utilities/extract_polygons.py --help")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("💡 Common issues:")
        print("   - Make sure PDF files are in docs/full_pdf/ directory")
        print("   - Ensure PDFs contain vector drawings (not just images)")
        print("   - Try different layers if extraction fails")
    
    return all(test_results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)