# PDF As-Built Drawing Extraction Utilities

A comprehensive Python pipeline for extracting and labeling electrical components from PDF as-built drawings using geometric processing and AI-powered text recognition.

## Overview

This utility suite is designed specifically for processing electrical as-built drawings to:
- **Extract component boundaries** from CAD-generated PDF layers
- **Generate accurate polygon geometries** representing electrical equipment
- **Automatically label components** using AI vision models
- **Produce structured data** for downstream analysis and visualization

## System Architecture

The following diagram illustrates the complete processing pipeline from PDF input to labeled component data:

## Key Components

### Core Processing Modules

#### `ExtractionConfig`
Configuration class that controls all processing parameters:
- **Geometric tolerances**: Control precision vs. connectivity trade-offs
- **Performance settings**: Balance speed vs. quality for different use cases
- **Visualization options**: Customize output plots and analysis displays
- **AI parameters**: Control image clipping and context for labeling

#### `extract_polygons_from_layer()`
Main pipeline function orchestrating the complete geometric extraction workflow:
- Loads PDF pages and extracts vector drawing data
- Filters elements by CAD layer (e.g., "E-CB_AREA", "key plan|BEI-BLOCK-OUTLINE")
- Applies clip region filtering to focus on relevant drawing areas
- Converts vector paths to line geometries with topology improvement
- Generates final component boundary polygons

#### `process_elements_on_page()`
AI-powered labeling pipeline for component identification:
- Creates visual clips with highlighting around each polygon
- Integrates with Google Gemini AI for label extraction
- Handles different component types (inverters, combiners)
- Returns structured label-to-polygon mappings

### Processing Phases

#### Phase 1: PDF Parsing & Layer Analysis
```python
# Load PDF and analyze structure
page, drawings = load_pdf_page("drawing.pdf", page_number=14)
layer_stats = analyze_layers(drawings)
clip_map = build_clip_map(page)
```

#### Phase 2: Geometric Extraction
```python
# Extract component boundaries
config = ExtractionConfig()
result = extract_polygons_from_layer(
    file_path="drawing.pdf",
    page_number=14, 
    layer_name="E-CB_AREA",
    config=config
)
```

#### Phase 3: AI-Powered Labeling
```python
# Generate labeled components
if result['success']:
    labels, token_count = process_elements_on_page(
        file_path="drawing.pdf",
        page_number=14,
        result=result,
        element_type="combiner"
    )
```

## Installation

### Prerequisites
- Python 3.8+
- Google Gemini AI API key (for labeling functionality)

### Required Dependencies
```bash
pip install pymupdf shapely numpy matplotlib pandas pillow google-generativeai python-dotenv
```

### Environment Setup
Create a `.env` file in your project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## Quick Start

### Basic Polygon Extraction
```python
from utilities.pdf_extraction_utils import extract_polygons_from_layer, ExtractionConfig

# Configure extraction parameters
config = ExtractionConfig()
config.tolerance = 1.0          # Endpoint clustering tolerance
config.extension_length = 40.0  # Maximum dangle extension distance
config.min_polygon_area = 10.0  # Filter small artifacts

# Extract component boundaries
result = extract_polygons_from_layer(
    file_path="electrical_drawing.pdf",
    page_number=14,
    layer_name="E-CB_AREA",  # Combiner box layer
    config=config
)

if result['success']:
    print(f"Extracted {len(result['polygons'])} components")
    print(f"Processing time: {result['processing_time']:.2f}s")
else:
    print(f"Extraction failed: {result['error']}")
```

### Complete Labeling Workflow
```python
from utilities.pdf_extraction_utils import *

# Step 1: Extract inverter block layout (overview page)
config = ExtractionConfig()
config.extension_length = 20
inverter_result = extract_polygons_from_layer(
    "drawing.pdf", 14, "key plan|BEI-BLOCK-OUTLINE", config
)
inverter_labels, tokens = process_elements_on_page(
    "drawing.pdf", 14, inverter_result, "inverter"
)

# Step 2: Process multiple detail pages for combiners  
config.extension_length = 50
all_combiner_data = {}

for page_num in range(14, 20):
    # Get page classification
    page_img = fitz.open("drawing.pdf")[page_num].get_pixmap(dpi=300).tobytes("png")
    page_label, _ = get_page_label_from_gemini(page_img, [])
    
    # Extract combiner boxes
    result = extract_polygons_from_layer("drawing.pdf", page_num, "E-CB_AREA", config)
    if result['success']:
        combiner_labels, tokens = process_elements_on_page(
            "drawing.pdf", page_num, result, "combiner"
        )
        all_combiner_data[page_num] = {
            "page_label": page_label,
            "combiners": combiner_labels
        }

print(f"Processed {len(all_combiner_data)} pages with combiner data")
```

## Performance Optimization

### Current Performance Characteristics
- **Geometric extraction**: 0.1-2.0 seconds per page
- **AI labeling**: 2-10 seconds per polygon (API dependent)
- **Memory usage**: 50-200MB per page
- **Network dependency**: Required for AI labeling

### Speed Optimization Recommendations

#### 1. Geometric Processing Optimizations
- **Increase `grid_size`** (3.0 → 5.0+) for faster Shapely operations at cost of precision
- **Reduce `extension_length`** for simpler networks with fewer connection attempts
- **Pre-filter layers** before processing to reduce input data volume
- **Cache PDF parsing** results when processing multiple layers from same page

#### 2. AI Labeling Optimizations
- **Reduce image DPI** (100 → 72) for faster processing and lower token costs
- **Batch API requests** where possible to minimize network overhead
- **Implement smart caching** for similar polygon regions
- **Use parallel processing** for independent polygon labeling
- **Pre-filter polygons** by area/aspect ratio to skip obvious non-components

#### 3. Memory Optimization
- **Process pages incrementally** rather than loading entire documents
- **Clear intermediate results** after each page to prevent memory accumulation
- **Use streaming for large batch operations**
- **Implement polygon simplification** for storage efficiency

#### 4. Network & API Optimization
- **Implement exponential backoff** for API rate limiting
- **Use connection pooling** for multiple requests
- **Cache frequently used prompts** and responses
- **Monitor token usage** to optimize prompt efficiency

### Recommended High-Performance Configuration
```python
# Speed-optimized configuration
config = ExtractionConfig()
config.grid_size = 5.0          # Faster spatial operations
config.tolerance = 2.0          # More aggressive clustering
config.extension_length = 30.0  # Shorter extensions
config.min_polygon_area = 25.0  # Filter more aggressively
config.dpi = 72                 # Lower resolution for speed

# For AI labeling, reduce image quality
clip_dpi = 72  # vs. default 100
buffer_amount = 15  # vs. default 20
```

## Data Formats

### Extraction Results
```python
{
    'success': True,
    'page': <PyMuPDF Page>,
    'lines': [<LineString>, ...],          # Processed line network
    'extensions': [<LineString>, ...],      # Generated connections
    'polygons': [<Polygon>, ...],          # Component boundaries
    'layer_name': 'E-CB_AREA',
    'processing_time': 1.23,
    'stats': {
        'total_drawings': 450,
        'layer_paths': 23,
        'filtered_paths': 18,
        'lines_extracted': 67,
        'lines_snapped': 65,
        'extensions': 3,
        'polygons': 8,
        'total_area': 12450.7
    }
}
```

### Labeled Component Data
```python
{
    'label_1.2.3.C.1': <Polygon object>,
    'label_1.2.3.C.2': <Polygon object>,
    'label_2.1.1.C.1': <Polygon object>,
    # ... more labeled components
}
```

## Common Layer Names

### Electrical As-Built Drawings
- **`E-CB_AREA`**: Combiner box areas
- **`key plan|BEI-BLOCK-OUTLINE`**: Inverter block outlines
- **`EQUIPMENT`**: General electrical equipment
- **`E-CONDUIT`**: Electrical conduit routing
- **`E-WIRE`**: Wire and cable paths

### Tips for Layer Discovery
```python
# Analyze all layers in a drawing
page, drawings = load_pdf_page("drawing.pdf", 14)
layer_stats = analyze_layers(drawings)
print(layer_stats)  # Shows layer names and element counts
```

## Troubleshooting

### Common Issues

#### No Polygons Extracted
- **Check layer names**: Use `analyze_layers()` to verify correct layer names
- **Adjust tolerances**: Increase `tolerance` and `extension_length` for sparse drawings
- **Verify PDF quality**: Ensure PDF contains vector data, not just scanned images
- **Check clip regions**: Some drawings may have restrictive clip boundaries

#### Poor Polygon Quality
- **Reduce `min_polygon_area`**: May be filtering out valid small components
- **Increase `extension_length`**: Helps connect gaps in drawing networks
- **Adjust `tolerance`**: Balance between over-connecting and under-connecting
- **Inspect intermediate results**: Use visualization tools to debug processing steps

#### AI Labeling Failures
- **Verify API key**: Ensure `GEMINI_API_KEY` is correctly set in environment
- **Check network connectivity**: AI labeling requires stable internet connection
- **Monitor rate limits**: Gemini API has usage limits that may cause temporary failures
- **Inspect clipped images**: Ensure polygon clips contain readable text labels

#### Performance Issues
- **Reduce batch sizes**: Process fewer pages/polygons simultaneously
- **Lower image quality**: Reduce DPI settings for faster processing
- **Filter input data**: Remove unnecessary layers and elements before processing
- **Monitor memory usage**: Large drawings may require incremental processing

## API Reference

See inline documentation in `utilities/pdf_extraction_utils.py` for complete function signatures, parameters, and detailed usage examples.

## Contributing

This utility is designed for electrical as-built drawing processing. For other drawing types or industries, consider:
- Customizing the `ExtractionConfig` parameters
- Modifying AI prompts for different labeling schemes
- Adapting layer filtering logic for different CAD standards
- Extending the polygon processing pipeline for domain-specific requirements
