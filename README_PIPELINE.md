# PDF to GeoJSON Extraction Pipeline

A consolidated, performance-optimized pipeline for extracting layout information from PDF as-built drawings and transforming them into a unified site coordinate system.

## Overview

This pipeline extracts various elements from PDF drawings (inverters, combiners, rack outlines, etc.) and transforms them into a shared coordinate system defined by the site layout overlay layer ("Xpanels - Northstar|M-PLAN-TRACKER OUTLINE"). All outputs are exported as GeoJSON in the unified coordinate system.

## Key Features

- **Two-stage coordinate transformation**: PDF → Inverter Block → Site Coordinates
- **Parallel processing** for performance optimization
- **Intelligent caching** to speed up repeated runs
- **Fine-tuned rack alignment** (optional)
- **Comprehensive GeoJSON export** with all elements in unified coordinates
- **User-configurable parameters** via JSON configuration file
- **Both Dagster and standalone execution** modes

## Installation

1. Ensure all dependencies are installed:
```bash
pip install dagster shapely numpy fitz google-generativeai python-dotenv
```

2. Set up your Gemini API key (for label extraction):
```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Quick Start

### Method 1: Standalone Script (Recommended for Simple Use)

```bash
# Run with default configuration
python run_pipeline.py

# Run with custom configuration file
python run_pipeline.py --config pipeline_config.json

# Run with command-line overrides
python run_pipeline.py --pdf myfile.pdf --pages 1 50 --output ./results
```

### Method 2: Dagster Pipeline (For Production/Monitoring)

```bash
# Start Dagster UI
dagster dev -f testing_notebooks/dagster_pdf_pipeline_enhanced.py

# Navigate to http://localhost:3000 and trigger the pipeline
```

## Configuration

Edit `pipeline_config.json` to customize the pipeline:

```json
{
  "pdf_path": "path/to/your/pdf.pdf",
  "page_start": 13,
  "page_end": 63,
  
  "site_layer": "Xpanels - Northstar|M-PLAN-TRACKER OUTLINE",
  "inverter_layer": "key plan|BEI-BLOCK-OUTLINE",
  "combiner_layer": "E-CB_AREA",
  
  "scale_mode": "fit",  // 'fit' or 'area'
  "enable_rack_alignment": true,
  
  "output_dir": "./output"
}
```

### Key Configuration Parameters

- **pdf_path**: Path to input PDF file
- **page_start/page_end**: Range of pages to process
- **overview_page**: Page containing inverter layout overview (default: 14)
- **site_layout_page**: Page containing site layout reference (default: 1)
- **scale_mode**: 
  - `"fit"`: Scale to fit entirely within target bounds
  - `"area"`: Scale based on area ratio
- **enable_rack_alignment**: Apply fine-tuning for rack positions
- **max_workers**: Number of parallel workers for processing
- **cache_enabled**: Enable/disable caching for faster re-runs

## Pipeline Workflow

1. **Site Layout Extraction**: Extracts reference coordinate system from M-PLAN-TRACKER OUTLINE
2. **Inverter Layout Extraction**: Extracts and labels inverter blocks from overview page
3. **Coordinate System Setup**: Computes transformation from inverter space to site space
4. **Page Element Extraction**: Extracts elements from each page (combiners, racks, etc.)
5. **Coordinate Transformation**: Transforms all elements to unified site coordinates
6. **Rack Alignment** (optional): Fine-tunes rack positions using optimization
7. **GeoJSON Export**: Exports all elements as GeoJSON in site coordinates

## Output Files

The pipeline generates two main output files:

1. **transformed_layout.geojson**: Complete GeoJSON with all extracted elements
   - All coordinates in unified site coordinate system
   - Features tagged with source page and element type
   - Includes metadata about the transformation

2. **pipeline_summary.json**: Summary of the extraction process
   - Configuration used
   - Number of features extracted
   - Performance metrics

## Command-Line Options

```bash
python run_pipeline.py [OPTIONS]

Options:
  --config PATH           Path to configuration file (JSON or YAML)
  --pdf PATH             Override PDF file path
  --pages START END      Override page range
  --output PATH          Override output directory
  --scale-mode MODE      Override scale mode (fit/area)
  --no-rack-alignment    Disable rack alignment
  --no-cache             Disable caching
```

## Coordinate Transformation Details

The pipeline uses affine transformations to map coordinates:

1. **Page to Inverter**: Each page's combiner outlines are aligned to their corresponding inverter block
2. **Inverter to Site**: The full inverter layout is aligned to the site reference layout
3. **Composition**: The two transformations are composed for direct page-to-site mapping

Transformation parameters (scale, translation) are preserved in the output metadata.

## Performance Optimization

- **Parallel Processing**: Multiple pages processed simultaneously
- **Batch API Calls**: Labels extracted in batches to reduce latency
- **Caching**: Results cached to disk for faster re-runs
- **Vectorized Operations**: NumPy used for efficient geometric computations

## Troubleshooting

### Issue: "GEMINI_API_KEY not set"
**Solution**: Set the environment variable or add to .env file:
```bash
export GEMINI_API_KEY="your-key"
```

### Issue: Missing layers in PDF
**Solution**: Check layer names in configuration match your PDF's layer structure

### Issue: Transformation alignment issues
**Solution**: Try changing `scale_mode` from "fit" to "area" or adjusting `simplify_tolerance`

## API Usage

For programmatic usage:

```python
from run_pipeline import PipelineRunner

config = {
    'pdf_path': 'path/to/pdf.pdf',
    'page_start': 1,
    'page_end': 10,
    'output_dir': './results'
}

runner = PipelineRunner(config)
runner.run_extraction()

# Access results
geojson = runner.results['geojson']
```

## Development

To extend the pipeline:

1. Add new extraction layers in `EnhancedPipelineConfig`
2. Implement extraction logic in `transformed_page_elements` asset
3. Update GeoJSON generation in `final_geojson` asset

## License

This pipeline is part of the as-built-processing project.