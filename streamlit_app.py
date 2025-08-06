"""
Interactive Streamlit App for PDF As-Built Drawing Polygon Extraction

This app provides an interactive interface for extracting polygons from 
PDF as-built drawings with real-time visualization and batch processing.
"""

import streamlit as st
import os
import sys
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from PIL import Image
import io
import zipfile
from typing import List, Dict, Any

# Add utilities to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utilities'))

from utilities.pdf_extraction_utils import (
    ExtractionConfig,
    load_pdf_page,
    analyze_layers,
    extract_polygons_from_layer,
    batch_extract_polygons,
    visualize_extraction_results,
    save_polygons_to_json,
    load_polygons_from_json
)

# Page configuration
st.set_page_config(
    page_title="PDF As-Built Polygon Extractor",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #2e8b57;
    border-bottom: 2px solid #2e8b57;
    padding-bottom: 0.5rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'polygons_cache' not in st.session_state:
        st.session_state.polygons_cache = {}
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = {}
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []

def get_pdf_files():
    """Get list of PDF files from docs/full_pdf directory"""
    pdf_dir = Path("docs/full_pdf")
    if not pdf_dir.exists():
        return []
    
    pdf_files = []
    for file in pdf_dir.glob("*.pdf"):
        pdf_files.append(str(file))
    
    return sorted(pdf_files)

def get_page_count(pdf_path: str) -> int:
    """Get number of pages in PDF"""
    try:
        import pymupdf
        doc = pymupdf.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except:
        return 0

@st.cache_data
def load_and_analyze_pdf_data(pdf_path: str, page_num: int):
    """Load PDF drawings and analyze layers with caching (serializable data only)"""
    try:
        page, drawings = load_pdf_page(pdf_path, page_num)
        layer_summary = analyze_layers(drawings)
        return drawings, layer_summary, None
    except Exception as e:
        return None, None, str(e)

def load_pdf_page_uncached(pdf_path: str, page_num: int):
    """Load PDF page without caching (for non-serializable page object)"""
    try:
        page, drawings = load_pdf_page(pdf_path, page_num)
        return page, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def extract_polygons_cached(pdf_path: str, page_num: int, layer_name: str, 
                           tolerance: float, extension_length: float, 
                           min_area: float, grid_size: float):
    """Extract polygons with caching (excluding non-serializable objects)"""
    config = ExtractionConfig()
    config.tolerance = tolerance
    config.extension_length = extension_length
    config.min_polygon_area = min_area
    config.grid_size = grid_size
    
    result = extract_polygons_from_layer(pdf_path, page_num, layer_name, config)
    
    # Remove non-serializable objects for caching
    if result['success']:
        # Convert Shapely objects to serializable format
        cached_result = result.copy()
        cached_result.pop('page', None)  # Remove page object
        
        # Convert polygons to WKT strings for serialization
        if 'polygons' in cached_result:
            cached_result['polygons_wkt'] = [poly.wkt for poly in cached_result['polygons']]
            cached_result.pop('polygons', None)
        
        # Convert lines and extensions to WKT strings  
        if 'lines' in cached_result:
            cached_result['lines_wkt'] = [line.wkt for line in cached_result['lines']]
            cached_result.pop('lines', None)
            
        if 'extensions' in cached_result:
            cached_result['extensions_wkt'] = [ext.wkt for ext in cached_result['extensions']]
            cached_result.pop('extensions', None)
            
        return cached_result
    
    return result

def extract_polygons_with_page(pdf_path: str, page_num: int, layer_name: str, 
                              tolerance: float, extension_length: float, 
                              min_area: float, grid_size: float):
    """Extract polygons including page object (not cached)"""
    # Get cached serializable data
    cached_result = extract_polygons_cached(pdf_path, page_num, layer_name, 
                                          tolerance, extension_length, min_area, grid_size)
    
    if not cached_result['success']:
        return cached_result
    
    # Load page object separately
    page, _ = load_pdf_page_uncached(pdf_path, page_num)
    
    # Reconstruct full result with geometry objects
    from shapely import wkt
    
    result = cached_result.copy()
    result['page'] = page
    
    # Convert WKT back to Shapely objects
    if 'polygons_wkt' in result:
        result['polygons'] = [wkt.loads(poly_wkt) for poly_wkt in result['polygons_wkt']]
        result.pop('polygons_wkt', None)
        
    if 'lines_wkt' in result:
        result['lines'] = [wkt.loads(line_wkt) for line_wkt in result['lines_wkt']]
        result.pop('lines_wkt', None)
        
    if 'extensions_wkt' in result:
        result['extensions'] = [wkt.loads(ext_wkt) for ext_wkt in result['extensions_wkt']]
        result.pop('extensions_wkt', None)
    
    return result

def create_layer_summary_plot(layer_summary: pd.DataFrame):
    """Create visualization of layer summary"""
    if layer_summary.empty:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Layer item counts
    layer_totals = layer_summary.sum(axis=1).sort_values(ascending=True)
    ax1.barh(range(len(layer_totals)), layer_totals.values)
    ax1.set_yticks(range(len(layer_totals)))
    ax1.set_yticklabels(layer_totals.index, fontsize=8)
    ax1.set_xlabel('Total Items')
    ax1.set_title('Items per Layer')
    
    # Item type distribution
    item_type_totals = layer_summary.sum(axis=0).sort_values(ascending=False)
    ax2.bar(range(len(item_type_totals)), item_type_totals.values)
    ax2.set_xticks(range(len(item_type_totals)))
    ax2.set_xticklabels(item_type_totals.index, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Total Items')
    ax2.set_title('Item Type Distribution')
    
    plt.tight_layout()
    return fig

def main():
    """Main Streamlit app"""
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">📐 PDF As-Built Polygon Extractor</h1>', 
                unsafe_allow_html=True)
    
    # Get PDF files for batch processing tab
    pdf_files = get_pdf_files()
    
    # Sidebar for configuration
    st.sidebar.markdown('<h2 class="section-header">⚙️ Configuration</h2>', 
                       unsafe_allow_html=True)
    
    # Processing parameters
    st.sidebar.subheader("Processing Parameters")
    tolerance = st.sidebar.slider("Endpoint Tolerance", 0.1, 10.0, 1.0, 0.1,
                                 help="Distance tolerance for clustering line endpoints")
    extension_length = st.sidebar.slider("Dangle Extension Length", 10.0, 200.0, 50.0, 5.0,
                                        help="Length to extend dangling lines")
    min_area = st.sidebar.slider("Minimum Polygon Area", 1.0, 100.0, 10.0, 1.0,
                                help="Minimum area for valid polygons")
    grid_size = st.sidebar.slider("Grid Size", 0.5, 10.0, 2.0, 0.5,
                                 help="Grid size for shapely operations (affects performance)")
    
    # Visualization options
    st.sidebar.subheader("Visualization Options")
    show_dangles = st.sidebar.checkbox("Show Dangles", True)
    show_extensions = st.sidebar.checkbox("Show Extensions", True)
    figure_dpi = st.sidebar.slider("Figure DPI", 72, 300, 150, 25)
    
    # Performance options
    st.sidebar.subheader("Performance")
    max_batch_size = st.sidebar.slider("Max Batch Size", 1, 20, 10, 1,
                                      help="Maximum number of extractions to process at once")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Single Page", "📊 Batch Processing", 
                                      "📈 Results Analysis", "⚡ Performance"])
    
    with tab1:
        st.markdown('<h2 class="section-header">Single Page Extraction</h2>', 
                   unsafe_allow_html=True)
        
        # Manual file input
        st.subheader("📁 File Input")
        
        # Help section
        with st.expander("ℹ️ How to use", expanded=False):
            st.markdown("""
            **Steps:**
            1. **Enter PDF path** - Type the path to your PDF file
            2. **Set page number** - Choose which page to analyze (0-indexed)
            3. **Click Load & Process** - This will load the page and show available layers
            4. **Select a layer** - Choose which layer to extract polygons from
            5. **Click Extract Polygons** - See the results overlaid on the page drawing
            
            **Example files:**
            - `docs/full_pdf/NorthStar As Built - Rev 2 2016-11-15.pdf` (page 14, layer "E-CB_AREA")
            - `docs/full_pdf/2024.06.18_SNJUAN_E_PV_AS-BUILT - Stamped.pdf` (page 0)
            """)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            # Text input for file path
            file_input = st.text_input(
                "PDF File Path", 
                value="docs/full_pdf/NorthStar As Built - Rev 2 2016-11-15.pdf",
                help="Enter the path to your PDF file"
            )
        
        with col2:
            page_num = st.number_input("Page Number", min_value=0, value=14, help="Page number (0-indexed)")
        
        with col3:
            process_button = st.button("🚀 Load & Process", type="primary")
        
        # Check if file exists
        if file_input and not os.path.exists(file_input):
            st.warning(f"File not found: {file_input}")
            st.info("Available files in docs/full_pdf/:")
            pdf_files = get_pdf_files()
            for pdf in pdf_files[:5]:
                st.text(f"  • {os.path.basename(pdf)}")
        
        # Process when button is clicked or load from session state
        current_key = f"{file_input}_{page_num}"
        
        # Check if we need to load new data
        if process_button and file_input and os.path.exists(file_input):
            with st.spinner("🔄 Loading PDF and analyzing layers..."):
                try:
                    # Load cached serializable data
                    drawings, layer_summary, error = load_and_analyze_pdf_data(file_input, page_num)
                    # Load page object separately (not cached due to serialization issues)
                    page, page_error = load_pdf_page_uncached(file_input, page_num)
                    if page_error and not error:
                        error = page_error
                    
                    # Store in session state
                    if not error:
                        st.session_state[f'loaded_data_{current_key}'] = {
                            'drawings': drawings,
                            'layer_summary': layer_summary,
                            'page': page,
                            'file_input': file_input,
                            'page_num': page_num
                        }
                        
                except Exception as e:
                    error = str(e)
                    drawings = None
                    layer_summary = None
                    page = None
            
            if error:
                st.error(f"Error loading PDF: {error}")
                return
        
        # Try to load from session state if available
        elif f'loaded_data_{current_key}' in st.session_state:
            loaded_data = st.session_state[f'loaded_data_{current_key}']
            drawings = loaded_data['drawings']
            layer_summary = loaded_data['layer_summary']
            page = loaded_data['page']
            error = None
        else:
            # No data loaded yet
            drawings = None
            layer_summary = None
            page = None
            error = None
        
        # Show analysis if data is loaded
        if drawings is not None and layer_summary is not None and not layer_summary.empty:
            
            # Display layer summary
            st.subheader("📊 Layer Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(layer_summary, use_container_width=True)
            
            with col2:
                total_layers = len(layer_summary)
                total_items = layer_summary.sum().sum()
                
                st.metric("Total Layers", total_layers)
                st.metric("Total Items", int(total_items))
            
            # Layer visualization
            layer_plot = create_layer_summary_plot(layer_summary)
            if layer_plot:
                st.pyplot(layer_plot)
            
            # Layer selection for extraction
            available_layers = layer_summary.index.tolist()
            selected_layer = st.selectbox("Select Layer to Extract", available_layers)
            
            if st.button("Extract Polygons", type="primary"):
                with st.spinner("🔺 Extracting polygons..."):
                    start_time = time.time()
                    
                    result = extract_polygons_with_page(
                        file_input, page_num, selected_layer,
                        tolerance, extension_length, min_area, grid_size
                    )
                    
                    processing_time = time.time() - start_time
                
                if result['success']:
                    st.success(f"Successfully extracted {len(result['polygons'])} polygons!")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Polygons Found", len(result['polygons']))
                    with col2:
                        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                    with col3:
                        st.metric("Total Area", f"{result['stats']['total_area']:.2f}")
                    with col4:
                        st.metric("Lines Processed", result['stats']['lines_extracted'])
                    
                    # Visualization - Show page with layer overlay
                    st.subheader("📄 Page Visualization")
                    
                    try:
                        # Create figure showing the actual page
                        fig, ax = plt.subplots(figsize=(15, 20))
                        
                        # Render the page to an image
                        pix = page.get_pixmap(dpi=150)
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Display the page image
                        ax.imshow(img, origin='upper')
                        ax.set_title(f"Page {page_num}: {os.path.basename(file_input)} - Layer: {selected_layer}", 
                                   fontsize=14, pad=20)
                        ax.axis('off')
                        
                        # Overlay the extracted polygons on the page
                        if result['polygons']:
                            colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'orange', 'purple']
                            
                            for i, poly in enumerate(result['polygons']):
                                if poly.exterior:
                                    x, y = poly.exterior.xy
                                    color = colors[i % len(colors)]
                                    ax.plot(y, x, color=color, linewidth=3, alpha=0.8)
                                    ax.fill(y, x, color=color, alpha=0.2)
                            
                            # Add legend
                            ax.text(0.02, 0.98, f"{len(result['polygons'])} polygons extracted", 
                                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    except Exception as viz_error:
                        st.error(f"Visualization error: {str(viz_error)}")
                        # Fallback to basic statistics
                        st.info("Showing statistics instead of visualization:")
                        stats_df = pd.DataFrame(list(result['stats'].items()), 
                                              columns=['Metric', 'Value'])
                        st.dataframe(stats_df)
                    
                    # Download options
                    st.subheader("Download Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # JSON download
                        json_data = {
                                'file': os.path.basename(file_input),
                                'page': page_num,
                                'layer': selected_layer,
                                'config': {
                                    'tolerance': tolerance,
                                    'extension_length': extension_length,
                                    'min_area': min_area,
                                    'grid_size': grid_size
                                },
                                'stats': result['stats'],
                                'polygons': [poly.wkt for poly in result['polygons']]
                        }
                        
                        st.download_button(
                            "Download as JSON",
                            json.dumps(json_data, indent=2),
                            file_name=f"{os.path.splitext(os.path.basename(file_input))[0]}_p{page_num}_{selected_layer}_polygons.json",
                            mime="application/json"
                        )
                    
                    with col2:
                        # Plot download
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=figure_dpi, bbox_inches='tight')
                        buf.seek(0)
                        
                        st.download_button(
                            "Download Plot",
                            buf.getvalue(),
                            file_name=f"{os.path.splitext(os.path.basename(file_input))[0]}_p{page_num}_{selected_layer}_plot.png",
                            mime="image/png"
                        )
                    
                    # Store in session state for analysis
                    cache_key = f"{file_input}_p{page_num}_{selected_layer}"
                    st.session_state.polygons_cache[cache_key] = result
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        'timestamp': time.time(),
                        'file': os.path.basename(file_input),
                        'page': page_num,
                        'layer': selected_layer,
                        'polygons_found': len(result['polygons']),
                        'processing_time': result['processing_time']
                    })
                    
                else:
                    st.error(f"Extraction failed: {result['error']}")
            
            else:
                st.warning("No layers found in the selected page")
                st.info("This could mean:")
                st.text("• The PDF page contains only images (no vector graphics)")
                st.text("• The page number is out of range")
                st.text("• The PDF file is corrupted or empty")
    
    with tab2:
        st.markdown('<h2 class="section-header">Batch Processing</h2>', 
                   unsafe_allow_html=True)
        
        st.info("Configure multiple extractions to run in batch mode for efficiency.")
        
        # Batch configuration
        st.subheader("Batch Configuration")
        
        # Option to upload batch config or configure manually
        config_method = st.radio("Configuration Method", 
                                ["Manual Setup", "Upload JSON Config"])
        
        if config_method == "Manual Setup":
            # Manual batch setup
            batch_entries = []
            
            st.subheader("Add Extraction Jobs")
            
            # Use session state to store batch entries
            if 'batch_entries' not in st.session_state:
                st.session_state.batch_entries = []
            
            # Add new entry
            with st.expander("Add New Extraction", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    batch_pdf = st.selectbox("PDF File", pdf_files, 
                                           key="batch_pdf",
                                           format_func=lambda x: os.path.basename(x))
                
                with col2:
                    batch_page_count = get_page_count(batch_pdf) if batch_pdf else 0
                    batch_page = st.number_input("Page", 0, max(0, batch_page_count-1), 0,
                                               key="batch_page")
                
                with col3:
                    # Load layers for selected PDF/page
                    if batch_pdf:
                        try:
                            _, batch_layer_summary, _ = load_and_analyze_pdf_data(batch_pdf, batch_page)
                            available_batch_layers = batch_layer_summary.index.tolist() if batch_layer_summary is not None and not batch_layer_summary.empty else []
                        except:
                            available_batch_layers = []
                    else:
                        available_batch_layers = []
                    
                    if available_batch_layers:
                        batch_layer = st.selectbox("Layer", available_batch_layers, 
                                                 key="batch_layer")
                    else:
                        batch_layer = st.text_input("Layer Name", key="batch_layer_manual")
                
                if st.button("Add to Batch", key="add_batch"):
                    if batch_pdf and batch_layer:
                        new_entry = {
                            'file': batch_pdf,
                            'page': batch_page,
                            'layer': batch_layer
                        }
                        st.session_state.batch_entries.append(new_entry)
                        st.success("Added to batch!")
                        st.experimental_rerun()
            
            # Display current batch
            if st.session_state.batch_entries:
                st.subheader("Current Batch")
                
                batch_df = pd.DataFrame(st.session_state.batch_entries)
                batch_df['File'] = batch_df['file'].apply(os.path.basename)
                
                # Allow editing/removing entries
                edited_df = st.data_editor(
                    batch_df[['File', 'page', 'layer']],
                    use_container_width=True,
                    key="batch_editor"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Clear Batch"):
                        st.session_state.batch_entries = []
                        st.experimental_rerun()
                
                with col2:
                    if st.button("Process Batch", type="primary", disabled=len(st.session_state.batch_entries) == 0):
                        process_batch(st.session_state.batch_entries, tolerance, 
                                    extension_length, min_area, grid_size, max_batch_size)
        
        else:
            # JSON config upload
            uploaded_config = st.file_uploader("Upload Batch Configuration", 
                                             type=['json'],
                                             help="Upload a JSON file with batch processing configuration")
            
            if uploaded_config:
                try:
                    config_data = json.load(uploaded_config)
                    
                    # Validate config
                    required_fields = ['files', 'pages', 'layers']
                    if all(field in config_data for field in required_fields):
                        st.success("Valid configuration uploaded!")
                        
                        # Show preview
                        preview_df = pd.DataFrame({
                            'File': [os.path.basename(f) for f in config_data['files']],
                            'Page': config_data['pages'],
                            'Layer': config_data['layers']
                        })
                        
                        st.subheader("Batch Preview")
                        st.dataframe(preview_df, use_container_width=True)
                        
                        if st.button("Process Uploaded Batch", type="primary"):
                            entries = [
                                {'file': f, 'page': p, 'layer': l}
                                for f, p, l in zip(config_data['files'], 
                                                 config_data['pages'], 
                                                 config_data['layers'])
                            ]
                            process_batch(entries, tolerance, extension_length, 
                                        min_area, grid_size, max_batch_size)
                    
                    else:
                        st.error(f"Invalid configuration. Required fields: {required_fields}")
                
                except json.JSONDecodeError:
                    st.error("Invalid JSON file")
    
    with tab3:
        st.markdown('<h2 class="section-header">Results Analysis</h2>', 
                   unsafe_allow_html=True)
        
        # Processing history
        if st.session_state.processing_history:
            st.subheader("Processing History")
            
            history_df = pd.DataFrame(st.session_state.processing_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], unit='s')
            
            st.dataframe(history_df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_extractions = len(history_df)
                st.metric("Total Extractions", total_extractions)
            
            with col2:
                total_polygons = history_df['polygons_found'].sum()
                st.metric("Total Polygons Found", total_polygons)
            
            with col3:
                avg_time = history_df['processing_time'].mean()
                st.metric("Avg Processing Time", f"{avg_time:.2f}s")
            
            # Visualizations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Processing time over time
            ax1.plot(history_df['timestamp'], history_df['processing_time'], 'o-')
            ax1.set_title('Processing Time Over Time')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Processing Time (s)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Polygons found distribution
            ax2.hist(history_df['polygons_found'], bins=10, alpha=0.7)
            ax2.set_title('Distribution of Polygons Found')
            ax2.set_xlabel('Number of Polygons')
            ax2.set_ylabel('Frequency')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        else:
            st.info("No processing history available. Run some extractions first!")
        
        # Cached results analysis
        if st.session_state.polygons_cache:
            st.subheader("Cached Results")
            
            cache_data = []
            for key, result in st.session_state.polygons_cache.items():
                cache_data.append({
                    'Key': key,
                    'Polygons': len(result['polygons']),
                    'Area': result['stats']['total_area'],
                    'Processing Time': f"{result['processing_time']:.2f}s"
                })
            
            cache_df = pd.DataFrame(cache_data)
            st.dataframe(cache_df, use_container_width=True)
            
            # Download all cached results
            if st.button("Download All Cached Results"):
                create_bulk_download()
    
    with tab4:
        st.markdown('<h2 class="section-header">Performance Optimization</h2>', 
                   unsafe_allow_html=True)
        
        st.info("Tips and tools for optimizing extraction performance")
        
        # Performance tips
        st.subheader("Performance Tips")
        
        tips = [
            "🎯 **Layer Selection**: Focus on specific layers rather than processing all layers",
            "⚡ **Grid Size**: Increase grid size (2-5) for faster shapely operations on complex drawings",
            "📏 **Tolerance**: Use larger tolerance values (2-5) for drawings with many small gaps",
            "🎛️ **Batch Processing**: Process multiple pages in batches for better efficiency",
            "💾 **Caching**: Results are automatically cached to avoid reprocessing",
            "🖥️ **Memory**: Close visualization windows when processing large batches"
        ]
        
        for tip in tips:
            st.markdown(tip)
        
        # Performance monitoring
        st.subheader("Performance Monitoring")
        
        if st.session_state.processing_history:
            history_df = pd.DataFrame(st.session_state.processing_history)
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fastest_time = history_df['processing_time'].min()
                st.metric("Fastest Extraction", f"{fastest_time:.2f}s")
            
            with col2:
                slowest_time = history_df['processing_time'].max()
                st.metric("Slowest Extraction", f"{slowest_time:.2f}s")
            
            with col3:
                efficiency = history_df['polygons_found'].sum() / history_df['processing_time'].sum()
                st.metric("Polygons per Second", f"{efficiency:.2f}")
        
        # System info
        st.subheader("System Information")
        
        try:
            import psutil
            
            col1, col2 = st.columns(2)
            
            with col1:
                cpu_percent = psutil.cpu_percent()
                st.metric("CPU Usage", f"{cpu_percent}%")
            
            with col2:
                memory = psutil.virtual_memory()
                st.metric("Memory Usage", f"{memory.percent}%")
        
        except ImportError:
            st.info("Install psutil for system monitoring: pip install psutil")
        
        # Cache management
        st.subheader("Cache Management")
        
        cache_size = len(st.session_state.polygons_cache)
        st.metric("Cached Results", cache_size)
        
        if st.button("Clear Cache"):
            st.session_state.polygons_cache = {}
            st.success("Cache cleared!")


def process_batch(entries: List[Dict], tolerance: float, extension_length: float, 
                 min_area: float, grid_size: float, max_batch_size: int):
    """Process batch of extractions"""
    
    st.subheader("Batch Processing Results")
    
    # Split into chunks if needed
    chunks = [entries[i:i + max_batch_size] for i in range(0, len(entries), max_batch_size)]
    
    all_results = {}
    total_polygons = []
    
    for chunk_idx, chunk in enumerate(chunks):
        st.write(f"Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} extractions)")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, entry in enumerate(chunk):
            status_text.text(f"Processing: {os.path.basename(entry['file'])} - Page {entry['page']} - Layer {entry['layer']}")
            
            config = ExtractionConfig()
            config.tolerance = tolerance
            config.extension_length = extension_length
            config.min_polygon_area = min_area
            config.grid_size = grid_size
            
            result = extract_polygons_from_layer(
                entry['file'], entry['page'], entry['layer'], config
            )
            
            key = f"{os.path.basename(entry['file'])}_p{entry['page']}_{entry['layer']}"
            all_results[key] = result
            
            if result['success']:
                total_polygons.extend(result['polygons'])
            
            progress_bar.progress((i + 1) / len(chunk))
        
        status_text.text("Chunk completed!")
    
    # Store batch results
    st.session_state.batch_results = all_results
    
    # Display summary
    successful = sum(1 for r in all_results.values() if r['success'])
    total_time = sum(r['processing_time'] for r in all_results.values())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Extractions", len(all_results))
    with col2:
        st.metric("Successful", successful)
    with col3:
        st.metric("Total Polygons", len(total_polygons))
    with col4:
        st.metric("Total Time", f"{total_time:.2f}s")
    
    # Results table
    results_data = []
    for key, result in all_results.items():
        results_data.append({
            'Key': key,
            'Success': result['success'],
            'Polygons': len(result['polygons']) if result['success'] else 0,
            'Time (s)': f"{result['processing_time']:.2f}",
            'Error': result.get('error', '') if not result['success'] else ''
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # Download batch results
    if total_polygons:
        batch_data = {
            'summary': {
                'total_extractions': len(all_results),
                'successful_extractions': successful,
                'total_polygons': len(total_polygons),
                'total_processing_time': total_time
            },
            'config': {
                'tolerance': tolerance,
                'extension_length': extension_length,
                'min_area': min_area,
                'grid_size': grid_size
            },
            'results': {key: {'success': r['success'], 
                            'polygons': [p.wkt for p in r['polygons']] if r['success'] else [],
                            'processing_time': r['processing_time']} 
                       for key, r in all_results.items()}
        }
        
        st.download_button(
            "Download Batch Results",
            json.dumps(batch_data, indent=2),
            file_name=f"batch_results_{int(time.time())}.json",
            mime="application/json"
        )


def create_bulk_download():
    """Create bulk download of all cached results"""
    
    if not st.session_state.polygons_cache:
        st.warning("No cached results to download")
        return
    
    # Create zip file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for key, result in st.session_state.polygons_cache.items():
            if result['success']:
                # Create JSON for each result
                json_data = {
                    'key': key,
                    'stats': result['stats'],
                    'processing_time': result['processing_time'],
                    'polygons': [poly.wkt for poly in result['polygons']]
                }
                
                zip_file.writestr(f"{key}.json", json.dumps(json_data, indent=2))
    
    zip_buffer.seek(0)
    
    st.download_button(
        "Download All Results (ZIP)",
        zip_buffer.getvalue(),
        file_name=f"all_polygon_results_{int(time.time())}.zip",
        mime="application/zip"
    )


if __name__ == "__main__":
    main()