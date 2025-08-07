import streamlit as st
import pickle
import json
import fitz
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
from shapely.geometry import Polygon, MultiLineString, mapping, shape
from shapely.affinity import affine_transform
from shapely.ops import polygonize
import pandas as pd
import os
import sys
import io
from pathlib import Path

# Add utilities to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import utilities
import utilities.pdf_extraction_utils as pdf_utils
from utilities.coordinate_transformation import align_multilinestrings_by_convex_hull, compose_affine

try:
    import transform_debug_helper as tdh
except ImportError:
    tdh = None

st.set_page_config(
    page_title="PDF Layout Extraction Troubleshooter",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'doc_json' not in st.session_state:
    st.session_state.doc_json = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = None
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'inverter_label_poly_dict' not in st.session_state:
    st.session_state.inverter_label_poly_dict = None

def load_pickle_file(file_path):
    """Load a pickle file and return its contents"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading pickle file: {e}")
        return None

def save_pickle_file(data, file_path):
    """Save data to a pickle file"""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        st.error(f"Error saving pickle file: {e}")
        return False

def plot_pdf_with_overlays(pdf_path, page_num, elements_dict, selected_layers):
    """Plot PDF page with element overlays"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Get page as image
    mat = fitz.Matrix(2, 2)  # 2x zoom
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 16))
    
    # Display PDF page
    from PIL import Image
    img = Image.open(io.BytesIO(img_data))
    ax.imshow(img)
    
    # Plot overlays
    colors = {
        'rack_outlines': 'red',
        'combiner_boxes': 'blue',
        'combiner_outlines': 'green',
        'inverter_boxed': 'orange',
        'ug_conduit': 'purple'
    }
    
    for layer_name in selected_layers:
        if layer_name in elements_dict:
            elements = elements_dict[layer_name]
            color = colors.get(layer_name, 'black')
            
            if isinstance(elements, list):
                for elem in elements:
                    if hasattr(elem, 'exterior'):
                        # Polygon
                        coords = list(elem.exterior.coords)
                        # Scale coordinates for display
                        scaled_coords = [(x*2, y*2) for x, y in coords]
                        poly_patch = MplPolygon(scaled_coords, fill=False, 
                                               edgecolor=color, linewidth=1, alpha=0.7)
                        ax.add_patch(poly_patch)
                    elif hasattr(elem, 'bounds'):
                        # Rectangle or other geometry
                        minx, miny, maxx, maxy = elem.bounds
                        rect = Rectangle((minx*2, miny*2), (maxx-minx)*2, (maxy-miny)*2,
                                       fill=False, edgecolor=color, linewidth=1, alpha=0.7)
                        ax.add_patch(rect)
    
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    ax.axis('off')
    
    doc.close()
    return fig

def plot_transform_debug(combiner_poly, inverter_outline, site_polygons, 
                        transformed_combiner=None, rack_outlines=None):
    """Create transform debugging visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot 1: Original combiner polygon
    ax = axes[0, 0]
    if hasattr(combiner_poly, 'geoms'):
        for line in combiner_poly.geoms:
            x, y = line.xy
            ax.plot(x, y, 'b-', linewidth=2)
    ax.set_title("Original Combiner Polygon")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Inverter outline
    ax = axes[0, 1]
    if inverter_outline:
        if hasattr(inverter_outline, 'exterior'):
            x, y = inverter_outline.exterior.xy
            ax.plot(x, y, 'r-', linewidth=2)
    ax.set_title("Inverter Outline")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Site polygons
    ax = axes[1, 0]
    for poly in site_polygons[:100]:  # Limit for performance
        if hasattr(poly, 'exterior'):
            x, y = poly.exterior.xy
            ax.plot(x, y, 'gray', linewidth=0.5, alpha=0.5)
    ax.set_title("Site Polygons")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Overlay of transformed elements
    ax = axes[1, 1]
    # Site polygons (background)
    for poly in site_polygons[:100]:
        if hasattr(poly, 'exterior'):
            x, y = poly.exterior.xy
            ax.plot(x, y, 'gray', linewidth=0.5, alpha=0.3)
    
    # Original combiner
    if hasattr(combiner_poly, 'geoms'):
        for line in combiner_poly.geoms:
            x, y = line.xy
            ax.plot(x, y, 'b:', linewidth=1, label='Original')
    
    # Transformed combiner
    if transformed_combiner and hasattr(transformed_combiner, 'geoms'):
        for line in transformed_combiner.geoms:
            x, y = line.xy
            ax.plot(x, y, 'g-', linewidth=2, label='Transformed')
    
    # Rack outlines if provided
    if rack_outlines:
        for poly in rack_outlines[:50]:  # Limit for performance
            if hasattr(poly, 'exterior'):
                x, y = poly.exterior.xy
                ax.plot(x, y, 'orange', linewidth=0.5, alpha=0.5)
    
    ax.set_title("Transform Overlay")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def explore_dict_structure(data, path=""):
    """Recursively explore dictionary structure"""
    items = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                items.append({
                    "Path": current_path,
                    "Type": "dict",
                    "Size": len(value),
                    "Keys": ", ".join(str(k) for k in list(value.keys())[:5])
                })
                items.extend(explore_dict_structure(value, current_path))
            elif isinstance(value, list):
                items.append({
                    "Path": current_path,
                    "Type": "list",
                    "Size": len(value),
                    "Keys": f"[{len(value)} items]"
                })
            else:
                value_type = type(value).__name__
                value_str = str(value)[:50] if not hasattr(value, 'geom_type') else value.geom_type
                items.append({
                    "Path": current_path,
                    "Type": value_type,
                    "Size": "-",
                    "Keys": value_str
                })
    
    return items

# Main UI
st.title("🔍 PDF Layout Extraction Troubleshooter")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # File loading section
    st.subheader("📁 Load Data")
    
    # Pickle file input
    pickle_file = st.file_uploader("Upload pickle file (.pkl)", type=['pkl'])
    if pickle_file:
        doc_json = pickle.load(pickle_file)
        st.session_state.doc_json = doc_json
        st.success(f"Loaded pickle with {len(doc_json)} pages")
    
    # Or load from path
    pickle_path = st.text_input("Or enter pickle file path:")
    if pickle_path and st.button("Load Pickle"):
        if os.path.exists(pickle_path):
            st.session_state.doc_json = load_pickle_file(pickle_path)
            if st.session_state.doc_json:
                st.success(f"Loaded {len(st.session_state.doc_json)} pages")
        else:
            st.error("File not found")
    
    # PDF file input
    st.subheader("📄 PDF File")
    pdf_path = st.text_input("PDF file path:", 
                             value="C:/Users/roba7/Documents/Programming projects/as-built-processing/pdf_layout_extraction_dev/docs/full_pdf/NorthStar As Built - Rev 2 2016-11-15.pdf")
    if pdf_path:
        st.session_state.pdf_path = pdf_path
    
    # Page selection
    if st.session_state.doc_json:
        st.subheader("📑 Page Selection")
        page_numbers = list(st.session_state.doc_json.keys())
        selected_page = st.selectbox("Select Page", page_numbers)
        st.session_state.current_page = selected_page

# Main content area with tabs
if st.session_state.doc_json and st.session_state.current_page:
    tabs = st.tabs(["📊 Data Explorer", "🖼️ PDF Visualization", "🔄 Transform Debug", 
                     "🏷️ Label Poly Dict", "⚙️ Run Pipeline", "📈 Statistics"])
    
    current_page_data = st.session_state.doc_json[st.session_state.current_page]
    
    # Tab 1: Data Explorer
    with tabs[0]:
        st.header("Data Structure Explorer")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Page Info")
            st.write(f"**Page Number:** {st.session_state.current_page}")
            if 'page_label' in current_page_data:
                st.write(f"**Page Label:** {current_page_data['page_label']}")
            
            st.subheader("Available Keys")
            for key in current_page_data.keys():
                st.write(f"- {key}")
        
        with col2:
            st.subheader("Structure Details")
            
            # Explore structure
            structure_df = pd.DataFrame(explore_dict_structure(current_page_data))
            st.dataframe(structure_df, use_container_width=True)
            
            # Raw data viewer
            st.subheader("Raw Data Viewer")
            selected_path = st.text_input("Enter path to explore (e.g., 'page_elements.rack_outlines'):")
            
            if selected_path:
                try:
                    # Navigate to the selected path
                    obj = current_page_data
                    for part in selected_path.split('.'):
                        if part.isdigit():
                            obj = obj[int(part)]
                        else:
                            obj = obj[part]
                    
                    if isinstance(obj, (dict, list)):
                        st.json(str(obj)[:1000])  # Limit display
                    else:
                        st.write(obj)
                except Exception as e:
                    st.error(f"Error accessing path: {e}")
    
    # Tab 2: PDF Visualization
    with tabs[1]:
        st.header("PDF Page with Overlays")
        
        if st.session_state.pdf_path and 'page_elements' in current_page_data:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.subheader("Select Layers")
                available_layers = list(current_page_data['page_elements'].keys())
                selected_layers = st.multiselect("Layers to display:", 
                                                available_layers, 
                                                default=available_layers[:3])
                
                show_plot = st.button("Generate Visualization")
            
            with col2:
                if show_plot:
                    with st.spinner("Generating visualization..."):
                        fig = plot_pdf_with_overlays(
                            st.session_state.pdf_path,
                            st.session_state.current_page,
                            current_page_data['page_elements'],
                            selected_layers
                        )
                        st.pyplot(fig)
                        plt.close()
    
    # Tab 3: Transform Debug
    with tabs[2]:
        st.header("Transform Debugging")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transform Parameters")
            
            # Show any affine parameters if available
            if 'affine_params' in current_page_data:
                st.write("**Affine Parameters:**")
                st.json(current_page_data['affine_params'])
        
        with col2:
            st.subheader("Visualization Controls")
            
            show_transform = st.button("Generate Transform Debug Plot")
            
            if show_transform and 'page_elements' in current_page_data:
                # Prepare data for transform debug plot
                combiner_outlines = current_page_data['page_elements'].get('combiner_outlines', [])
                rack_outlines = current_page_data['page_elements'].get('rack_outlines', [])
                
                # Create simple placeholder data if needed
                if combiner_outlines:
                    combiner_poly = MultiLineString([list(poly.exterior.coords) 
                                                    for poly in combiner_outlines 
                                                    if hasattr(poly, 'exterior')])
                else:
                    combiner_poly = None
                
                # Mock site polygons (would need actual site data)
                site_polygons = []
                
                fig = plot_transform_debug(
                    combiner_poly,
                    None,  # inverter_outline
                    site_polygons,
                    None,  # transformed_combiner
                    rack_outlines
                )
                st.pyplot(fig)
                plt.close()
    
    # Tab 4: Label Poly Dict Viewer
    with tabs[3]:
        st.header("Label Poly Dictionary Viewer")
        
        if 'combiner_label_poly_dict' in current_page_data:
            label_dict = current_page_data['combiner_label_poly_dict']
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Labels")
                labels = list(label_dict.keys())
                selected_label = st.selectbox("Select Label:", labels)
            
            with col2:
                if selected_label:
                    st.subheader(f"Details for: {selected_label}")
                    
                    poly = label_dict[selected_label]
                    
                    if hasattr(poly, 'bounds'):
                        bounds = poly.bounds
                        st.write(f"**Bounds:** {bounds}")
                        st.write(f"**Area:** {poly.area:.2f}")
                        st.write(f"**Perimeter:** {poly.length:.2f}")
                    
                    # Visualize the polygon
                    fig, ax = plt.subplots(figsize=(8, 8))
                    if hasattr(poly, 'exterior'):
                        x, y = poly.exterior.xy
                        ax.plot(x, y, 'b-', linewidth=2)
                        ax.fill(x, y, alpha=0.3)
                    ax.set_aspect('equal')
                    ax.grid(True, alpha=0.3)
                    ax.set_title(f"Polygon: {selected_label}")
                    st.pyplot(fig)
                    plt.close()
    
    # Tab 5: Run Pipeline
    with tabs[4]:
        st.header("Run Pipeline for Page")
        
        st.info("This section allows you to run the extraction pipeline for a specific page")
        
        col1, col2 = st.columns(2)
        
        with col1:
            page_to_process = st.number_input("Page Number:", 
                                             min_value=0, 
                                             value=14,
                                             step=1)
            
            layers_dict = {
                'rack_outlines': {'layer': "Xpanels - Northstar|E-PLAN-MCB", 'type': 'polygon', 'fill_lines': False},
                'combiner_boxes': {'layer': "E-CB", 'type': 'rectangle', 'fill_lines': True},
                'combiner_outlines': {'layer': "E-CB_AREA", 'type': 'polygon', 'fill_lines': True}
            }
            
            st.write("**Layers Configuration:**")
            st.json(layers_dict)
            
            run_pipeline = st.button("Run Extraction")
        
        with col2:
            if run_pipeline and st.session_state.pdf_path:
                with st.spinner(f"Processing page {page_to_process}..."):
                    try:
                        config = pdf_utils.ExtractionConfig()
                        config.extension_length = 40
                        
                        # Extract page elements
                        page_elements = pdf_utils.extract_page_elements(
                            st.session_state.pdf_path,
                            page_to_process,
                            config,
                            layers_dict
                        )
                        
                        st.success("Extraction complete!")
                        st.write(f"**Extracted elements:**")
                        for key, value in page_elements.items():
                            if isinstance(value, list):
                                st.write(f"- {key}: {len(value)} items")
                            else:
                                st.write(f"- {key}: {type(value).__name__}")
                        
                        # Store in session state
                        if page_to_process not in st.session_state.doc_json:
                            st.session_state.doc_json[page_to_process] = {}
                        st.session_state.doc_json[page_to_process]['page_elements'] = page_elements
                        
                    except Exception as e:
                        st.error(f"Error running pipeline: {e}")
    
    # Tab 6: Statistics
    with tabs[5]:
        st.header("Statistics & Analysis")
        
        if 'page_elements' in current_page_data:
            elements = current_page_data['page_elements']
            
            # Element counts
            st.subheader("Element Counts")
            counts_df = pd.DataFrame([
                {"Element Type": key, "Count": len(value) if isinstance(value, list) else 1}
                for key, value in elements.items()
            ])
            st.dataframe(counts_df, use_container_width=True)
            
            # Polygon statistics
            st.subheader("Polygon Statistics")
            
            for element_type in ['rack_outlines', 'combiner_outlines']:
                if element_type in elements:
                    polys = elements[element_type]
                    if polys and isinstance(polys, list):
                        areas = []
                        perimeters = []
                        
                        for poly in polys:
                            if hasattr(poly, 'area'):
                                areas.append(poly.area)
                                perimeters.append(poly.length)
                        
                        if areas:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(f"{element_type} - Mean Area", f"{np.mean(areas):.2f}")
                            with col2:
                                st.metric(f"{element_type} - Std Area", f"{np.std(areas):.2f}")
                            with col3:
                                st.metric(f"{element_type} - Count", len(areas))
                            
                            # Histogram
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            ax1.hist(areas, bins=30, edgecolor='black')
                            ax1.set_xlabel("Area")
                            ax1.set_ylabel("Count")
                            ax1.set_title(f"{element_type} - Area Distribution")
                            ax1.grid(True, alpha=0.3)
                            
                            ax2.hist(perimeters, bins=30, edgecolor='black')
                            ax2.set_xlabel("Perimeter")
                            ax2.set_ylabel("Count")
                            ax2.set_title(f"{element_type} - Perimeter Distribution")
                            ax2.grid(True, alpha=0.3)
                            
                            st.pyplot(fig)
                            plt.close()

else:
    st.info("👈 Please load a pickle file and select a page from the sidebar to begin troubleshooting")

# Footer
st.markdown("---")
st.markdown("PDF Layout Extraction Troubleshooter v1.0")