import fitz  # PyMuPDF
import subprocess
import os
import re

def extract_pdf_page(input_pdf, page_number, output_pdf):
    doc = fitz.open(input_pdf)
    single_page = fitz.open()
    single_page.insert_pdf(doc, from_page=page_number, to_page=page_number)
    single_page.save(output_pdf)
    print(f"Saved single page to {output_pdf}")

def convert_to_svg_or_dxf(input_pdf, output_file, export_format="svg"):
    if export_format not in ["svg", "dxf"]:
        raise ValueError("export_format must be 'svg' or 'dxf'")
    
    print(f"Converting {input_pdf} to {output_file} in {export_format} format")
    
    # Get absolute paths to avoid confusion
    input_pdf_abs = os.path.abspath(input_pdf)
    output_file_abs = os.path.abspath(output_file)
    output_dir = os.path.dirname(output_file_abs)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Change to output directory to ensure files are created there
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    
    try:
        # Use the output filename directly with Inkscape
        # Add --batch-process to prevent GUI interactions
        cmd = [
            "inkscape",
            "--batch-process",
            "--export-type=" + export_format,
            "--export-filename=" + os.path.basename(output_file_abs),
            input_pdf_abs
        ]

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # 2 minute timeout

        if result.returncode == 0:
            if os.path.exists(os.path.basename(output_file_abs)):
                print(f"Successfully exported to {output_file}")
            else:
                print(f"Export command succeeded but file not found: {os.path.basename(output_file_abs)}")
        else:
            print("Inkscape failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print(f"Inkscape command timed out after 120 seconds")
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def run_pipeline(input_pdf, page_number, output_format="svg",output_dir="output"):
    # Create output directory if it doesn't exist
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    for page in page_number:
        base_name = os.path.splitext(os.path.basename(input_pdf))[0]
        base_name = re.sub(r'[^a-zA-Z0-9_]', '', base_name)  # Sanitize base_name
        temp_pdf = f"{output_dir}/{base_name}_page{page}.pdf"
        output_file = f"{output_dir}/{base_name}_page{page}.{output_format}"

        extract_pdf_page(input_pdf, page-1, temp_pdf)
        convert_to_svg_or_dxf(temp_pdf, output_file, export_format=output_format)

        # Optional: cleanup
        #os.remove(temp_pdf)

# Example usage:
if __name__ == "__main__":
    # run_pipeline("2024.09.11_Hornshadow_SV_IFC R3 (1).pdf", page_number=47, output_format="dxf")  # page_number is 0-indexed
    run_pipeline("../sample_drawings/NorthStar As Built - Rev 2 2016-11-15.pdf", page_number=[2,14,15], output_format="svg",output_dir="NorthStar_As_Built")
