#!/usr/bin/env python3
"""
Setup script for PDF As-Built Drawing Polygon Extractor

This script helps set up the environment and run initial tests.
"""

import os
import sys
import subprocess
from pathlib import Path


def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "uv", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_pdf_directory():
    """Check if PDF directory exists and has files"""
    print("📁 Checking PDF directory...")
    
    pdf_dir = Path("docs/full_pdf")
    
    if not pdf_dir.exists():
        print("⚠️  Creating PDF directory: docs/full_pdf/")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        print("📋 Please place your PDF files in the docs/full_pdf/ directory")
        return False
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found in docs/full_pdf/")
        print("📋 Please place your PDF files in the docs/full_pdf/ directory")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files[:5]:  # Show first 5
        print(f"   - {pdf.name}")
    
    if len(pdf_files) > 5:
        print(f"   ... and {len(pdf_files) - 5} more")
    
    return True


def run_integration_test():
    """Run integration tests"""
    print("🧪 Running integration tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_integration.py"], 
                              capture_output=True, text=True)
        
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False


def create_sample_config():
    """Create sample batch configuration"""
    print("📄 Creating sample configuration files...")
    
    # Sample batch config
    sample_batch = {
        "files": [
            "docs/full_pdf/sample1.pdf",
            "docs/full_pdf/sample2.pdf"
        ],
        "pages": [0, 0],
        "layers": ["E-CB_AREA", "E-CONDUIT"],
        "description": "Sample batch configuration - update with your actual files and layers"
    }
    
    try:
        import json
        with open("sample_batch_config.json", "w") as f:
            json.dump(sample_batch, f, indent=2)
        
        print("✅ Created sample_batch_config.json")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample config: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 PDF As-Built Drawing Polygon Extractor Setup")
    print("=" * 60)
    
    steps = [
        ("Installing Dependencies", install_dependencies),
        ("Checking PDF Directory", check_pdf_directory),
        ("Creating Sample Config", create_sample_config),
        ("Running Integration Tests", run_integration_test)
    ]
    
    results = []
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        results.append(step_func())
    
    print("\n" + "=" * 60)
    print("📊 Setup Summary:")
    
    for (step_name, _), result in zip(steps, results):
        status = "✅ COMPLETE" if result else "⚠️  NEEDS ATTENTION"
        print(f"   {status}: {step_name}")
    
    if all(results):
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("   1. Ensure your PDF files are in docs/full_pdf/")
        print("   2. Launch the web app: streamlit run streamlit_app.py")
        print("   3. Or use command line: python utilities/extract_polygons.py --help")
        
    else:
        print("\n⚠️  Setup completed with some issues.")
        print("💡 Check the messages above and resolve any issues before proceeding.")
        
        if not results[1]:  # PDF directory check failed
            print("\n📋 Important: Add PDF files to docs/full_pdf/ directory to test extraction")


if __name__ == "__main__":
    main()