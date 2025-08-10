#!/usr/bin/env python3
"""
Specialized installation script for the Market & Social Insights Dashboard
Handles numpy installation issues on Windows with Python 3.12
"""

import os
import sys
import subprocess
import platform

def check_system():
    """Check system information"""
    print(f"🖥️  System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version}")
    print(f"📦 pip: {subprocess.check_output([sys.executable, '-m', 'pip', '--version']).decode().strip()}")
    return True

def install_numpy_first():
    """Install numpy first to avoid conflicts"""
    print("📦 Installing numpy first...")
    try:
        # Try to install numpy with specific version for Python 3.12
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "numpy>=1.26.0", "--upgrade"
        ])
        print("✅ numpy installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing numpy: {e}")
        print("🔄 Trying alternative installation method...")
        
        try:
            # Try with --no-deps to avoid conflicts
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "numpy", "--no-deps", "--force-reinstall"
            ])
            print("✅ numpy installed with --no-deps!")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ Alternative method failed: {e2}")
            return False

def install_other_dependencies():
    """Install other dependencies"""
    print("📦 Installing other dependencies...")
    
    # Install packages one by one to avoid conflicts
    packages = [
        "pandas>=2.0.0",
        "streamlit>=1.28.0",
        "plotly>=5.0.0",
        "yfinance>=0.2.0",
        "newsapi-python>=0.2.6",
        "google-api-python-client>=2.100.0",
        "openai>=1.0.0",
        "google-generativeai>=0.3.0",
        "vaderSentiment>=3.3.0",
        "textblob>=0.17.0",
        "requests>=2.25.0",
        "python-dotenv>=1.0.0",
        "streamlit-plotly-events>=0.0.8",
        "streamlit-option-menu>=0.3.6",
        "streamlit-extras>=0.3.6",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0"
    ]
    
    failed_packages = []
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"✅ {package} installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  Failed to install: {', '.join(failed_packages)}")
        print("You may need to install these manually or use conda instead.")
        return False
    
    print("✅ All dependencies installed successfully!")
    return True

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    required_modules = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'yfinance',
        'requests',
        'vaderSentiment',
        'textblob'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful!")
    return True

def create_env_file():
    """Create .env file from template"""
    env_template = "env_example.txt"
    env_file = ".env"
    
    if os.path.exists(env_file):
        print("⚠️  .env file already exists")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            return True
    
    if os.path.exists(env_template):
        try:
            import shutil
            shutil.copy(env_template, env_file)
            print("✅ Created .env file from template")
            print("📝 Please edit .env file with your API keys")
            return True
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            return False
    else:
        print("❌ env_example.txt not found")
        return False

def provide_alternative_instructions():
    """Provide alternative installation instructions"""
    print("\n🔄 Alternative Installation Methods:")
    print("=" * 50)
    
    print("\n1. Using conda (recommended for Windows):")
    print("   conda create -n market-dashboard python=3.11")
    print("   conda activate market-dashboard")
    print("   pip install -r requirements.txt")
    
    print("\n2. Using virtual environment:")
    print("   python -m venv market-dashboard")
    print("   market-dashboard\\Scripts\\activate  # Windows")
    print("   pip install -r requirements.txt")
    
    print("\n3. Manual installation:")
    print("   pip install numpy==1.26.0")
    print("   pip install pandas streamlit plotly")
    print("   pip install yfinance newsapi-python")
    print("   pip install vaderSentiment textblob")
    print("   pip install openai google-generativeai")
    
    print("\n4. Using pre-built wheels:")
    print("   pip install --only-binary=all -r requirements.txt")

def main():
    """Main installation function"""
    print("🚀 Market & Social Insights Dashboard - Dependency Installation")
    print("=" * 60)
    
    # Check system
    check_system()
    
    # Install numpy first
    if not install_numpy_first():
        print("\n❌ Failed to install numpy. Trying alternative methods...")
        provide_alternative_instructions()
        return
    
    # Install other dependencies
    if not install_other_dependencies():
        print("\n❌ Some dependencies failed to install.")
        provide_alternative_instructions()
        return
    
    # Test imports
    if not test_imports():
        print("\n❌ Some modules failed to import.")
        provide_alternative_instructions()
        return
    
    # Create .env file
    create_env_file()
    
    print("\n🎉 Installation completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: streamlit run app.py")
    print("3. Or run demo: streamlit run demo.py")
    print("4. Open http://localhost:8501 in your browser")

if __name__ == "__main__":
    main() 