# Requirements and Installation Check
import subprocess
import sys

def install_requirements():
    """Install all required packages with error handling"""
    
    requirements = [
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "transformers>=4.20.0",
        "gradio>=3.40.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "pillow>=8.3.0",
        "plotly>=5.0.0",
        "psutil>=5.8.0",
        "opencv-python>=4.5.0",
        "GPUtil>=1.4.0"
    ]
    
    print("📦 Installing required packages...")
    print("=" * 40)
    
    for requirement in requirements:
        try:
            print(f"Installing {requirement}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", requirement, "--quiet"
            ])
            print(f"✅ {requirement} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {requirement}: {str(e)}")
            print(f"💡 Try: pip install {requirement}")
    
    print("\n✅ Installation complete!")
    print("🚀 You can now run the Advanced ML Studio")

def check_installation():
    """Check if all required packages are installed"""
    
    packages = {
        'torch': 'PyTorch',
        'gradio': 'Gradio',
        'transformers': 'Transformers',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'PIL': 'Pillow',
        'plotly': 'Plotly',
        'psutil': 'PSUtil',
        'cv2': 'OpenCV'
    }
    
    print("🔍 Checking installed packages...")
    print("=" * 35)
    
    missing_packages = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name}: Installed")
        except ImportError:
            print(f"❌ {name}: Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("💡 Run install_requirements() to install missing packages")
        return False
    else:
        print("\n🎉 All packages are installed!")
        return True

if __name__ == "__main__":
    if not check_installation():
        install_requirements()
    else:
        print("🚀 Ready to launch Advanced ML Studio!")
