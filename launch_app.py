# Launch Script for Advanced ML Studio
import sys
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main launch function with comprehensive error handling"""
    try:
        print("🚀 Starting Advanced ML Studio...")
        print("=" * 50)
        
        # Check Python version
        if sys.version_info < (3, 7):
            raise RuntimeError("Python 3.7 or higher is required")
        
        print(f"✅ Python version: {sys.version}")
        
        # Import and check dependencies
        print("📦 Checking dependencies...")
        
        try:
            import torch
            print(f"✅ PyTorch: {torch.__version__}")
        except ImportError:
            print("❌ PyTorch not found. Please install: pip install torch")
            return
        
        try:
            import gradio as gr
            print(f"✅ Gradio: {gr.__version__}")
        except ImportError:
            print("❌ Gradio not found. Please install: pip install gradio")
            return
        
        try:
            import transformers
            print(f"✅ Transformers: {transformers.__version__}")
        except ImportError:
            print("❌ Transformers not found. Please install: pip install transformers")
            return
        
        # Check system resources
        print("\n🔍 System Information:")
        print("=" * 30)
        
        import psutil
        print(f"💾 RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"🖥️  CPU Cores: {psutil.cpu_count()}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"🎮 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print("🎮 GPU: Not available (CPU mode)")
        
        print("\n🚀 Launching Advanced ML Studio...")
        print("=" * 50)
        
        # Import our modules
        from ml_models import ml_models
        from gradio_interface import interface, demo
        from performance_optimizer import performance_optimizer
        
        print("✅ All modules loaded successfully!")
        
        # Launch the interface
        print("\n🌟 Features Available:")
        print("  📝 Text Classification")
        print("  💭 Sentiment Analysis") 
        print("  🖼️ Image Classification")
        print("  📝 Text Generation")
        print("  📊 Analytics Dashboard")
        print("  ℹ️ Model Information")
        
        print("\n🔧 Optimizations Active:")
        print("  ⚡ GPU acceleration (if available)")
        print("  🧠 Automatic memory management")
        print("  📊 Real-time performance monitoring")
        print("  🛡️ Comprehensive error handling")
        print("  🎨 Professional UI/UX")
        
        print("\n🚀 Starting web interface...")
        print("🌐 The app will be available at: http://localhost:7860")
        print("🔗 Public link will be generated for sharing")
        print("\n" + "=" * 50)
        
        # Launch with optimal settings
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            debug=False,
            quiet=False,
            show_tips=True,
            enable_queue=True,
            max_threads=10
        )
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Shutting down Advanced ML Studio...")
        print("👋 Thank you for using Advanced ML Studio!")
        
    except Exception as e:
        print(f"\n❌ Error launching application: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("  1. Check all dependencies are installed")
        print("  2. Ensure sufficient system memory")
        print("  3. Try restarting the application")
        print("  4. Check Python version compatibility")
        
        import traceback
        print(f"\n🐛 Full error details:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
