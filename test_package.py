#!/usr/bin/env python
"""
Test script to verify the CeyeHao package installation.
"""

def test_imports():
    """Test that all main modules can be imported."""
    try:
        import ceyehao
        print("✅ ceyehao package imported successfully")
        
        import ceyehao.config
        print("✅ ceyehao.config imported successfully")
        
        import ceyehao.data
        print("✅ ceyehao.data imported successfully")
        
        import ceyehao.gui
        print("✅ ceyehao.gui imported successfully")
        
        import ceyehao.models
        print("✅ ceyehao.models imported successfully")
        
        import ceyehao.tools
        print("✅ ceyehao.tools imported successfully")
        
        import ceyehao.utils
        print("✅ ceyehao.utils imported successfully")
        
        print(f"✅ Package version: {ceyehao.__version__}")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_cli():
    """Test that CLI functions can be imported."""
    try:
        from ceyehao.cli import main, launch_gui, train, search
        print("✅ CLI functions imported successfully")
        return True
    except ImportError as e:
        print(f"❌ CLI import failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing CeyeHao package installation...")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_cli()
    
    print("=" * 50)
    if success:
        print("✅ All tests passed! Package is ready to use.")
        print("\nYou can now use:")
        print("  ceyehao gui")
        print("  ceyehao train")
        print("  ceyehao search")
    else:
        print("❌ Some tests failed. Please check the installation.") 