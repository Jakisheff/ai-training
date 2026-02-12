import sys
import importlib

def check_python_version():
    print(f"Python version: {sys.version}")
    if sys.version_info >= (3, 9):
        print("✅ Python version >= 3.9")
    else:
        print("❌ Python version < 3.9")

def check_library(lib_name):
    try:
        importlib.import_module(lib_name)
        print(f"✅ {lib_name} is installed")
    except ImportError:
        print(f"❌ {lib_name} is NOT installed")

def main():
    print("Checking environment...")
    check_python_version()
    libraries = ['pandas', 'numpy', 'jupyter', 'matplotlib', 'sklearn']
    for lib in libraries:
        check_library(lib)

if __name__ == "__main__":
    main()
