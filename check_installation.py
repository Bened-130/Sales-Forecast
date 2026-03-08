"""
Check if all dependencies are installed correctly
"""

print("Checking dependencies...\n")

try:
    import pandas
    print(f"✅ Pandas: {pandas.__version__}")
except ImportError:
    print("❌ Pandas: NOT INSTALLED")

try:
    import numpy
    print(f"✅ NumPy: {numpy.__version__}")
except ImportError:
    print("❌ NumPy: NOT INSTALLED")

try:
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
except ImportError:
    print("❌ Scikit-learn: NOT INSTALLED")

try:
    import prophet
    print(f"✅ Prophet: INSTALLED")
except ImportError:
    print("❌ Prophet: NOT INSTALLED")

try:
    import matplotlib
    print(f"✅ Matplotlib: {matplotlib.__version__}")
except ImportError:
    print("❌ Matplotlib: NOT INSTALLED")

try:
    import scipy
    print(f"✅ SciPy: {scipy.__version__}")
except ImportError:
    print("❌ SciPy: NOT INSTALLED")

print("\n" + "="*50)
print("Run this in terminal: pip install -r requirements.txt")