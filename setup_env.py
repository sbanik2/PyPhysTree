import subprocess
import sys
import os
import importlib.util
import re

def check_pip():
    """Ensures pip is available."""
    if importlib.util.find_spec("pip") is None:
        print("❌ Error: 'pip' is not found. Please install pip first.")
        sys.exit(1)

def install_scanners():
    """Installs pipreqsnb and vermin."""
    print("🔍 Installing dependency scanners...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pipreqsnb", "vermin", "--quiet"])
    except subprocess.CalledProcessError:
        print("❌ Error: Failed to install scanners.")
        sys.exit(1)

def scan_imports():
    """Generates requirements.txt from .py and .ipynb files."""
    print("📂 Scanning code for imports...")
    # Generate the raw requirements file first
    cmd = ["pipreqsnb", ".", "--force", "--ignore", ".venv,.git,__pycache__", "--savepath", "requirements.txt"]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        print("⚠️  Warning: pipreqsnb failed. Trying fallback...")
        subprocess.run([sys.executable, "-m", "pipreqsnb", ".", "--force", "--savepath", "requirements.txt"])

def clean_and_relax_requirements():
    """
    1. Removes duplicates.
    2. Converts strict '==' to flexible '>='.
    3. Forces numpy < 2.0.0.
    """
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print("❌ Error: requirements.txt not found.")
        return

    print("🛡️  Cleaning and relaxing requirements...")
    
    with open(req_file, "r") as f:
        lines = f.readlines()

    clean_deps = {}
    
    # Process each line
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        # Split package name and version
        # Matches: "pandas==2.0.3" or "pandas" or "pandas>=1.0"
        match = re.match(r"^([a-zA-Z0-9_\-]+)(.*)$", line)
        if match:
            pkg_name = match.group(1).lower() # Use lowercase key to avoid duplicates
            original_name = match.group(1)    # Keep original casing for file
            version_part = match.group(2)
            
            # SPECIAL RULE: Numpy must be < 2.0.0
            if pkg_name == "numpy":
                clean_deps[pkg_name] = "numpy<2.0.0"
                continue
                
            # GENERAL RULE: Convert '==' to '>=' for flexibility
            if version_part.startswith("=="):
                new_version = version_part.replace("==", ">=")
                clean_deps[pkg_name] = f"{original_name}{new_version}"
            else:
                # If it's already >= or no version, keep as is
                clean_deps[pkg_name] = line

    # Ensure numpy exists if not found
    if "numpy" not in clean_deps:
        clean_deps["numpy"] = "numpy<2.0.0"

    # Write back sorted unique requirements
    with open(req_file, "w") as f:
        for pkg in sorted(clean_deps.keys()):
            f.write(clean_deps[pkg] + "\n")
            
    print("✅ requirements.txt fixed: Duplicates removed & versions relaxed.")

def main():
    print("=== Auto-Dependency Generator (Robust) ===")
    check_pip()
    install_scanners()
    scan_imports()
    clean_and_relax_requirements() # <--- The new step
    
    print("\n🎉 Success! Run this to install:")
    print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()