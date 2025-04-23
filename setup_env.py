#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
import venv
import platform

REQUIRED_BINARIES = ["ffmpeg"]
REQUIRED_PYTHON_PACKAGES = ["spotdl"]
REQUIREMENTS_FILE = "requirements.txt"
VENV_DIR = "venv"

def is_command_available(command):
    return shutil.which(command) is not None

def check_python_version(min_version=(3, 7)):
    current = sys.version_info
    return current >= min_version

def check_required_binaries():
    missing = []
    for binary in REQUIRED_BINARIES:
        if not is_command_available(binary):
            missing.append(binary)
    return missing

def check_required_python_packages():
    missing = []
    for package in REQUIRED_PYTHON_PACKAGES:
        try:
            subprocess.check_output([sys.executable, "-m", "pip", "show", package], stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            missing.append(package)
    return missing

def create_virtual_environment(path):
    print(f"🐍 Creating virtual environment at '{path}'...")
    venv.EnvBuilder(with_pip=True).create(path)

def install_requirements(venv_python):
    if os.path.exists(REQUIREMENTS_FILE):
        print(f"📦 Installing packages from '{REQUIREMENTS_FILE}'...")
        subprocess.check_call([venv_python, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
    else:
        print(f"⚠️ '{REQUIREMENTS_FILE}' not found. Skipping Python package installation.")

def main():
    print("🔧 Starting environment setup...")

    if not check_python_version():
        print("❌ Python 3.7 or newer is required.")
        sys.exit(1)
    else:
        print(f"✅ Python version {platform.python_version()} is sufficient.")

    print("🔍 Checking for required system binaries...")
    missing_binaries = check_required_binaries()
    if missing_binaries:
        print("❌ Missing system dependencies:")
        for bin in missing_binaries:
            print(f"   - {bin}")
        print("👉 Please install them and re-run this script.")
        sys.exit(1)
    else:
        print("✅ All required system binaries found.")

    print("🔍 Checking for required Python packages...")
    missing_packages = check_required_python_packages()
    if missing_packages:
        print("❌ Missing Python packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("👉 Please install them manually (e.g. `pip install spotdl`) or add to requirements.txt.")
        sys.exit(1)
    else:
        print("✅ All required Python packages are installed.")

    if not os.path.exists(VENV_DIR):
        create_virtual_environment(VENV_DIR)
    else:
        print(f"ℹ️ Virtual environment already exists at '{VENV_DIR}'.")

    venv_python = os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin", "python")

    install_requirements(venv_python)

    print("\n🎉 Environment setup complete!")
    print(f"👉 To activate the virtual environment, run:\n   source {VENV_DIR}/bin/activate" if os.name != "nt"
          else f"   {VENV_DIR}\\Scripts\\activate.bat")

if __name__ == "__main__":
    main()
