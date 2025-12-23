#!/usr/bin/env python3
"""
Quick setup checker untuk WebRTC deployment
"""
import sys
import subprocess
import os

print("=" * 60)
print("🧪 STREAMLIT MOTORCYCLE DETECTOR - SETUP CHECKER")
print("=" * 60)

required_packages = {
    "streamlit": "streamlit",
    "numpy": "numpy",
    "pillow": "PIL",
    "torch": "torch",
    "ultralytics": "ultralytics",
    "imageio": "imageio",
    "scipy": "scipy",
    "streamlit-webrtc": "streamlit_webrtc",
    "aiortc": "aiortc",
    "av": "av",
}

missing_packages = []
installed_packages = []

print("\n📋 Checking dependencies...\n")

for display_name, import_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"✅ {display_name}")
        installed_packages.append(display_name)
    except ImportError:
        print(f"❌ {display_name}")
        missing_packages.append(display_name)

print("\n" + "=" * 60)

if missing_packages:
    print(f"\n⚠️  Missing {len(missing_packages)} package(s):\n")
    for pkg in missing_packages:
        print(f"  - {pkg}")
    
    print("\n🔧 Install with:")
    print("   pip install -r requirements.txt\n")
    print("Or install manually:")
    print(f"   pip install {' '.join(missing_packages)}\n")
else:
    print("\n✅ All dependencies installed!\n")

# Check files
print("=" * 60)
print("📁 Checking required files...\n")

required_files = [
    "streamlit_app.py",
    "requirements.txt",
    ".streamlit/config.toml",
]

for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} (missing)")

print("\n" + "=" * 60)

# Check models
print("\n🤖 Checking model files...\n")

model_dir = "models"
if os.path.exists(model_dir):
    models = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if models:
        print(f"Found {len(models)} model(s):")
        for model in models:
            print(f"  ✅ {model}")
    else:
        print(f"❌ No .pt models in {model_dir}/")
        print("   Models will be auto-downloaded on first run")
else:
    print(f"❌ {model_dir}/ directory not found")
    print("   Creating directory...")
    os.makedirs(model_dir, exist_ok=True)

print("\n" + "=" * 60)
print("\n✨ Setup Status:\n")

if not missing_packages:
    print("✅ Ready to run!")
    print("\nStart with:")
    print("  1. Test: streamlit run test_webrtc.py")
    print("  2. Main: streamlit run streamlit_app.py")
else:
    print("❌ Install dependencies first")
    print(f"\npip install -r requirements.txt")

print("\n" + "=" * 60 + "\n")
