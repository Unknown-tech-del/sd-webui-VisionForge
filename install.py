import launch
import sys

# Ensure opencv-python (cv2) is installed
if not launch.is_installed("opencv-python"):
    print("VisionForge: Installing opencv-python...")
    launch.run_pip("install opencv-python", "requirement for VisionForge extension (opencv-python)")
else:
    print("VisionForge: opencv-python is already installed.")

# WebUI already includes numpy and Pillow (PIL), but we can verify them
if not launch.is_installed("numpy"):
    print("VisionForge: Installing numpy...")
    launch.run_pip("install numpy", "requirement for VisionForge extension (numpy)")

if not launch.is_installed("Pillow"):
    print("VisionForge: Installing Pillow...")
    launch.run_pip("install Pillow", "requirement for VisionForge extension (Pillow)")
