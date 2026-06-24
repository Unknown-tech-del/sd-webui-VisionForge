import sys
import os
import cv2
import numpy as np
from PIL import Image

# Import visionforge functions
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from scripts import visionforge

def test_pipeline():
    print("VisionForge Verification Test")
    print("=============================")
    
    portrait_path = "assets/default_portrait.png"
    ref_path = "assets/color_reference.png"
    output_path = "assets/test_output_cinematic.png"
    
    if not os.path.exists(portrait_path):
        print(f"Error: Portrait image not found at {portrait_path}")
        return False
        
    if not os.path.exists(ref_path):
        print(f"Error: Color reference image not found at {ref_path}")
        return False
        
    # 1. Load image
    print("Loading test assets...")
    pil_img = Image.open(portrait_path)
    ref_pil = Image.open(ref_path)
    
    # 2. Setup mock objects matching WebUI pipeline arguments
    class MockProcessing:
        seed = 1337

    class MockPostprocessImage:
        def __init__(self, img):
            self.image = img

    p = MockProcessing()
    pp = MockPostprocessImage(pil_img)
    
    # 3. Verify Preset Load / Save / Delete functions
    print("Verifying Preset JSON configuration loading, saving, and deletion...")
    presets = visionforge.load_presets()
    assert "S-Tier Cinematic" in presets
    print(f"Initially loaded presets: {list(presets.keys())}")
    
    # Save a mock preset
    test_preset_name = "Automated Test Preset"
    test_preset_vals = dict(presets["S-Tier Cinematic"])
    test_preset_vals["exposure"] = 1.00
    
    visionforge.save_preset_to_json(test_preset_name, test_preset_vals)
    presets_after_save = visionforge.load_presets()
    assert test_preset_name in presets_after_save
    assert presets_after_save[test_preset_name]["exposure"] == 1.00
    print(f"Saved custom preset successfully: '{test_preset_name}'")
    
    # Delete the mock preset
    visionforge.delete_preset_from_json(test_preset_name)
    presets_after_delete = visionforge.load_presets()
    assert test_preset_name not in presets_after_delete
    print("Deleted custom preset successfully.")
    
    # 4. Instantiate the Script extension class
    print("Instantiating Script extension...")
    script = visionforge.Script()
    
    # 4. Pack argument values matching S-Tier Cinematic Preset
    print("Packing S-Tier Cinematic preset arguments...")
    args = [
        True,                     # enabled
        "S-Tier Cinematic",       # preset
        True, 2.5, 0.06,          # denoise_enabled, sigma, threshold
        True, 0.70,               # sharpen_enabled, sharpen_amount
        0.35, 0.12,               # glow_small, glow_diffuse
        0.90, 0.50, 0.23, 0.77,   # bloom_intensity, threshold, smoothing, saturation
        True, 0.55,               # clarity_enabled, clarity_strength
        -0.25, 1.00, 1.12, 0.00,  # exposure, gamma, contrast, brightness
        1.00, 0.0, -0.15, 0.05,   # saturation, hue, temperature, tint
        0, 0, 0,                  # rgb_r, rgb_g, rgb_b bias offsets
        True, 0.70, ref_pil,      # colormatch_enabled, colormatch_strength, colormatch_image
        "Radial Subject Focus",   # depth_mode
        None,                     # custom_depth_image
        True, 0.70, 0.18, 4.50,   # dof_enabled, dof_focus, dof_fstop, dof_intensity
        True, "#5cd3ff", 0.05, 0.12, # haze_enabled, haze_color, haze_strength, haze_offset
        3.00,                     # ca_amount
        0.40, 0.85, 1.15,         # vignette_intensity, vignette_feather, vignette_zoom
        "#000000",                # vignette_color
        0.12, 1.00,               # grain_power, grain_scale
        0.35, 0.08                # highlight_rolloff, lift_blacks
    ]
    
    # 5. Execute postprocessing hook
    print("Executing postprocess_image pipeline hook...")
    try:
        script.postprocess_image(p, pp, *args)
        processed_image = pp.image
        
        # 6. Verify outputs
        print("Verifying outputs...")
        if processed_image is None:
            print("Error: Output image is None")
            return False
            
        print(f"Processed output dimensions: {processed_image.size[0]}x{processed_image.size[1]}")
        
        # 7. Save output
        print(f"Saving cinematic processed image to {output_path}...")
        processed_image.save(output_path)
        print("Verification completed successfully! Native pipeline executes perfectly.")
        return True
    except Exception as e:
        import traceback
        print(f"Pipeline Execution Failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
