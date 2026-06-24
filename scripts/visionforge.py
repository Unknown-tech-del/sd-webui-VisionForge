import os
import json
import random
import cv2
import numpy as np
from PIL import Image
import gradio as gr

try:
    import modules.scripts as scripts
    from modules import images, shared
    from modules.processing import Processed, StableDiffusionProcessing
    IS_WEBUI = True
except ImportError:
    # Standalone mock fallback for local unit testing
    class MockScripts:
        class Script:
            pass
    scripts = MockScripts()
    class MockProcessed:
        pass
    Processed = MockProcessed
    class MockProcessing:
        pass
    StableDiffusionProcessing = MockProcessing
    IS_WEBUI = False

# Preset values dictionary for front-end synchronization (Default read-only presets)
PRESETS = {
    "S-Tier Cinematic": {
        "denoise_enabled": True, "denoise_sigma": 2.5, "denoise_threshold": 0.06,
        "sharpen_enabled": True, "sharpen_amount": 0.70,
        "glow_small": 0.35, "glow_diffuse": 0.12, "bloom_intensity": 0.90,
        "bloom_threshold": 0.50, "bloom_smoothing": 0.23, "bloom_saturation": 0.77,
        "clarity_enabled": True, "clarity_strength": 0.55,
        "exposure": -0.25, "gamma": 1.00, "contrast": 1.12, "brightness": 0.00,
        "saturation": 1.00, "hue": 0.0, "temperature": -0.15, "tint": 0.05,
        "rgb_r": 0, "rgb_g": 0, "rgb_b": 0,
        "colormatch_enabled": False, "colormatch_strength": 0.70,
        "depth_mode": "Radial Subject Focus",
        "dof_enabled": True, "dof_focus": 0.70, "dof_fstop": 0.18, "dof_intensity": 4.50,
        "haze_enabled": True, "haze_color": "#5cd3ff", "haze_strength": 0.05, "haze_offset": 0.12,
        "ca_amount": 3.00, "vignette_intensity": 0.40, "vignette_feather": 0.85, "vignette_zoom": 1.15,
        "vignette_color": "#000000", "grain_power": 0.12, "grain_scale": 1.00,
        "rolloff": 0.35, "lift": 0.08
    },
    "Soft Glow Portrait": {
        "denoise_enabled": True, "denoise_sigma": 4.0, "denoise_threshold": 0.08,
        "sharpen_enabled": True, "sharpen_amount": 0.35,
        "glow_small": 0.50, "glow_diffuse": 0.30, "bloom_intensity": 1.20,
        "bloom_threshold": 0.40, "bloom_smoothing": 0.50, "bloom_saturation": 0.90,
        "clarity_enabled": False, "clarity_strength": 0.00,
        "exposure": 0.10, "gamma": 1.05, "contrast": 0.95, "brightness": 0.02,
        "saturation": 1.05, "hue": 0.0, "temperature": 0.15, "tint": -0.02,
        "rgb_r": 5, "rgb_g": 0, "rgb_b": -5,
        "colormatch_enabled": False, "colormatch_strength": 0.70,
        "depth_mode": "Radial Subject Focus",
        "dof_enabled": True, "dof_focus": 0.45, "dof_fstop": 0.12, "dof_intensity": 7.00,
        "haze_enabled": False, "haze_color": "#ffffff", "haze_strength": 0.00, "haze_offset": 0.00,
        "ca_amount": 2.00, "vignette_intensity": 0.30, "vignette_feather": 0.90, "vignette_zoom": 1.25,
        "vignette_color": "#180c05", "grain_power": 0.05, "grain_scale": 1.20,
        "rolloff": 0.20, "lift": 0.05
    },
    "Moody Landscape": {
        "denoise_enabled": False, "denoise_sigma": 1.0, "denoise_threshold": 0.05,
        "sharpen_enabled": True, "sharpen_amount": 0.80,
        "glow_small": 0.10, "glow_diffuse": 0.05, "bloom_intensity": 0.30,
        "bloom_threshold": 0.65, "bloom_smoothing": 0.15, "bloom_saturation": 0.50,
        "clarity_enabled": True, "clarity_strength": 0.65,
        "exposure": -0.10, "gamma": 0.95, "contrast": 1.20, "brightness": -0.02,
        "saturation": 1.15, "hue": -5.0, "temperature": -0.20, "tint": 0.08,
        "rgb_r": -8, "rgb_g": 2, "rgb_b": 10,
        "colormatch_enabled": False, "colormatch_strength": 0.70,
        "depth_mode": "Linear Landscape Gradient",
        "dof_enabled": True, "dof_focus": 0.85, "dof_fstop": 0.40, "dof_intensity": 2.50,
        "haze_enabled": True, "haze_color": "#8ca8b8", "haze_strength": 0.25, "haze_offset": 0.40,
        "ca_amount": 1.50, "vignette_intensity": 0.55, "vignette_feather": 0.70, "vignette_zoom": 1.00,
        "vignette_color": "#000000", "grain_power": 0.08, "grain_scale": 0.80,
        "rolloff": 0.50, "lift": -0.04
    },
    "Default (Bypass)": {
        "denoise_enabled": False, "denoise_sigma": 2.0, "denoise_threshold": 0.05,
        "sharpen_enabled": False, "sharpen_amount": 0.00,
        "glow_small": 0.00, "glow_diffuse": 0.00, "bloom_intensity": 0.00,
        "bloom_threshold": 0.50, "bloom_smoothing": 0.25, "bloom_saturation": 1.00,
        "clarity_enabled": False, "clarity_strength": 0.00,
        "exposure": 0.00, "gamma": 1.00, "contrast": 1.00, "brightness": 0.00,
        "saturation": 1.00, "hue": 0.0, "temperature": 0.00, "tint": 0.00,
        "rgb_r": 0, "rgb_g": 0, "rgb_b": 0,
        "colormatch_enabled": False, "colormatch_strength": 0.70,
        "depth_mode": "Radial Subject Focus",
        "dof_enabled": False, "dof_focus": 0.50, "dof_fstop": 0.28, "dof_intensity": 0.00,
        "haze_enabled": False, "haze_color": "#ffffff", "haze_strength": 0.00, "haze_offset": 0.00,
        "ca_amount": 0.00, "vignette_intensity": 0.00, "vignette_feather": 0.80, "vignette_zoom": 1.20,
        "vignette_color": "#000000", "grain_power": 0.00, "grain_scale": 1.00,
        "rolloff": 0.00, "lift": 0.00
    }
}

# Mapping order of all slider configuration keys
SLIDER_KEYS = [
    "denoise_enabled", "denoise_sigma", "denoise_threshold",
    "sharpen_enabled", "sharpen_amount",
    "glow_small", "glow_diffuse", "bloom_intensity", "bloom_threshold", "bloom_smoothing", "bloom_saturation",
    "clarity_enabled", "clarity_strength", "exposure", "gamma", "contrast", "brightness", "saturation", "hue", "temperature", "tint",
    "rgb_r", "rgb_g", "rgb_b",
    "colormatch_enabled", "colormatch_strength",
    "depth_mode", "dof_enabled", "dof_focus", "dof_fstop", "dof_intensity",
    "haze_enabled", "haze_color", "haze_strength", "haze_offset",
    "ca_amount", "vignette_intensity", "vignette_feather", "vignette_zoom", "vignette_color",
    "grain_power", "grain_scale", "rolloff", "lift"
]

# File path to persist custom presets
PRESETS_FILE = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "presets.json")

def load_presets():
    """Load both default read-only and custom user-saved presets from JSON."""
    all_presets = dict(PRESETS)
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                custom = json.load(f)
                all_presets.update(custom)
        except Exception as e:
            print(f"VisionForge Error: Failed to read presets.json: {e}")
    return all_presets

def save_preset_to_json(name, values_dict):
    """Write custom preset values dictionary back to local JSON configuration."""
    custom = {}
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                custom = json.load(f)
        except Exception:
            pass
    custom[name] = values_dict
    try:
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(custom, f, indent=4)
    except Exception as e:
        print(f"VisionForge Error: Failed to write presets.json: {e}")

def delete_preset_from_json(name):
    """Delete a custom preset key from configuration file."""
    if name in PRESETS:
        return # Cannot delete default presets
    custom = {}
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                custom = json.load(f)
        except Exception:
            pass
    if name in custom:
        del custom[name]
    try:
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(custom, f, indent=4)
    except Exception as e:
        print(f"VisionForge Error: Failed to update presets.json during deletion: {e}")

# Hex to RGB color helper
def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 3:
        hex_str = ''.join([c*2 for c in hex_str])
    return [int(hex_str[i:i+2], 16) for i in (0, 2, 4)]

# -------------------------------------------------------------------------
# Post-Processing Algorithms
# -------------------------------------------------------------------------

def clarity_fx(img, strength):
    if strength <= 0.0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), 12.0)
    high_freq = np.clip(img - blurred + 0.5, 0.0, 1.0)
    
    # Hard Light Blend Mode
    mask = high_freq < 0.5
    blended = np.empty_like(img)
    blended[mask] = 2.0 * img[mask] * high_freq[mask]
    blended[~mask] = 1.0 - 2.0 * (1.0 - img[~mask]) * (1.0 - high_freq[~mask])
    
    return (1.0 - strength) * img + strength * blended

def color_match(src, ref, strength):
    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    src_mean, src_std = cv2.meanStdDev(src_lab)
    ref_mean, ref_std = cv2.meanStdDev(ref_lab)
    
    src_mean = src_mean.flatten()
    src_std = src_std.flatten()
    ref_mean = ref_mean.flatten()
    ref_std = ref_std.flatten()
    
    matched = np.empty_like(src_lab)
    for i in range(3):
        matched[:,:,i] = (src_lab[:,:,i] - src_mean[i]) * (ref_std[i] / (src_std[i] + 1e-5)) + ref_mean[i]
        
    matched = np.clip(matched, 0.0, 255.0).astype(np.uint8)
    matched_rgb = cv2.cvtColor(matched, cv2.COLOR_LAB2RGB)
    
    return cv2.addWeighted(src, 1.0 - strength, matched_rgb, strength, 0.0)

def apply_bloom_glow(img, small_glow, diffuse_glow, bloom_intensity, threshold, smoothing, saturation):
    luma = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
    luma = luma[:, :, np.newaxis]
    
    knee = threshold * smoothing
    soft = luma - threshold + knee
    soft = np.clip(soft, 0.0, 2.0 * knee)
    soft = (soft * soft) / (4.0 * knee + 1e-5)
    factor = np.maximum(luma - threshold, soft) / (np.maximum(luma, 1e-5))
    bright = img * factor
    
    h, w = img.shape[:2]
    bright_down = cv2.resize(bright, (w//2, h//2))
    
    glow_small = cv2.GaussianBlur(bright_down, (0, 0), 3.0)
    glow_diffuse = cv2.GaussianBlur(bright_down, (0, 0), 14.0)
    bloom = cv2.GaussianBlur(bright_down, (0, 0), 8.0)
    
    glow_small = cv2.resize(glow_small, (w, h))
    glow_diffuse = cv2.resize(glow_diffuse, (w, h))
    bloom = cv2.resize(bloom, (w, h))
    
    if saturation != 1.0:
        bloom_luma = (0.2126 * bloom[:,:,0] + 0.7152 * bloom[:,:,1] + 0.0722 * bloom[:,:,2])[:, :, np.newaxis]
        bloom = bloom_luma + saturation * (bloom - bloom_luma)
        bloom = np.clip(bloom, 0.0, 1.0)
        
    out = img + glow_small * small_glow + glow_diffuse * diffuse_glow + bloom * bloom_intensity
    return np.clip(out, 0.0, 1.0)

def color_grade(img, exposure, gamma, contrast, brightness, saturation, hue, temp, tint, rgb_adj):
    img = img * (2.0 ** exposure)
    
    img[:,:,0] += rgb_adj[0] / 255.0
    img[:,:,1] += rgb_adj[1] / 255.0
    img[:,:,2] += rgb_adj[2] / 255.0
    
    img[:,:,0] += temp * 0.12 
    img[:,:,2] -= temp * 0.12 
    img[:,:,1] += tint * 0.08 
    img[:,:,0] -= tint * 0.04
    img[:,:,2] -= tint * 0.04
    img = np.clip(img, 0.0, 1.0)
    
    img = (img - 0.5) * contrast + 0.5
    img = np.clip(img, 0.0, 1.0)
    
    img += brightness
    img = np.clip(img, 0.0, 1.0)
    
    if saturation != 1.0 or hue != 0.0:
        hsv = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + hue / 2.0) % 180.0
        hsv[:,:,1] *= saturation
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0.0, 255.0)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        
    if gamma != 1.0:
        img = np.power(img, 1.0 / gamma)
        img = np.clip(img, 0.0, 1.0)
        
    return img

def apply_dof_haze(img, depth, dof_enabled, focus_point, fstop, dof_intensity, haze_enabled, haze_color_hex, haze_strength, haze_offset):
    h, w = img.shape[:2]
    
    if haze_enabled and haze_strength > 0.0:
        haze_factor = np.clip((depth - haze_offset) / (1.0 - haze_offset + 1e-5), 0.0, 1.0)
        haze_factor = np.power(haze_factor, 1.5) * haze_strength
        haze_factor = haze_factor[:, :, np.newaxis]
        
        color_rgb = np.array(hex_to_rgb(haze_color_hex), dtype=np.float32) / 255.0
        img = img * (1.0 - haze_factor) + color_rgb * haze_factor
        
    if dof_enabled and dof_intensity > 0.0:
        coc = np.abs(depth - focus_point)
        coc = coc / (fstop + 1e-5)
        coc = np.clip(coc, 0.0, 1.0)
        coc = coc * coc * (3.0 - 2.0 * coc)
        
        blur_radius = coc * dof_intensity
        
        blur_2 = cv2.GaussianBlur(img, (0, 0), 2.0)
        blur_5 = cv2.GaussianBlur(img, (0, 0), 5.0)
        blur_11 = cv2.GaussianBlur(img, (0, 0), 11.0)
        
        out = np.empty_like(img)
        
        mask_0 = blur_radius <= 2.0
        mask_1 = (blur_radius > 2.0) & (blur_radius <= 5.0)
        mask_2 = (blur_radius > 5.0) & (blur_radius <= 11.0)
        mask_3 = blur_radius > 11.0
        
        t = (blur_radius / 2.0)[:, :, np.newaxis]
        out = np.where(mask_0[:, :, np.newaxis], img * (1.0 - t) + blur_2 * t, out)
        
        t = ((blur_radius - 2.0) / 3.0)[:, :, np.newaxis]
        out = np.where(mask_1[:, :, np.newaxis], blur_2 * (1.0 - t) + blur_5 * t, out)
        
        t = ((blur_radius - 5.0) / 6.0)[:, :, np.newaxis]
        out = np.where(mask_2[:, :, np.newaxis], blur_5 * (1.0 - t) + blur_11 * t, out)
        
        out = np.where(mask_3[:, :, np.newaxis], blur_11, out)
        img = out
        
    return img

def apply_lens_fx(img, ca_amount, vignette_intensity, vignette_feather, vignette_zoom, vignette_color_hex, grain_power, grain_scale, highlight_rolloff, lift_blacks, seed):
    h, w = img.shape[:2]
    
    if ca_amount > 0.0:
        y, x = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        dx = (x - cx) / cx
        dy = (y - cy) / cy
        
        shift = ca_amount / w
        map_x_r = (x - dx * shift * cx).astype(np.float32)
        map_y_r = (y - dy * shift * cy).astype(np.float32)
        map_x_b = (x + dx * shift * cx).astype(np.float32)
        map_y_b = (y + dy * shift * cy).astype(np.float32)
        
        r = cv2.remap(img[:,:,0], map_x_r, map_y_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        g = img[:,:,1]
        b = cv2.remap(img[:,:,2], map_x_b, map_y_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        img = np.stack([r, g, b], axis=-1)
        
    if highlight_rolloff > 0.0:
        img = img / (1.0 + img * highlight_rolloff)
        
    if abs(lift_blacks) > 0.0:
        luma = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
        shadow_weight = np.power(1.0 - luma, 2.0)[:, :, np.newaxis]
        img += (1.0 - img) * lift_blacks * shadow_weight
        img = np.clip(img, 0.0, 1.0)
        
    if vignette_intensity > 0.0:
        y, x = np.mgrid[0:h, 0:w]
        dx = (x - w/2.0) / (w/2.0)
        dy = (y - h/2.0) / (h/2.0)
        dist = np.sqrt(dx*dx + dy*dy) * vignette_zoom
        
        start = 1.0 - vignette_feather
        vignette_mask = np.clip((dist - start * 0.5) / (0.7 - start * 0.5 + 1e-5), 0.0, 1.0)
        vignette_mask = vignette_mask * vignette_mask * (3.0 - 2.0 * vignette_mask)
        vignette_mask = vignette_mask[:, :, np.newaxis] * vignette_intensity
        
        color_rgb = np.array(hex_to_rgb(vignette_color_hex), dtype=np.float32) / 255.0
        img = img * (1.0 - vignette_mask) + color_rgb * vignette_mask
        
    if grain_power > 0.0:
        np.random.seed(int(seed))
        gh, gw = int(h / max(grain_scale, 0.5)), int(w / max(grain_scale, 0.5))
        noise = np.random.normal(0.0, 1.0, (gh, gw))
        noise = cv2.resize(noise, (w, h))[:, :, np.newaxis]
        img += noise * grain_power
        img = np.clip(img, 0.0, 1.0)
        
    return img

def cas_sharpen(img, amount):
    kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    laplacian = cv2.filter2D(img, -1, kernel)
    
    gray = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    local_max = cv2.dilate(gray, np.ones((3, 3)))
    local_min = cv2.erode(gray, np.ones((3, 3)))
    edge_contrast = local_max - local_min
    
    weight = amount * (1.0 - edge_contrast)
    weight = np.clip(weight, 0.0, 1.0)[:, :, np.newaxis]
    
    return np.clip(img - laplacian * (0.25 * weight), 0.0, 1.0)

# -------------------------------------------------------------------------
# Main Gradio UI Integration Script
# -------------------------------------------------------------------------

class Script(scripts.Script):
    def title(self):
        return "VisionForge Post-Processor"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        all_presets = load_presets()
        
        with gr.Accordion("VisionForge Post-Processing 📷", open=False):
            enabled = gr.Checkbox(label="Enable VisionForge", value=False)
            
            # --- Custom Styled Presets Accordion Block (Matches reference image) ---
            with gr.Accordion("Presets", open=True):
                with gr.Row():
                    with gr.Column(scale=3):
                        preset = gr.Dropdown(
                            label="Presets",
                            choices=list(all_presets.keys()),
                            value="S-Tier Cinematic"
                        )
                        new_preset_name = gr.Textbox(
                            label="Preset Name",
                            placeholder="Enter name to save..."
                        )
                    with gr.Column(scale=2):
                        btn_apply_preset = gr.Button(
                            "Apply Presets",
                            elem_id="visionforge_apply_preset_btn"
                        )
                        btn_save_preset = gr.Button(
                            "Save to Presets",
                            elem_id="visionforge_save_preset_btn"
                        )
                
            # Accordion 1: Denoise & Sharpen
            with gr.Accordion("Denoise & Sharpen", open=False):
                with gr.Row():
                    denoise_enabled = gr.Checkbox(label="Enable Bilateral Denoise", value=True)
                    denoise_sigma = gr.Slider(label="Denoise Sigma (Radius)", minimum=0.5, maximum=8.0, step=0.1, value=2.5)
                    denoise_threshold = gr.Slider(label="Color Similarity Threshold", minimum=0.01, maximum=0.25, step=0.005, value=0.06)
                with gr.Row():
                    sharpen_enabled = gr.Checkbox(label="Enable CAS Sharpen", value=True)
                    sharpen_amount = gr.Slider(label="Sharpen Strength", minimum=0.0, maximum=1.5, step=0.05, value=0.70)
                    
            # Accordion 2: Bloom & Glow
            with gr.Accordion("Bloom & Glow", open=False):
                with gr.Row():
                    glow_small = gr.Slider(label="Glow Small Intensity", minimum=0.0, maximum=1.0, step=0.05, value=0.35)
                    glow_diffuse = gr.Slider(label="Glow Diffuse Intensity", minimum=0.0, maximum=1.0, step=0.05, value=0.12)
                with gr.Row():
                    bloom_intensity = gr.Slider(label="Arcane Bloom Intensity", minimum=0.0, maximum=2.0, step=0.05, value=0.90)
                    bloom_threshold = gr.Slider(label="Bloom Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.50)
                with gr.Row():
                    bloom_smoothing = gr.Slider(label="Bloom Smoothing", minimum=0.01, maximum=1.0, step=0.01, value=0.23)
                    bloom_saturation = gr.Slider(label="Bloom Saturation", minimum=0.0, maximum=2.0, step=0.05, value=0.77)
                    
            # Accordion 3: Color & Clarity
            with gr.Accordion("Color & Clarity", open=False):
                with gr.Row():
                    clarity_enabled = gr.Checkbox(label="Enable Clarity FX", value=True)
                    clarity_strength = gr.Slider(label="Clarity Strength", minimum=0.0, maximum=1.0, step=0.05, value=0.55)
                    exposure = gr.Slider(label="Exposure Offset", minimum=-2.0, maximum=2.0, step=0.05, value=-0.25)
                with gr.Row():
                    gamma = gr.Slider(label="Gamma Correction", minimum=0.4, maximum=2.0, step=0.05, value=1.00)
                    contrast = gr.Slider(label="Contrast Multiplier", minimum=0.5, maximum=1.8, step=0.02, value=1.12)
                    brightness = gr.Slider(label="Brightness Offset", minimum=-0.2, maximum=0.2, step=0.01, value=0.00)
                with gr.Row():
                    saturation = gr.Slider(label="Saturation Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.00)
                    hue = gr.Slider(label="Hue Shift (Degrees)", minimum=-180, maximum=180, step=1, value=0)
                with gr.Row():
                    temperature = gr.Slider(label="Temperature Shift", minimum=-1.0, maximum=1.0, step=0.05, value=-0.15)
                    tint = gr.Slider(label="Tint Shift", minimum=-1.0, maximum=1.0, step=0.05, value=0.05)
                with gr.Row():
                    rgb_r = gr.Slider(label="Red Offset Bias", minimum=-100, maximum=100, step=1, value=0)
                    rgb_g = gr.Slider(label="Green Offset Bias", minimum=-100, maximum=100, step=1, value=0)
                    rgb_b = gr.Slider(label="Blue Offset Bias", minimum=-100, maximum=100, step=1, value=0)
                    
            # Accordion 4: Color Reference Match
            with gr.Accordion("Color Reference Match", open=False):
                with gr.Row():
                    colormatch_enabled = gr.Checkbox(label="Enable Reinhard Color Match", value=False)
                    colormatch_strength = gr.Slider(label="Match Strength", minimum=0.0, maximum=1.0, step=0.05, value=0.70)
                with gr.Row():
                    colormatch_image = gr.Image(label="Grade Reference Image", type="pil")

            # Accordion 5: Depth & Haze
            with gr.Accordion("Depth & Haze", open=False):
                with gr.Row():
                    depth_mode = gr.Dropdown(
                        label="Depth Map Estimation",
                        choices=["Radial Subject Focus", "Linear Landscape Gradient", "Custom Upload"],
                        value="Radial Subject Focus"
                    )
                with gr.Row():
                    custom_depth_image = gr.Image(label="Custom Black & White Depth Map Upload", type="pil", visible=False)
                
                with gr.Row():
                    dof_enabled = gr.Checkbox(label="Enable Depth of Field (DoF)", value=True)
                    dof_focus = gr.Slider(label="Focus Distance", minimum=0.0, maximum=1.0, step=0.01, value=0.70)
                    dof_fstop = gr.Slider(label="Aperture (f-stop)", minimum=0.02, maximum=0.80, step=0.01, value=0.18)
                    dof_intensity = gr.Slider(label="Max DoF Blur Radius", minimum=0.0, maximum=12.0, step=0.25, value=4.50)
                
                with gr.Row():
                    haze_enabled = gr.Checkbox(label="Enable Atmospheric Haze", value=True)
                    haze_color = gr.ColorPicker(label="Haze Color", value="#5cd3ff")
                    haze_strength = gr.Slider(label="Haze Fog Density", minimum=0.0, maximum=1.0, step=0.02, value=0.05)
                    haze_offset = gr.Slider(label="Haze Start Distance", minimum=0.0, maximum=1.0, step=0.02, value=0.12)

            # Accordion 6: Lens FX & Finishing
            with gr.Accordion("Lens FX & Finishing", open=False):
                with gr.Row():
                    ca_amount = gr.Slider(label="Chromatic Aberration Shift", minimum=0.0, maximum=10.0, step=0.2, value=3.00)
                    vignette_intensity = gr.Slider(label="Vignette Strength", minimum=0.0, maximum=1.0, step=0.02, value=0.40)
                    vignette_feather = gr.Slider(label="Vignette Feather", minimum=0.1, maximum=1.0, step=0.02, value=0.85)
                with gr.Row():
                    vignette_zoom = gr.Slider(label="Vignette Scale/Zoom", minimum=0.5, maximum=2.0, step=0.05, value=1.15)
                    vignette_color = gr.ColorPicker(label="Vignette Color", value="#000000")
                with gr.Row():
                    grain_power = gr.Slider(label="Film Grain Power", minimum=0.0, maximum=0.3, step=0.01, value=0.12)
                    grain_scale = gr.Slider(label="Film Grain Scale", minimum=0.5, maximum=3.0, step=0.1, value=1.00)
                with gr.Row():
                    rolloff = gr.Slider(label="Highlight Rolloff", minimum=0.0, maximum=1.5, step=0.05, value=0.35)
                    lift = gr.Slider(label="Lift Blacks (Shadows)", minimum=-0.2, maximum=0.2, step=0.01, value=0.08)

        # ---------------------------------------------------------------------
        # Front-End UI Bindings (Callbacks)
        # ---------------------------------------------------------------------
        
        # Hide custom depth map slot unless Custom Mode selected
        depth_mode.change(
            fn=lambda m: gr.update(visible=(m == "Custom Upload")),
            inputs=[depth_mode],
            outputs=[custom_depth_image]
        )

        # List of outputs to synchronize when loading a preset
        slider_outputs = [
            denoise_enabled, denoise_sigma, denoise_threshold,
            sharpen_enabled, sharpen_amount,
            glow_small, glow_diffuse, bloom_intensity, bloom_threshold, bloom_smoothing, bloom_saturation,
            clarity_enabled, clarity_strength, exposure, gamma, contrast, brightness, saturation, hue, temperature, tint,
            rgb_r, rgb_g, rgb_b,
            colormatch_enabled, colormatch_strength,
            depth_mode, dof_enabled, dof_focus, dof_fstop, dof_intensity,
            haze_enabled, haze_color, haze_strength, haze_offset,
            ca_amount, vignette_intensity, vignette_feather, vignette_zoom, vignette_color,
            grain_power, grain_scale, rolloff, lift
        ]

        def update_presets(preset_name):
            current_presets = load_presets()
            p = current_presets.get(preset_name, PRESETS["S-Tier Cinematic"])
            return [
                p["denoise_enabled"], p["denoise_sigma"], p["denoise_threshold"],
                p["sharpen_enabled"], p["sharpen_amount"],
                p["glow_small"], p["glow_diffuse"], p["bloom_intensity"], p["bloom_threshold"], p["bloom_smoothing"], p["bloom_saturation"],
                p["clarity_enabled"], p["clarity_strength"], p["exposure"], p["gamma"], p["contrast"], p["brightness"], p["saturation"], p["hue"], p["temperature"], p["tint"],
                p["rgb_r"], p["rgb_g"], p["rgb_b"],
                p["colormatch_enabled"], p["colormatch_strength"],
                p["depth_mode"], p["dof_enabled"], p["dof_focus"], p["dof_fstop"], p["dof_intensity"],
                p["haze_enabled"], p["haze_color"], p["haze_strength"], p["haze_offset"],
                p["ca_amount"], p["vignette_intensity"], p["vignette_feather"], p["vignette_zoom"], p["vignette_color"],
                p["grain_power"], p["grain_scale"], p["rolloff"], p["lift"]
            ]

        # Apply Presets button applies the preset configurations to all sliders
        btn_apply_preset.click(
            fn=update_presets,
            inputs=[preset],
            outputs=slider_outputs
        )

        # Custom Presets: Save to Presets Callback
        def on_save_click(name, *values):
            if not name or name.strip() == "":
                return gr.update()
            name = name.strip()
            if name in PRESETS:
                return gr.update() # Prevent overwriting built-in templates
                
            values_dict = {}
            for k, val in zip(SLIDER_KEYS, values):
                values_dict[k] = val
                
            save_preset_to_json(name, values_dict)
            current_presets = load_presets()
            return gr.update(choices=list(current_presets.keys()), value=name)

        btn_save_preset.click(
            fn=on_save_click,
            inputs=[new_preset_name] + slider_outputs,
            outputs=[preset]
        )

        # Return UI variables to process positionally in postprocess_image
        return [
            enabled, preset,
            denoise_enabled, denoise_sigma, denoise_threshold,
            sharpen_enabled, sharpen_amount,
            glow_small, glow_diffuse, bloom_intensity, bloom_threshold, bloom_smoothing, bloom_saturation,
            clarity_enabled, clarity_strength, exposure, gamma, contrast, brightness, saturation, hue, temperature, tint,
            rgb_r, rgb_g, rgb_b,
            colormatch_enabled, colormatch_strength, colormatch_image,
            depth_mode, custom_depth_image, dof_enabled, dof_focus, dof_fstop, dof_intensity,
            haze_enabled, haze_color, haze_strength, haze_offset,
            ca_amount, vignette_intensity, vignette_feather, vignette_zoom, vignette_color,
            grain_power, grain_scale, rolloff, lift
        ]

    def postprocess_image(self, p, pp, *args):
        # Unpack positional arguments returned from UI list
        (
            enabled, preset,
            denoise_enabled, denoise_sigma, denoise_threshold,
            sharpen_enabled, sharpen_amount,
            glow_small, glow_diffuse, bloom_intensity, bloom_threshold, bloom_smoothing, bloom_saturation,
            clarity_enabled, clarity_strength, exposure, gamma, contrast, brightness, saturation, hue, temperature, tint,
            rgb_r, rgb_g, rgb_b,
            colormatch_enabled, colormatch_strength, colormatch_image,
            depth_mode, custom_depth_image, dof_enabled, dof_focus, dof_fstop, dof_intensity,
            haze_enabled, haze_color, haze_strength, haze_offset,
            ca_amount, vignette_intensity, vignette_feather, vignette_zoom, vignette_color,
            grain_power, grain_scale, rolloff, lift
        ) = args
        
        if not enabled:
            return

        pil_img = pp.image
        img_np = np.array(pil_img)
        
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        
        if denoise_enabled:
            img_cv = cv2.bilateralFilter(
                img_cv,
                d=-1,
                sigmaColor=denoise_threshold * 255.0,
                sigmaSpace=denoise_sigma
            )

        img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        if colormatch_enabled and colormatch_image is not None:
            try:
                ref_np = np.array(colormatch_image)
                if len(ref_np.shape) == 3 and ref_np.shape[2] == 3:
                    src_uint8 = (img * 255.0).astype(np.uint8)
                    matched_uint8 = color_match(src_uint8, ref_np, colormatch_strength)
                    img = matched_uint8.astype(np.float32) / 255.0
            except Exception as e:
                print(f"VisionForge Error: Reinhard Color Match failed: {e}")

        if sharpen_enabled and sharpen_amount > 0.0:
            img = cas_sharpen(img, sharpen_amount)

        if clarity_enabled and clarity_strength > 0.0:
            img = clarity_fx(img, clarity_strength)

        img = color_grade(
            img, exposure, gamma, contrast, brightness, saturation, hue,
            temperature, tint, [rgb_r, rgb_g, rgb_b]
        )

        img = apply_bloom_glow(
            img, glow_small, glow_diffuse, bloom_intensity,
            bloom_threshold, bloom_smoothing, bloom_saturation
        )

        depth = np.zeros((h, w), dtype=np.float32)
        
        if depth_mode == "Custom Upload" and custom_depth_image is not None:
            try:
                custom_depth_np = np.array(custom_depth_image.convert('L'))
                depth = cv2.resize(custom_depth_np, (w, h)).astype(np.float32) / 255.0
            except Exception as e:
                print(f"VisionForge Error: Custom Depth loading failed: {e}")
                depth_mode = "Radial Subject Focus"
                
        if depth_mode == "Radial Subject Focus":
            cx, cy = w / 2.0, h * 0.35
            max_rad = max(w, h) * 0.8
            y, x = np.mgrid[0:h, 0:w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            depth = np.clip(dist / max_rad, 0.0, 1.0)
            
        elif depth_mode == "Linear Landscape Gradient":
            y, x = np.mgrid[0:h, 0:w]
            depth = np.clip(1.0 - (y.astype(np.float32) / h), 0.0, 1.0)

        img = apply_dof_haze(
            img, depth, dof_enabled, dof_focus, dof_fstop, dof_intensity,
            haze_enabled, haze_color, haze_strength, haze_offset
        )

        final_seed = p.seed if p.seed != -1 else random.randint(0, 100000)
        img = apply_lens_fx(
            img, ca_amount, vignette_intensity, vignette_feather, vignette_zoom,
            vignette_color, grain_power, grain_scale, rolloff, lift, final_seed
        )

        out_uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
        pp.image = Image.fromarray(out_uint8)
