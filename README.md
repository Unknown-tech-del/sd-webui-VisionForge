# sd-webui-VisionForge

Professional, S-tier post-processing extension for Stable Diffusion WebUI (AUTOMATIC1111) and Forge. Enhance your generated artwork with custom bloom, sharpening, advanced color grading, lens distortion, depth-of-field, atmospheric haze, and cinematic film finishing tools—all built natively into your WebUI generation tab! 🚀

---

## 🎨 Key Features

1.  **Bilateral Denoise & CAS Sharpen**: Clean up JPEG/generation artifacts and apply Contrast Adaptive Sharpening (CAS) without haloing or crisping skin textures.
2.  **Arcane Bloom & Glow**: Extract high-luminance elements and blend diffuse, small, and wide cinematic bloom layers back additively (featuring saturation and knee smoothing controls).
3.  **Clarity & Advanced Color Grading**: Boost midtone local contrast (Clarity) in Hard Light blend mode. Fine-tune exposure, contrast, temperature, tint, RGB balance, and highlights.
4.  **Reinhard Color Reference Match**: Copy the color palette and grading curves from any reference image directly onto your generation using statistical LAB color transfer.
5.  **Depth of Field & Fog Haze**: Simulate focal depth of field (DoF bokeh) and distance-based atmospheric fog using procedural gradients (radial subject focus, landscape linear slope) or custom black & white depth maps.
6.  **Finishing Lens FX**: Apply radial chromatic aberration, customizable vignettes, filmic highlights rolloff, shadow lifts (lift blacks), and procedural film grain.

---

## ⚙️ Installation

1.  Open Stable Diffusion WebUI or Forge.
2.  Navigate to the **Extensions** tab -> **Install from URL** sub-tab.
3.  Paste the URL of this repository:
    `https://github.com/Unknown-tech-del/sd-webui-VisionForge`
4.  Click **Install**.
5.  Navigate to the **Installed** tab, click **Apply and restart UI**.

---

## 🚀 How to Use

1.  Open either the **txt2img** or **img2img** tab.
2.  Locate the collapsible **VisionForge Post-Processing 📷** accordion panel (usually near the Scripts section).
3.  Check **Enable VisionForge**.
4.  Choose one of the built-in artistic presets (e.g., **S-Tier Cinematic**, **Soft Glow Portrait**, **Moody Landscape**) to automatically set parameters, or manually slide and adjust values to craft your own custom signature look.
5.  Generate your image! The post-processing filters will execute automatically and output your styled artwork immediately.
