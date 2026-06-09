# ComfyUI-Camera-ISP

This is a physics-based Image Signal Processor (ISP) and film grain simulator custom node for ComfyUI.

Unlike standard artistic grain filters, this node simulates the physical pipeline of a digital camera sensor using GPU-native PyTorch operations. It is designed for computational photography research, forensic visual analysis, and high-end photorealistic post-processing.

## 🚀 Key Engineering Features

* **Bayer CFA & Demosaicing:** Simulates a single-channel raw mosaic and reconstructs the image using normalized convolution demosaicing, mathematically generating accurate micro-artifacts (e.g., zipper artifacts).
* **Constant-Memory PRNU (Sensor Fingerprint):** Implements deterministic spatial Photo-Response Non-Uniformity. Instead of memory-heavy tensor repetition or frequency-destructive interpolation, it uses modulo indexing. This ensures O(1) VRAM scaling for the noise map, preventing CUDA Out-Of-Memory errors even on massive resolutions.
* **GPU-Native Execution:** Completely vectorized. Eliminated CPU fallbacks and graph breaks, ensuring perfect compatibility with modern PyTorch compilation and maximum GPU utilization.
* **Optimized Memory Management:** Heavy kernels and Gaussian filters are pre-computed and stored using internal buffers, preventing VRAM allocation churn during the forward pass.
* **Diagnostic Output:** Includes a dual-output system, allowing users to isolate and visualize the pure high-frequency noise/PRNU delta applied to the base image.

---

## 📦 Installation

1. Navigate to your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes

```


2. Clone this repository (or copy the `camera_isp_grain` folder):
```bash
git clone https://github.com/Manuil1/ComfyUI-Camera-ISP.git

```


3. Restart ComfyUI. This node relies entirely on the native PyTorch environment and requires no external dependencies.

---

## 🖼️ Included Workflow: Flux.2 Klein Img2Img Pro

The `workflows/` directory contains `Flux2Klein_Img2Img_Portrait_Pipeline.json`. This is a highly optimized pipeline demonstrating the Camera ISP node applied at the end of an AI upscaling chain to restore true photographic realism to synthetic subjects .

**Pipeline Architecture:**
`LoadImage` -> `Flux.2 Klein 9B` -> `Optional LoRA` -> `VAE Decode` -> `Local AI Upscale` -> `Camera ISP & Grain` -> `SaveImage`

### Required Models

Place the following files in their respective ComfyUI directories:

* **UNET:** `ComfyUI/models/diffusion_models/Flux.2/flux-2-klein-base-9b-fp8.safetensors`
* **Text Encoder:** `ComfyUI/models/text_encoders/qwen_3_8b_fp8mixed.safetensors` *(Ensure you select the 8B variant in the CLIPLoader, not 4B)*
* **VAE:** `ComfyUI/models/vae/flux2-vae.safetensors`
* **Upscaler:** `ComfyUI/models/upscale_models/4x_NMKD-Superscale-SP_178000_G.pth`

### Recommended Settings & Tips

* **Sampling:** Use 20 steps, CFG 4.5, Euler/Simple.
* **Denoise Strength:** Set to 0.35 - 0.50 for strict identity retention, or 0.60 - 0.75 for heavy transformation.
* **LoRA Slot:** The workflow includes a bypassed LoRA slot. When using your own style/identity LoRA, disable the bypass and set `strength_model` between 0.55 and 0.85 (leave `strength_clip` at 0.0 for Flux/DiT architecture).
* **Upscaling:** The default is a robust ESRGAN. .
* **Camera ISP:** Keep the global effect strength between 0.45 and 0.70 for the most natural photographic integration.

### ⚖️ Ethical Use

This workflow and ISP node are built for transparent synthetic media and high-end graphics post-production. Do not use this toolset to impersonate real individuals or mislead the public. Always position the output transparently as synthetic content or virtual influencer media.
