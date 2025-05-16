# 🎨 SDXL WebUI

A powerful and user-friendly interface for Stable Diffusion XL models, optimized for Google Colab.

## ✨ Features

- **🖼️ Multiple Generation Methods**:
  - Text-to-Image: Create images from text descriptions
  - Image-to-Image: Transform existing images
  - Image-to-Video: Convert still images to short videos
  - ControlNet: Precise control over image generation

- **🧠 Rich Model Support**:
  - Dozens of SDXL models (both SFW and NSFW)
  - Anime, realistic, and furry specialized models
  - Extensive VAE options
  - 40+ LoRA adapters for style fine-tuning

- **🔧 Advanced Tools**:
  - Real-time image upscaling
  - Watermarking capabilities
  - Gallery system for reviewing and searching past generations
  - Token counting for optimal prompting

- **☁️ Storage Integration**:
  - Google Drive mounting and syncing
  - Automatic saving of all generated content
  - Import/export capabilities

- **🌐 Sharing Options**:
  - Discord webhook integration
  - Telegram sharing
  - Ngrok tunnel for remote access

## 🚀 Getting Started

### Google Colab (Recommended)

1. Open the [SDXL WebUI Colab Notebook](https://colab.research.google.com/github/YOUR_USERNAME/SDXL-WebUI/blob/main/SDXL_WebUI.ipynb)
2. Click "Runtime" → "Run all" or run each cell in sequence
3. Follow the link provided to access the WebUI interface

### Local Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SDXL-WebUI.git
   cd SDXL-WebUI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the WebUI:
   ```bash
   python main.py
   ```

## 📋 Usage Guide

### Text-to-Image

1. Enter your prompt in the "Prompt" field
2. Add any negative prompts to guide what you don't want to see
3. Adjust settings like image size, steps, and guidance scale
4. Click "Generate Text to Image"

### Image-to-Image

1. Upload an initial image
2. Enter your prompt describing the desired changes
3. Adjust the "Strength" slider to control how much of the original image to preserve
4. Click "Generate Image to Image"

### ControlNet

1. Select a ControlNet type (canny, openpose, depth, etc.)
2. Upload a control image
3. Process the control image to see the guidance map
4. Adjust ControlNet settings
5. Click "Generate with ControlNet"

### Additional Features

- **Upscalers**: Enhance image resolution using AI upscalers
- **Schedulers**: Switch between different sampling algorithms
- **Watermarking**: Add customizable watermarks to your images
- **Google Drive**: Save and load models and generations directly to/from Google Drive

## 📝 Notes

- NSFW capabilities depend on the selected model and may be restricted on some platforms
- For best results, use the T4 or A100 GPU runtime in Google Colab
- Google Drive integration requires authorization on first use

## 🔄 Models & LoRAs

This WebUI supports any SDXL-compatible model and LoRA from:
- Hugging Face
- CivitAI (requires API key)
- Google Drive (manual upload)

## 📄 License

This project is released under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgements

- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) by Stability AI
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- [Gradio](https://github.com/gradio-app/gradio) for the web interface
- All model creators who have shared their work publicly

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/YOUR_USERNAME/SDXL-WebUI/issues). 