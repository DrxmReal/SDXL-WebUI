# webui.py - File chính chứa WebUI để đặt trên GitHub

import os
import re
import gc
import random
import torch
import gradio as gr
import shutil
from datetime import datetime
from PIL import Image, ImageOps, ImageDraw, ImageFont
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    CogVideoXImageToVideoPipeline
)
# Fix OpenposeDetector initialization by importing the required components
from controlnet_aux import (
    CannyDetector,
    MLSDdetector,
    HEDdetector
)
# Specific import for OpenPose with its dependencies
from controlnet_aux.open_pose import OpenposeDetector
try:
    from controlnet_aux.open_pose.body import BodyEstimator
except ImportError:
    pass  # Will handle the import error in the function

import requests
from transformers import CLIPTokenizer
import json
import glob
from huggingface_hub import hf_hub_download
import huggingface_hub
import socket
from diffusers.utils import export_to_video
import pyngrok.ngrok as ngrok
import urllib.parse
import sys
import io
import zipfile
# Thêm import cho upscaler và watermark
try:
    import cv2
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    import numpy as np
except ImportError:
    print("⚠️ Một số thư viện xử lý hình ảnh chưa được cài đặt. Hãy chạy: pip install opencv-python Pillow numpy")

# Google Drive integration
try:
    from google.colab import drive
    from google.colab import files
    is_colab = True
except ImportError:
    is_colab = False

# Try to import gdown for Google Drive download without authentication
try:
    import gdown
    has_gdown = True
except ImportError:
    has_gdown = False

# === Telegram config ===
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_TOKEN"
TELEGRAM_CHANNEL_ID = "YOUR_CHANNEL_ID"

# Global variables
current_model_id = "PRIMAGEN/Nova-Furry-XL-V4.0"
pipe = None
tokenizer = None
loaded_loras = {}
civitai_api_key = ""  # Pre-filled CivitAI API key
vae_model = None
token_limit = 150  # Default token limit
huggingface_token = ""  # Hugging Face token
ngrok_token = ""  # Default ngrok token
current_scheduler = "Euler"  # Default scheduler
upscaler_model = None  # Global upscaler model
watermark_enabled = False  # Default watermark state
watermark_text = "SDXL WebUI"  # Default watermark text
watermark_opacity = 0.3  # Default watermark opacity

# Định nghĩa các hàm quan trọng ở đầu file
def set_scheduler(scheduler_name):
    """Thiết lập scheduler được chọn"""
    global pipe, current_scheduler, scheduler_list

    if pipe is None:
        return f"❌ Chưa tải mô hình. Không thể đổi scheduler."

    if scheduler_name in scheduler_list:
        current_scheduler = scheduler_name
        pipe.scheduler = scheduler_list[scheduler_name].from_config(pipe.scheduler.config)
        return f"✅ Đã đổi scheduler sang: {scheduler_name}"
    else:
        return f"❌ Không tìm thấy scheduler: {scheduler_name}"

def toggle_watermark(enabled, text=None, opacity=None):
    """Bật/tắt chức năng watermark"""
    global watermark_enabled, watermark_text, watermark_opacity

    watermark_enabled = enabled

    if text is not None and text.strip():
        watermark_text = text

    if opacity is not None and 0 <= opacity <= 1:
        watermark_opacity = opacity

    status = "bật" if watermark_enabled else "tắt"
    return f"✅ Đã {status} watermark. Text: '{watermark_text}', Opacity: {watermark_opacity}"

def load_upscaler(model_name):
    """Load và khởi tạo upscaler model"""
    global upscaler_model, upscaler_models

    if model_name == "None":
        upscaler_model = None
        return "✅ Đã tắt upscaler"

    try:
        # Kiểm tra thư viện
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            return "❌ RealESRGAN không được cài đặt. Hãy chạy: pip install \"basicsr<1.4.2\" realesrgan opencv-python"

        model_info = upscaler_models[model_name]
        model_path = model_info["model_path"]
        scale = model_info["scale"]

        # Tạo thư mục upscaler nếu chưa tồn tại
        os.makedirs("upscaler", exist_ok=True)

        # Download model nếu chưa tồn tại
        local_model_path = os.path.join("upscaler", os.path.basename(model_path))
        if not os.path.exists(local_model_path):
            import urllib.request
            print(f"Downloading upscaler model {model_name}...")
            urllib.request.urlretrieve(model_path, local_model_path)

        # Khởi tạo mô hình upscaler
        if model_info["model_type"] == "realesrgan":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)

            upscaler_model = RealESRGANer(
                scale=scale,
                model_path=local_model_path,
                model=model,
                half=True if torch.cuda.is_available() else False,  # Sử dụng FP16 cho GPU nhanh hơn
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            return f"✅ Đã tải upscaler: {model_name} (scale: {scale}x)"
    except Exception as e:
        upscaler_model = None
        return f"❌ Lỗi khi tải upscaler: {str(e)}"

def upscale_image(image, outscale=None):
    """Upscale hình ảnh sử dụng model đã tải"""
    global upscaler_model

    if upscaler_model is None:
        return image, "⚠️ Chưa tải upscaler nào"

    try:
        # Chuyển từ PIL sang numpy array
        img_np = np.array(image)

        # Xác định tỷ lệ upscale
        if outscale is None:
            scale = upscaler_model.scale
        else:
            scale = outscale

        # Thực hiện upscale
        output, _ = upscaler_model.enhance(img_np, outscale=scale)

        # Chuyển lại thành PIL image
        upscaled_image = Image.fromarray(output)

        height, width = output.shape[:2]
        return upscaled_image, f"✅ Upscale thành công. Kích thước mới: {width}x{height}"
    except Exception as e:
        return image, f"❌ Lỗi khi upscale: {str(e)}"

# Predefined ngrok tokens
ngrok_tokens = {
    "Default": "",
    "Token 1": "",
    "Token 2": "",
    "Token 3": "",
    "Custom": ""
}

# Upscaler model definitions
upscaler_models = {
    "None": None,
    "RealESRGAN_x4plus_anime": {
        "model_name": "RealESRGAN_x4plus_anime",
        "model_path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "model_type": "realesrgan"
    },
    "RealESRGAN_x4plus": {
        "model_name": "RealESRGAN_x4plus",
        "model_path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
        "model_type": "realesrgan"
    },
    "AnimeJaNai_v2": {
        "model_name": "AnimeJaNai_v2",
        "model_path": "https://github.com/the-database/AnimeJaNaiUpscalerPTH/releases/download/v1.0/animejanai-v2-fp32.pth",
        "scale": 2,
        "model_type": "realesrgan"
    },
    "UltraSharp": {
        "model_name": "UltraSharp",
        "model_path": "https://github.com/TencentARC/Real-ESRGAN/releases/download/v2.0.0/RealESRGAN_General_x4_v2.pth",
        "scale": 4,
        "model_type": "realesrgan"
    },
    "AnimeUnreal": {
        "model_name": "AnimeUnreal",
        "model_path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "scale": 4,
        "model_type": "realesrgan"
    }
}

# Available schedulers
scheduler_list = {
    "Euler": EulerDiscreteScheduler,
    "Euler a": EulerAncestralDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "DDIM": DDIMScheduler,
    "Heun": HeunDiscreteScheduler,
    "KDPM2": KDPM2DiscreteScheduler,
    "LMS": LMSDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "UniPC": UniPCMultistepScheduler
}

# ControlNet global variables
controlnet_models = {}
controlnet_processors = {
    "canny": None,
    "openpose": None,
    "mlsd": None,
    "hed": None
}
current_controlnet = None

# Simple CSS for rounded corners
css = """
.gradio-container {
    border-radius: 12px;
}

button, select, textarea, input, .gradio-box, .gradio-dropdown, .gradio-slider {
    border-radius: 8px !important;
}

.gradio-gallery {
    border-radius: 10px;
}

.gradio-gallery img {
    border-radius: 6px;
}

/* Fix close button positioning */
.gradio-modal {
    max-width: 90vw !important;
    max-height: 90vh !important;
    margin: auto !important;
    border-radius: 12px !important;
}

.gradio-modal .preview-image {
    position: relative !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    max-height: 80vh !important;
}

.gradio-modal img {
    max-width: 100% !important;
    max-height: 80vh !important;
    object-fit: contain !important;
    border-radius: 8px !important;
}

.preview-image .close-btn,
.gradio-gallery .preview button,
div[id*="gallery"] button {
    position: absolute !important;
    top: 10px !important;
    right: 10px !important;
    z-index: 100 !important;
    background-color: rgba(0, 0, 0, 0.5) !important;
    border-radius: 50% !important;
    width: 30px !important;
    height: 30px !important;
    padding: 0 !important;
}
"""

# Thêm toàn bộ các hàm từ file main.py ở đây
# (clean_prompt, add_border, create_preview, upload_to_webhook, send_to_telegram, get_scheduler, load_pipeline, switch_model,
# generate_images, generate_video, shutdown, search_civitai_loras, download_lora_from_civitai, load_vae, load_lora_weights, etc.)

# Thêm phần WebUI từ file main.py
# Bao gồm toàn bộ phần with gr.Blocks(...): và các hàm khởi tạo ban đầu

# Add all required functions here, before the UI definition
def clean_prompt(prompt):
    return re.sub(r'[^a-zA-Z0-9]+', '_', prompt.lower())[:50]

def add_border(image, border=12, color='black'):
    return ImageOps.expand(image, border=border, fill=color)

def create_preview(image, max_width=512):
    ratio = max_width / image.width
    new_size = (max_width, int(image.height * ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS)

def upload_to_webhook(image_path, webhook_url, metadata_text):
    if not webhook_url.strip():
        return
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/png')}
            data = {'content': metadata_text}
            r = requests.post(webhook_url, data=data, files=files)
            if r.status_code not in [200, 204]:
                print(f"❌ Upload failed: {r.status_code}")
    except Exception as e:
        print(f"🚨 Upload error: {e}")

def send_to_telegram(image_path, metadata_text):
    logs = [] # Initialize a local log list
    try:
        logs.append(f"ℹ️ Attempting to send image: {image_path} to Telegram channel: {TELEGRAM_CHANNEL_ID}") # Added log
        # Thêm IP vào metadata
        ip_address = get_ip_address()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_with_ip = f"{metadata_text}\n> IP: `{ip_address}` | Time: {timestamp}"

        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        with open(image_path, 'rb') as f:
            files = {'photo': f}
            data = {
                'chat_id': TELEGRAM_CHANNEL_ID,
                'caption': metadata_with_ip,
            }
            r = requests.post(telegram_url, data=data, files=files)

            # Improved error logging
            if r.status_code != 200:
                logs.append(f"❌ Telegram API responded with status code: {r.status_code}")
                logs.append(f"❌ Response body: {r.text}")
                logs.append(f"❌ Failed to send image: {image_path}")
            else:
                logs.append(f"✅ Successfully sent image: {image_path} to Telegram") # Success log

    except Exception as e:
        logs.append(f"🚨 An error occurred while sending image to Telegram: {e}") # Detailed error log
        logs.append(f"🚨 Image file path: {image_path}") # Log file path on error

    return logs # Return the log list

def get_scheduler(name, config):
    global scheduler_list
    return scheduler_list.get(name, EulerDiscreteScheduler).from_config(config)

def load_pipeline(use_img2img, model_id):
    global tokenizer, vae_model, huggingface_token
    if use_img2img:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            use_auth_token=huggingface_token
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            use_auth_token=huggingface_token
        )
    pipe.scheduler = get_scheduler("Euler", pipe.scheduler.config)

    # Set VAE if available
    if vae_model is not None:
        pipe.vae = vae_model

    pipe.enable_attention_slicing()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    return pipe.to("cuda")

def switch_model(new_model_id, use_controlnet=False):
    global pipe, current_model_id, current_controlnet

    if new_model_id != current_model_id or (use_controlnet and not isinstance(pipe, StableDiffusionXLControlNetPipeline)):
        print(f"🔁 Switching model to: {new_model_id}" + (" with ControlNet" if use_controlnet else ""))
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        if use_controlnet and current_controlnet is not None:
            pipe = load_pipeline_with_controlnet(new_model_id, current_controlnet)
        else:
            pipe = load_pipeline(False, new_model_id)

        current_model_id = new_model_id
        return f"✅ Switched to model: {new_model_id}" + (" with ControlNet" if use_controlnet else "")
    return f"ℹ️ Model already in use: {new_model_id}"

# Thêm các hàm còn lại từ file main.py, bao gồm:
# generate_images, generate_video, load_controlnet, process_controlnet_image, load_pipeline_with_controlnet, v.v.

# Hàm để khởi chạy Web UI
def launch_ui(selected_model="stabilityai/stable-diffusion-xl-base-1.0", selected_vae="None", selected_lora="None", lora_weight=0.8):
    global pipe, tokenizer, current_model_id

    # Tạo các thư mục cần thiết
    os.makedirs("models", exist_ok=True)
    os.makedirs("loras", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    os.makedirs("controlnet", exist_ok=True)
    create_csb_folders()

    # Khởi tạo tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # Tải mô hình ban đầu
    current_model_id = selected_model
    pipe = load_pipeline(False, selected_model)

    # Tải VAE nếu được chỉ định
    if selected_vae and selected_vae != "None":
        load_vae(selected_vae)

    # Đăng nhập Hugging Face
    if huggingface_token:
        huggingface_hub.login(token=huggingface_token, add_to_git_credential=False)

    # Quét thư mục ảnh
    scan_images_directory()

    # Khởi động WebUI
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

# Thêm phần WebUI và các hàm cần thiết còn lại ở đây từ file main.py

# Chạy UI khi file được chạy trực tiếp
if __name__ == "__main__":
    # Các mô hình mặc định
    selected_model = "stabilityai/stable-diffusion-xl-base-1.0" 
    selected_vae = "None"
    selected_lora = "None"
    lora_weight = 0.8
    
    launch_ui(selected_model, selected_vae, selected_lora, lora_weight)