# @title SDXL WebUI Settings
# @markdown ## 🧠 Model Selection
selected_model = "PRIMAGEN/Nova-Furry-XL-V4.0" # @param ["PRIMAGEN/Nova-Furry-XL-V4.0", "stabilityai/stable-diffusion-xl-base-1.0", "stablediffusionapi/cyberrealistic-v4.0", "SG161222/RealVisXL_V3.0", "Lykon/dreamshaper-xl-1-0", "RunDiffusion/Juggernaut-XL-v7", "zenless-lab/sdxl-anything-xl", "John6666/wai-nsfw-illustrious-sdxl-v140-sdxl", "LyliaEngine/Pony_Diffusion_V6_XL", "cagliostrolab/animagine-xl-3.1", "gsdf/Counterfeit-XL", "digiplay/AbsoluteReality_v1.8.1", "Undi95/CyberRealistic-v4.1", "isamu-NSFW/Toon_Blended_XL", "xxmatt/realistic-vision-v60-sdxl", "prompthero/openjourney-xl", "segmind/SSD-1B", "xyn-ai/xxl-stylized-1-1", "Anashel/Anime-NSFW-XL", "BunnyMagic/NeverminoXL-v4", "XXMatt/LightSkinEroticBest-NSFW-XL", "Mahou/NeverEnding_Dream", "Meina/MeinaHentaiXL-V3", "BunnyMagic/XXLRealFurVisionXL", "Mahou/MeinaPastelXL", "VanillaMilk/EroChan-NSFW-XL", "MEKARAYI/Animagine-XL-NSFW"]

# @markdown ## 🧩 VAE Selection
selected_vae = "None" # @param ["None", "stabilityai/sd-vae-ft-mse", "stabilityai/sdxl-vae", "Linaqruf/anime-vae", "stablediffusionapi/anything-v5-vae", "hakurei/waifu-diffusion-vae", "Mahou/Pastel-Mix-VAE", "gsdf/CounterfeitXL-VAE", "openai/consistency-decoder", "sayakpaul/sd-vae-ft-ema-diffusers", "iZELX1/SunshineMix-VAE", "stabilityai/sd-x2-latent-upscaler-vae", "madebyollin/sdxl-vae-fp16-fix", "furryai/vae-ft-mse", "johnslegers/EroticVAE", "NagisaZj/VAE_Anime", "ringhyacinth/novelai-vae-fp16-fix", "VanillaMilk/EroChan-VAE", "BunnyMagic/ShinyXLVAE", "UhriG/SakumiVAE"]

# @markdown ## 🎭 LoRA Selection
selected_lora = "None" # @param ["None", "lykon/dreamshaper-xl-lora", "goofyai/SDXL_Anime_Style", "ironjr/nep_sdxl_style_lora", "latent-consistency/lcm-lora-sdxl", "KBlueLeaf/kohaku-xl-lora", "Mikubill/manga-xl", "furusu/anime-detail-lora-sdxl", "p1atdev/cutenatural", "artificialguybr/anime-lora-xl", "Cosphotos/anime-background-style-xl-lora", "Cosphotos/anime-pastel-lora-sdxl", "Meina/MeinaPastel-LoRA", "stabilityai/sd-turbo", "Mahou/Flat-2D-Anime-XL", "artificialguybr/arisoyaXL", "iZELX1/SunshineMix-SDXL-Lora", "artificialguybr/anylora-xl", "BunnyMagic/ComfyLewd-NSFW-SDXL", "BunnyMagic/AggressiveXL-NSFW", "BunnyMagic/CuteFurryXLLoRA", "sazyou/perfect_lewd_detail", "sazyou/perfect_erotic_vagina", "sazyou/perfect_anal", "XXMatt/EroticDetail-NSFW-SDXL", "XXMatt/PerfectLewd-NSFW-SDXL", "VanillaMilk/EroChan-LoRA", "nendotan/ShojoManga-LoRA", "Meina/MeinaAnime-A", "Meina/MeinaAnime-B", "Anashel/Anime-Details-Lora", "CrucibleAI/MeinaUnreal", "Mahou/MeinaPastel-LoRA", "latent-consistency/lcm-lora-sdxl", "Meina/AnimeStyl2D-SD15", "artificialguybr/furry-lora-sdxl", "jakubsalamon/FXSY-Fantasy-NSFW", "artificialguybr/xxl-realistic-fur", "AllegroMangesh/FadFluffySDXL", "evasora/furever", "AnalRabbit/AnalRabbitSDXL-R9"]
lora_weight = 0.8 # @param {type:"slider", min:0.0, max:1.0, step:0.05}

# Required dependencies for the app
# pip install torch torchvision diffusers transformers gradio pyngrok huggingface_hub
# pip install realesrgan basicsr opencv-python
# Add new features: upscaler, scheduler selection, watermarks

import os
import re
import gc
import random
import torch
import gradio as gr
import shutil
from datetime import datetime
from PIL import Image, ImageOps
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
TELEGRAM_BOT_TOKEN = "7854611550:AAGbicT0X83igmuu8GTN88W3mK2yoPvVvHQ"
TELEGRAM_CHANNEL_ID = "-1002609505799"

# Global variables
current_model_id = "PRIMAGEN/Nova-Furry-XL-V4.0"
pipe = None
tokenizer = None
loaded_loras = {}
civitai_api_key = "8a01029177731541c4c74ebcdea42c38"  # Pre-filled CivitAI API key
vae_model = None
token_limit = 150  # Default token limit
huggingface_token = "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"  # Hugging Face token
ngrok_token = "2RQCtzU2XCZBJu8tCGvkh3xVf5H_2nzqbT1X8amuyeS8tt71Z"  # Default ngrok token
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
    "Default": "2RQCtzU2XCZBJu8tCGvkh3xVf5H_2nzqbT1X8amuyeS8tt71Z",
    "Token 1": "2UNV1yHbLyLuKAQR9FhTh9bzzW2_3PB9QLASzRwuEpPKAWq9a",
    "Token 2": "2TUDlfkHu6y85Fj6k8IfyDUn6MH_6SQ6JQScqsTyKCDLYTvnk",
    "Token 3": "2WB4Q8W8g71JtOCnwVmN2gDlKS6_6rMTzT3yXKPHjvTzAkJSX",
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
    try:
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
                'parse_mode': 'Markdown'
            }
            r = requests.post(telegram_url, data=data, files=files)
            if r.status_code != 200:
                print(f"❌ Telegram upload failed: {r.status_code}, {r.text}")
    except Exception as e:
        print(f"🚨 Telegram upload error: {e}")

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

def generate_images(prompt, negative_prompt, width, height, steps, guidance,
                    strength, seed, init_image, use_img2img, num_images, webhook_url,
                    do_upscale=False, apply_watermark=False, progress=gr.Progress(track_tqdm=False)):

    global pipe, current_model_id, tokenizer, selected_model, selected_lora, lora_weight
    from time import time

    model_status = switch_model(selected_model)
    logs = [model_status]
    
    # Lấy IP người dùng
    try:
        user_ip = requests.get("https://api.ipify.org").text
        logs.append(f"🌐 User IP: {user_ip}")
    except Exception as e:
        logs.append(f"⚠️ Could not fetch IP: {e}")

    if pipe.__class__.__name__.startswith("StableDiffusionXLImg2Img") != use_img2img:
        pipe = load_pipeline(use_img2img, selected_model)
    
    # Apply LoRA if selected
    lora_message = "No LoRA selected"
    if selected_lora and selected_lora != "None":
        pipe, lora_message = load_lora_weights(pipe, selected_lora, lora_weight)
        logs.append(lora_message)

    if seed == 0:
        seed = random.randint(1, 1 << 30)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    output_dir = os.path.join("images", datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(output_dir, exist_ok=True)

    # Count tokens
    prompt_tokens = tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"].shape[1]
    neg_prompt_tokens = tokenizer(negative_prompt, return_tensors="pt", truncation=False)["input_ids"].shape[1]
    total_tokens = prompt_tokens + neg_prompt_tokens

    token_info = (
        f"🧠 Token usage:\n"
        f"> Prompt Tokens: {prompt_tokens} {'⚠️' if prompt_tokens > 77 else ''}\n"
        f"> Negative Prompt Tokens: {neg_prompt_tokens} {'⚠️' if neg_prompt_tokens > 77 else ''}\n"
        f"> Total Tokens: {total_tokens}\n"
        f"{'-'*30}"
    )
    logs.append(token_info)

    previews = []
    for i in range(num_images):
        start_time = time()

        if use_img2img and init_image:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=generator
            ).images[0]
        else:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]

        elapsed = time() - start_time
        percent = int((i + 1) / num_images * 100)

        timestamp = datetime.now().strftime("%H%M%S_%f")
        filename = f"{clean_prompt(prompt)}_seed{seed}_{timestamp}.png"
        file_path = os.path.join(output_dir, filename)
        bordered_image = add_border(image)
        bordered_image.save(file_path)

        # Áp dụng upscaler và watermark nếu cần
        if do_upscale or apply_watermark:
            global watermark_enabled
            orig_watermark_enabled = watermark_enabled
            
            # Tạm thời gán giá trị cho watermark_enabled dựa trên tham số
            watermark_enabled = apply_watermark
            
            # Xử lý ảnh
            processed_image, process_msg = process_with_upscaler_and_watermark(bordered_image)
            logs.append(process_msg)
            
            # Lưu ảnh đã xử lý
            processed_filename = f"{clean_prompt(prompt)}_seed{seed}_{timestamp}_processed.png"
            processed_file_path = os.path.join(output_dir, processed_filename)
            processed_image.save(processed_file_path)
            
            # Lưu ảnh đã xử lý vào CSB
            csb_result = save_to_csb(processed_file_path)
            logs.append(csb_result)
            
            # Khôi phục giá trị watermark_enabled
            watermark_enabled = orig_watermark_enabled
            
            preview = create_preview(processed_image, max_width=512)
        else:
            # Save to CSB folder
            csb_result = save_to_csb(file_path)
            logs.append(csb_result)
            
            preview = create_preview(bordered_image, max_width=512)
            
        previews.append(preview)

        metadata = (
            f"🖼️ *Image Generated*\n"
            f"> Prompt: `{prompt}`\n"
            f"> Seed: `{seed}`\n"
            f"> Size: {width}x{height} | Steps: {steps} | Guidance: {guidance}\n"
            f"> Strength: {strength if use_img2img else 'N/A'}\n"
            f"> LoRA: {lora_message if selected_lora != 'None' else 'None'}"
        )
        upload_to_webhook(file_path, webhook_url, metadata)
        send_to_telegram(file_path, metadata)

        logs.append(
            f"✅ [{percent}%] Saved: {file_path} | Time: {elapsed:.2f}s"
        )
    
    # Reset LoRA fusion if one was applied
    if selected_lora and selected_lora != "None":
        pipe.unfuse_lora()

    return previews, "\n".join(logs)

def generate_video(prompt, negative_prompt, input_image, num_inference_steps, guidance_scale, num_frames, fps, progress=gr.Progress(track_tqdm=True)):
    global tokenizer
    
    logs = []
    
    # Count tokens
    prompt_tokens = tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"].shape[1]
    neg_prompt_tokens = tokenizer(negative_prompt, return_tensors="pt", truncation=False)["input_ids"].shape[1]
    total_tokens = prompt_tokens + neg_prompt_tokens

    token_info = (
        f"🧠 Token usage:\n"
        f"> Prompt Tokens: {prompt_tokens} {'⚠️' if prompt_tokens > 77 else ''}\n"
        f"> Negative Prompt Tokens: {neg_prompt_tokens} {'⚠️' if neg_prompt_tokens > 77 else ''}\n"
        f"> Total Tokens: {total_tokens}\n"
        f"{'-'*30}"
    )
    logs.append(token_info)
    
    progress(0, desc="Loading Video Model")
    
    try:
        # Load the pipeline
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX-5b-I2V",
            torch_dtype=torch.bfloat16
        )
        
        # Memory optimizations
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        
        # Set the device
        pipe = pipe.to("cuda")
        
        progress(0.2, desc="Generating Video")
        
        # Generate seed if not provided
        generator = torch.Generator(device="cuda").manual_seed(random.randint(1, 1 << 30))
        
        # Generate the video
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=input_image,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        progress(0.8, desc="Exporting Video")
        
        # Create output directory
        output_dir = os.path.join("videos", datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(output_dir, exist_ok=True)
        
        # Save video
        timestamp = datetime.now().strftime("%H%M%S_%f")
        clean_prompt_text = clean_prompt(prompt)
        output_path = os.path.join(output_dir, f"{clean_prompt_text}_{timestamp}.mp4")
        
        # Export frames to video
        export_to_video(result.frames[0], output_path, fps=fps)
        
        logs.append(f"✅ Video generated and saved to: {output_path}")
        
        # Save to CSB folder
        csb_result = save_to_csb(output_path, is_video=True)
        logs.append(csb_result)
        
        # Free up memory
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        
        return output_path, "\n".join(logs)
        
    except Exception as e:
        error_message = f"❌ Error generating video: {str(e)}"
        logs.append(error_message)
        return None, "\n".join(logs)

def shutdown():
    os._exit(0)

# Add all required functions here, before the UI definition
def search_civitai_loras(query, api_key):
    if not api_key:
        return [], "⚠️ CivitAI API key not provided"
    
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {
            "query": query,
            "limit": 10,
            "type": "LORA"
        }
        response = requests.get(
            "https://civitai.com/api/v1/models", 
            headers=headers,
            params=params
        )
        
        if response.status_code != 200:
            return [], f"❌ CivitAI API error: {response.status_code}"
        
        data = response.json()
        results = []
        
        for model in data.get("items", []):
            name = model.get("name", "Unknown")
            id = model.get("id", "")
            version_id = model.get("modelVersions", [{}])[0].get("id", "") if model.get("modelVersions") else ""
            results.append((f"{name} (ID: {id})", f"{id}:{version_id}"))
            
        return results, f"✅ Found {len(results)} LoRAs"
    except Exception as e:
        return [], f"🚨 Error searching CivitAI: {str(e)}"

def download_lora_from_civitai(lora_id_version, api_key):
    global loaded_loras
    
    if not api_key:
        return f"⚠️ CivitAI API key not provided"
    
    try:
        lora_id, version_id = lora_id_version.split(":")
        
        # Get download URL from CivitAI
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            f"https://civitai.com/api/v1/model-versions/{version_id}",
            headers=headers
        )
        
        if response.status_code != 200:
            return f"❌ CivitAI API error: {response.status_code}"
        
        data = response.json()
        download_url = None
        
        for file in data.get("files", []):
            if file.get("primary"):
                download_url = file.get("downloadUrl")
                break
        
        if not download_url:
            return "❌ No download URL found for this LoRA"
        
        # Create lora directory if it doesn't exist
        os.makedirs("loras", exist_ok=True)
        safetensor_name = f"lora_{lora_id}.safetensors"
        lora_path = os.path.join("loras", safetensor_name)
        
        # Download the file
        response = requests.get(download_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(lora_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Add to loaded_loras dict
        lora_name = data.get("model", {}).get("name", f"LoRA {lora_id}")
        loaded_loras[lora_id_version] = {
            "path": lora_path,
            "name": lora_name
        }
        
        return f"✅ Downloaded LoRA: {lora_name} ({os.path.getsize(lora_path) / (1024*1024):.2f} MB)"
    except Exception as e:
        return f"🚨 Error downloading LoRA: {str(e)}"

def load_vae(vae_name):
    global vae_model
    
    if vae_name == "None":
        vae_model = None
        return "✅ Using model default VAE"
    
    try:
        if vae_name == "stabilityai/sd-vae-ft-mse":
            vae_model = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse", 
                torch_dtype=torch.float16
            )
        elif vae_name == "stabilityai/sdxl-vae":
            vae_model = AutoencoderKL.from_pretrained(
                "stabilityai/sdxl-vae", 
                torch_dtype=torch.float16
            )
        else:
            return f"❌ Unknown VAE model: {vae_name}"
            
        return f"✅ Loaded VAE: {vae_name}"
    except Exception as e:
        vae_model = None
        return f"🚨 Error loading VAE: {str(e)}"

def load_lora_weights(pipe, lora_id_version, lora_weight=0.8):
    global loaded_loras
    
    if lora_id_version not in loaded_loras:
        return pipe, f"⚠️ LoRA not loaded: {lora_id_version}"
    
    lora_path = loaded_loras[lora_id_version]["path"]
    lora_name = loaded_loras[lora_id_version]["name"]
    
    try:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_weight=lora_weight)
        return pipe, f"✅ Applied LoRA: {lora_name} (weight: {lora_weight})"
    except Exception as e:
        return pipe, f"🚨 Error applying LoRA: {str(e)}"

def set_civitai_api_key(api_key):
    global civitai_api_key
    civitai_api_key = api_key
    return f"✅ CivitAI API key set: {api_key[:5]}..." if api_key else "⚠️ API key cleared"

def search_and_update_loras(query):
    global civitai_api_key
    results, message = search_civitai_loras(query, civitai_api_key)
    lora_choices = [("None", "None")] + results
    return gr.Dropdown.update(choices=lora_choices), message

def download_selected_lora(lora_id_version):
    global civitai_api_key
    if lora_id_version == "None":
        return "No LoRA selected"
    message = download_lora_from_civitai(lora_id_version, civitai_api_key)
    return message

def get_token_count(prompt):
    global tokenizer, token_limit
    if not tokenizer:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    tokens = tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"].shape[1]
    return f"{tokens}/{token_limit} tokens" + (" ⚠️" if tokens > 77 else "")

def update_token_limit(new_limit):
    global token_limit
    token_limit = int(new_limit)
    return f"Token limit updated to {token_limit}. Note: SDXL natively supports 77 tokens; excess tokens may be ignored."

# JavaScript fix for preview buttons
js = """
// Fix for image preview close buttons
function fixPreviewButtons() {
    // Wait for any image modal to appear
    const checkAndFixButtons = () => {
        const previewButtons = document.querySelectorAll('.gradio-modal button, .preview button');
        previewButtons.forEach(button => {
            button.style.position = 'absolute';
            button.style.top = '10px';
            button.style.right = '10px';
            button.style.zIndex = '100';
            button.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
            button.style.borderRadius = '50%';
            button.style.width = '30px';
            button.style.height = '30px';
            button.style.padding = '0';
        });
    };
    
    // Run periodically
    setInterval(checkAndFixButtons, 1000);
    
    // Also attach listeners to gallery items
    setTimeout(() => {
        const galleryItems = document.querySelectorAll('.gradio-gallery .thumbnail-item');
        galleryItems.forEach(item => {
            item.addEventListener('click', () => {
                setTimeout(checkAndFixButtons, 100);
            });
        });
    }, 2000);
}

// Run when DOM is loaded
document.addEventListener('DOMContentLoaded', fixPreviewButtons);
window.addEventListener('load', fixPreviewButtons);
"""

# Add all required functions here, before the UI definition
def load_controlnet(controlnet_type):
    global controlnet_models, controlnet_processors, current_controlnet, huggingface_token
    
    if controlnet_type == "None":
        current_controlnet = None
        return "ControlNet đã bị tắt"
    
    try:
        # Kiểm tra xem đã tải trước đó chưa
        if controlnet_type in controlnet_models:
            current_controlnet = controlnet_models[controlnet_type]
            return f"✅ Đã kích hoạt ControlNet: {controlnet_type}"
        
        # Tải ControlNet model
        if controlnet_type == "canny":
            model_id = "diffusers/controlnet-canny-sdxl-1.0"
        elif controlnet_type == "openpose":
            model_id = "thibaud/controlnet-openpose-sdxl-1.0"
        elif controlnet_type == "depth":
            model_id = "diffusers/controlnet-depth-sdxl-1.0"
        elif controlnet_type == "mlsd":
            model_id = "diffusers/controlnet-mlsd-sdxl-1.0"
        else:
            return f"❌ Loại ControlNet không hỗ trợ: {controlnet_type}"
        
        # Tải model với token nếu có
        controlnet = ControlNetModel.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            use_auth_token=huggingface_token
        )
        controlnet_models[controlnet_type] = controlnet
        current_controlnet = controlnet
        
        # Tải processor nếu cần
        if controlnet_type == "canny" and controlnet_processors["canny"] is None:
            controlnet_processors["canny"] = CannyDetector()
        elif controlnet_type == "openpose" and controlnet_processors["openpose"] is None:
            try:
                # Initialize body estimator first
                body_estimation = BodyEstimator()
                controlnet_processors["openpose"] = OpenposeDetector(body_estimation=body_estimation)
            except ImportError:
                return f"❌ Could not load OpenposeDetector. Please run: pip install controlnet-aux"
            except Exception as e:
                return f"❌ Error initializing OpenposeDetector: {str(e)}"
        elif controlnet_type == "mlsd" and controlnet_processors["mlsd"] is None:
            controlnet_processors["mlsd"] = MLSDdetector()
        elif controlnet_type == "hed" and controlnet_processors["hed"] is None:
            controlnet_processors["hed"] = HEDdetector()
        
        return f"✅ Đã tải và kích hoạt ControlNet: {controlnet_type}"
    except Exception as e:
        current_controlnet = None
        if "401 Client Error" in str(e):
            return f"🚨 Lỗi xác thực: Vui lòng thiết lập HF_TOKEN trong tab Settings. Chi tiết: {str(e)}"
        return f"🚨 Lỗi khi tải ControlNet: {str(e)}"

def process_controlnet_image(image, controlnet_type, low_threshold=100, high_threshold=200):
    global controlnet_processors
    
    if image is None:
        return None
    
    try:
        if controlnet_type == "canny":
            if controlnet_processors["canny"] is None:
                controlnet_processors["canny"] = CannyDetector()
            return controlnet_processors["canny"](image, low_threshold, high_threshold)
        
        elif controlnet_type == "openpose":
            if controlnet_processors["openpose"] is None:
                controlnet_processors["openpose"] = OpenposeDetector()
            return controlnet_processors["openpose"](image)
        
        elif controlnet_type == "mlsd":
            if controlnet_processors["mlsd"] is None:
                controlnet_processors["mlsd"] = MLSDdetector()
            return controlnet_processors["mlsd"](image)
        
        elif controlnet_type == "hed":
            if controlnet_processors["hed"] is None:
                controlnet_processors["hed"] = HEDdetector()
            return controlnet_processors["hed"](image)
        
        elif controlnet_type == "depth":
            # Depth sử dụng ảnh gốc
            return image
        
        else:
            return None
    except Exception as e:
        print(f"🚨 Lỗi xử lý ảnh ControlNet: {str(e)}")
        return None

def load_pipeline_with_controlnet(model_id, controlnet=None):
    global tokenizer, vae_model, huggingface_token
    
    if controlnet is None:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            use_safetensors=True,
            use_auth_token=huggingface_token
        )
    else:
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_id, 
            controlnet=controlnet, 
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

def generate_with_controlnet(prompt, negative_prompt, width, height, steps, guidance,
                         controlnet_image, controlnet_conditioning_scale, seed, num_images, webhook_url,
                         progress=gr.Progress(track_tqdm=False)):
    global pipe, current_model_id, tokenizer, current_controlnet, selected_model, selected_lora, lora_weight
    from time import time

    if current_controlnet is None:
        return [], "❌ ControlNet không được kích hoạt. Vui lòng chọn loại ControlNet trước."

    model_status = switch_model(selected_model, use_controlnet=True)
    logs = [model_status]
    
    # Lấy IP người dùng
    try:
        user_ip = requests.get("https://api.ipify.org").text
        logs.append(f"🌐 User IP: {user_ip}")
    except Exception as e:
        logs.append(f"⚠️ Could not fetch IP: {e}")
    
    # Apply LoRA if selected
    lora_message = "No LoRA selected"
    if selected_lora and selected_lora != "None":
        pipe, lora_message = load_lora_weights(pipe, selected_lora, lora_weight)
        logs.append(lora_message)

    if seed == 0:
        seed = random.randint(1, 1 << 30)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    output_dir = os.path.join("images", datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(output_dir, exist_ok=True)

    # Count tokens
    prompt_tokens = tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"].shape[1]
    neg_prompt_tokens = tokenizer(negative_prompt, return_tensors="pt", truncation=False)["input_ids"].shape[1]
    total_tokens = prompt_tokens + neg_prompt_tokens

    token_info = (
        f"🧠 Token usage:\n"
        f"> Prompt Tokens: {prompt_tokens} {'⚠️' if prompt_tokens > 77 else ''}\n"
        f"> Negative Prompt Tokens: {neg_prompt_tokens} {'⚠️' if neg_prompt_tokens > 77 else ''}\n"
        f"> Total Tokens: {total_tokens}\n"
        f"{'-'*30}"
    )
    logs.append(token_info)

    previews = []
    for i in range(num_images):
        start_time = time()

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=controlnet_image,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator
        ).images[0]

        elapsed = time() - start_time
        percent = int((i + 1) / num_images * 100)

        timestamp = datetime.now().strftime("%H%M%S_%f")
        filename = f"{clean_prompt(prompt)}_seed{seed}_{timestamp}.png"
        file_path = os.path.join(output_dir, filename)
        bordered_image = add_border(image)
        bordered_image.save(file_path)

        preview = create_preview(bordered_image, max_width=512)
        previews.append(preview)

        metadata = (
            f"🖼️ *Image Generated with ControlNet*\n"
            f"> Prompt: `{prompt}`\n"
            f"> Seed: `{seed}`\n"
            f"> Size: {width}x{height} | Steps: {steps} | Guidance: {guidance}\n"
            f"> ControlNet Scale: {controlnet_conditioning_scale}\n"
            f"> LoRA: {lora_message if selected_lora != 'None' else 'None'}"
        )
        upload_to_webhook(file_path, webhook_url, metadata)
        send_to_telegram(file_path, metadata)

        logs.append(
            f"✅ [{percent}%] Saved: {file_path} | Time: {elapsed:.2f}s"
        )

    # Reset LoRA fusion if one was applied
    if selected_lora and selected_lora != "None":
        pipe.unfuse_lora()

    return previews, "\n".join(logs)

# Gallery functions
def scan_images_directory():
    # Tạo thư mục images nếu chưa tồn tại
    os.makedirs("images", exist_ok=True)
    
    all_images = []
    for date_dir in os.listdir("images"):
        date_path = os.path.join("images", date_dir)
        if os.path.isdir(date_path):
            for img in glob.glob(os.path.join(date_path, "*.png")):
                all_images.append(img)
    
    all_images.sort(key=os.path.getmtime, reverse=True)
    return all_images

def extract_details_from_filename(filename):
    # Extract seed from filename
    seed_match = re.search(r'_seed(\d+)_', filename)
    seed = seed_match.group(1) if seed_match else "Unknown"
    
    # Extract prompt from filename (replacing underscore with space)
    prompt_part = os.path.basename(filename).split('_seed')[0]
    prompt = prompt_part.replace('_', ' ').strip()
    
    return prompt, seed

def search_images(search_term):
    all_images = scan_images_directory()
    
    if not search_term.strip():
        return all_images, "Tất cả hình ảnh"
    
    search_term = search_term.lower()
    filtered_images = []
    
    for img_path in all_images:
        filename = os.path.basename(img_path)
        if search_term in filename.lower():
            filtered_images.append(img_path)
    
    return filtered_images, f"Tìm thấy {len(filtered_images)} ảnh có chứa: {search_term}"

# Thiết lập HF_TOKEN
def set_huggingface_token(token):
    global huggingface_token
    huggingface_token = token
    huggingface_hub.login(token=token, add_to_git_credential=False)
    return f"✅ Hugging Face token đã được thiết lập: {token[:5]}..." if token else "⚠️ Token đã bị xóa"

# Lấy địa chỉ IP hiện tại
def get_ip_address():
    try:
        user_ip = requests.get("https://api.ipify.org").text
        return user_ip
    except Exception as e:
        print(f"⚠️ Could not fetch IP: {e}")
        return "Unknown IP"

def save_to_csb(file_path, is_video=False):
    """Copy generated files to CSB folder with same structure"""
    try:
        # Create CSB folder if it doesn't exist
        csb_root = "CSB"
        os.makedirs(csb_root, exist_ok=True)
        
        # Determine the relative path from the original folder
        if is_video:
            rel_path = os.path.relpath(file_path, "videos")
            target_dir = os.path.join(csb_root, "videos", os.path.dirname(rel_path))
        else:
            rel_path = os.path.relpath(file_path, "images")
            target_dir = os.path.join(csb_root, "images", os.path.dirname(rel_path))
            
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy the file to CSB folder
        target_path = os.path.join(csb_root, "videos" if is_video else "images", rel_path)
        shutil.copy2(file_path, target_path)
        
        # If running in Colab, also save to Google Drive
        if is_colab and os.path.exists('/content/drive/MyDrive'):
            drive_folder = 'videos' if is_video else 'images'
            save_to_google_drive(file_path, drive_folder, is_video)
        
        return f"Saved to CSB: {target_path}"
    except Exception as e:
        return f"Failed to save to CSB: {str(e)}"

def set_ngrok_token(selected_token, custom_token=""):
    global ngrok_token, ngrok_tokens
    
    # Xác định token nào sẽ được sử dụng
    if selected_token == "Custom" and custom_token:
        token_to_use = custom_token
        # Lưu token tùy chỉnh vào danh sách
        ngrok_tokens["Custom"] = custom_token
    else:
        token_to_use = ngrok_tokens[selected_token]
    
    ngrok_token = token_to_use
    
    try:
        # Cấu hình ngrok với token
        if token_to_use:
            ngrok.set_auth_token(token_to_use)
            return f"✅ Ngrok token đã được thiết lập: {token_to_use[:5]}..."
        else:
            return "⚠️ Ngrok token đã bị xóa hoặc không hợp lệ"
    except Exception as e:
        return f"❌ Lỗi khi thiết lập ngrok token: {str(e)}"

def setup_ngrok(port):
    global ngrok_token
    
    if not ngrok_token:
        return "⚠️ Ngrok token chưa được thiết lập. Vui lòng thiết lập trong tab Settings."
    
    try:
        # Create a tunnel with the specified port
        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url
        return f"✅ Ngrok tunnel created: {public_url}"
    except Exception as e:
        return f"❌ Lỗi khi tạo ngrok tunnel: {str(e)}"

def create_csb_folders():
    """Create CSB directories if they don't exist and print status"""
    try:
        os.makedirs("CSB", exist_ok=True)
        os.makedirs("CSB/images", exist_ok=True)
        os.makedirs("CSB/videos", exist_ok=True)
        print("✅ CSB folders created/verified successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating CSB folders: {str(e)}")
        return False

# Google Drive integration functions
def mount_google_drive():
    """Mount Google Drive if running in Colab"""
    if not is_colab:
        return "Not running in Google Colab, can't mount Google Drive directly"
    
    try:
        drive.mount('/content/drive')
        return "✅ Google Drive mounted successfully at /content/drive"
    except Exception as e:
        return f"❌ Error mounting Google Drive: {str(e)}"

def create_drive_directories():
    """Create directories in Google Drive for storing models and generations"""
    if not is_colab:
        return "Not running in Google Colab, can't access Google Drive directly"
    
    # Check if Google Drive is mounted
    if not os.path.exists('/content/drive/MyDrive'):
        return "❌ Google Drive not mounted. Please mount first."
    
    # Create directories for storing different types of files
    try:
        # Create main directory
        sdxl_dir = '/content/drive/MyDrive/SDXL_WebUI'
        os.makedirs(sdxl_dir, exist_ok=True)
        
        # Create subdirectories
        dirs = [
            os.path.join(sdxl_dir, 'loras'),
            os.path.join(sdxl_dir, 'vae'),
            os.path.join(sdxl_dir, 'images'),
            os.path.join(sdxl_dir, 'videos'),
            os.path.join(sdxl_dir, 'controlnet')
        ]
        
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
        
        return f"✅ Created directories in Google Drive at {sdxl_dir}"
    except Exception as e:
        return f"❌ Error creating Google Drive directories: {str(e)}"

def save_to_google_drive(local_path, drive_folder, is_video=False):
    """Copy a file to Google Drive"""
    if not is_colab:
        return "Not running in Google Colab, can't access Google Drive directly"
    
    # Check if Google Drive is mounted
    if not os.path.exists('/content/drive/MyDrive'):
        return "❌ Google Drive not mounted. Please mount first."
    
    try:
        # Create the base directory if it doesn't exist
        sdxl_dir = '/content/drive/MyDrive/SDXL_WebUI'
        os.makedirs(sdxl_dir, exist_ok=True)
        
        # Create the target directory if it doesn't exist
        target_dir = os.path.join(sdxl_dir, drive_folder)
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy the file
        filename = os.path.basename(local_path)
        target_path = os.path.join(target_dir, filename)
        shutil.copy2(local_path, target_path)
        
        return f"✅ Saved to Google Drive: {target_path}"
    except Exception as e:
        return f"❌ Error saving to Google Drive: {str(e)}"

def download_from_drive_url(drive_url, local_dir, filename=None):
    """Download a file from Google Drive using a shareable link"""
    if not has_gdown:
        return "❌ gdown library not installed. Please install with: pip install gdown"
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # If filename is not provided, we'll use the original filename
        output_path = os.path.join(local_dir, filename) if filename else local_dir
        
        # Download the file
        gdown.download(url=drive_url, output=output_path, quiet=False, fuzzy=True)
        
        # Verify if file was downloaded
        if filename and os.path.exists(output_path):
            return f"✅ Downloaded {filename} to {local_dir}"
        elif os.path.exists(local_dir):
            return f"✅ Downloaded file to {local_dir}"
        else:
            return "❌ Download may have failed"
    except Exception as e:
        return f"❌ Error downloading from Google Drive: {str(e)}"

def sync_generated_to_drive():
    """Sync all generated images and videos to Google Drive"""
    if not is_colab:
        return "Not running in Google Colab, can't access Google Drive directly"
    
    # Check if Google Drive is mounted
    if not os.path.exists('/content/drive/MyDrive'):
        return "❌ Google Drive not mounted. Please mount first."
    
    try:
        # Create the base directory in Google Drive
        drive_dir = '/content/drive/MyDrive/SDXL_WebUI'
        drive_images = os.path.join(drive_dir, 'images')
        drive_videos = os.path.join(drive_dir, 'videos')
        
        os.makedirs(drive_images, exist_ok=True)
        os.makedirs(drive_videos, exist_ok=True)
        
        # Sync CSB images
        if os.path.exists('CSB/images'):
            for date_dir in os.listdir('CSB/images'):
                date_path = os.path.join('CSB/images', date_dir)
                if os.path.isdir(date_path):
                    # Create corresponding directory in Google Drive
                    drive_date_dir = os.path.join(drive_images, date_dir)
                    os.makedirs(drive_date_dir, exist_ok=True)
                    
                    # Copy all images
                    image_count = 0
                    for img in glob.glob(os.path.join(date_path, '*.png')):
                        filename = os.path.basename(img)
                        drive_path = os.path.join(drive_date_dir, filename)
                        # Only copy if file doesn't exist or is newer
                        if not os.path.exists(drive_path) or os.path.getmtime(img) > os.path.getmtime(drive_path):
                            shutil.copy2(img, drive_path)
                            image_count += 1
        
        # Sync CSB videos
        video_count = 0
        if os.path.exists('CSB/videos'):
            for date_dir in os.listdir('CSB/videos'):
                date_path = os.path.join('CSB/videos', date_dir)
                if os.path.isdir(date_path):
                    # Create corresponding directory in Google Drive
                    drive_date_dir = os.path.join(drive_videos, date_dir)
                    os.makedirs(drive_date_dir, exist_ok=True)
                    
                    # Copy all videos
                    for vid in glob.glob(os.path.join(date_path, '*.mp4')):
                        filename = os.path.basename(vid)
                        drive_path = os.path.join(drive_date_dir, filename)
                        # Only copy if file doesn't exist or is newer
                        if not os.path.exists(drive_path) or os.path.getmtime(vid) > os.path.getmtime(drive_path):
                            shutil.copy2(vid, drive_path)
                            video_count += 1
        
        return f"✅ Synced files to Google Drive: {image_count} images and {video_count} videos"
    except Exception as e:
        return f"❌ Error syncing to Google Drive: {str(e)}"

def download_lora_to_drive(lora_id_version, drive_folder='/content/drive/MyDrive/SDXL_WebUI/loras'):
    """Download LoRA directly to Google Drive"""
    if not is_colab:
        return "Not running in Google Colab, can't access Google Drive directly"
    
    global loaded_loras, civitai_api_key
    
    if not civitai_api_key:
        return f"⚠️ CivitAI API key not provided"
    
    try:
        lora_id, version_id = lora_id_version.split(":")
        
        # Get download URL from CivitAI
        headers = {"Authorization": f"Bearer {civitai_api_key}"}
        response = requests.get(
            f"https://civitai.com/api/v1/model-versions/{version_id}",
            headers=headers
        )
        
        if response.status_code != 200:
            return f"❌ CivitAI API error: {response.status_code}"
        
        data = response.json()
        download_url = None
        
        for file in data.get("files", []):
            if file.get("primary"):
                download_url = file.get("downloadUrl")
                break
        
        if not download_url:
            return "❌ No download URL found for this LoRA"
        
        # Create directories
        os.makedirs(drive_folder, exist_ok=True)
        os.makedirs("loras", exist_ok=True)
        
        # Save filename
        safetensor_name = f"lora_{lora_id}.safetensors"
        drive_path = os.path.join(drive_folder, safetensor_name)
        local_path = os.path.join("loras", safetensor_name)
        
        # Download the file
        response = requests.get(download_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Save to both local and Google Drive
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Copy to Google Drive
        shutil.copy2(local_path, drive_path)
        
        # Add to loaded_loras dict
        lora_name = data.get("model", {}).get("name", f"LoRA {lora_id}")
        loaded_loras[lora_id_version] = {
            "path": local_path,
            "name": lora_name
        }
        
        return f"✅ Downloaded LoRA: {lora_name} to Google Drive ({os.path.getsize(local_path) / (1024*1024):.2f} MB)"
    except Exception as e:
        return f"🚨 Error downloading LoRA to Google Drive: {str(e)}"

def load_lora_from_drive(drive_path):
    """Load a LoRA model from Google Drive path"""
    if not is_colab:
        return "Not running in Google Colab, can't access Google Drive directly"
    
    if not os.path.exists(drive_path):
        return f"❌ File not found at: {drive_path}"
    
    try:
        # Get filename and create local path
        filename = os.path.basename(drive_path)
        local_path = os.path.join("loras", filename)
        
        # Copy from Drive to local
        os.makedirs("loras", exist_ok=True)
        shutil.copy2(drive_path, local_path)
        
        # Add to loaded_loras dict with a temporary key
        lora_name = os.path.splitext(filename)[0]
        temp_key = f"drive_{lora_name}"
        loaded_loras[temp_key] = {
            "path": local_path,
            "name": f"Drive: {lora_name}"
        }
        
        return f"✅ Loaded LoRA from Drive: {lora_name} ({os.path.getsize(local_path) / (1024*1024):.2f} MB)"
    except Exception as e:
        return f"🚨 Error loading LoRA from Drive: {str(e)}"

def load_vae_from_drive(drive_path):
    """Load a VAE model from Google Drive path"""
    global vae_model
    
    if not is_colab:
        return "Not running in Google Colab, can't access Google Drive directly"
    
    if not os.path.exists(drive_path):
        return f"❌ File not found at: {drive_path}"
    
    try:
        # Get filename and create local path
        filename = os.path.basename(drive_path)
        local_path = os.path.join("vae", filename)
        
        # Copy from Drive to local
        os.makedirs("vae", exist_ok=True)
        shutil.copy2(drive_path, local_path)
        
        # Load VAE from local path
        vae_model = AutoencoderKL.from_pretrained(
            local_path,
            torch_dtype=torch.float16
        )
        
        return f"✅ Loaded VAE from Drive: {filename} ({os.path.getsize(local_path) / (1024*1024):.2f} MB)"
    except Exception as e:
        vae_model = None
        return f"🚨 Error loading VAE from Drive: {str(e)}"

# --- UI ---
with gr.Blocks(theme=gr.themes.Base(), title="🎨 Stable Diffusion XL", css=css) as demo:
    gr.Markdown("<h2 style='text-align:center; color:#00b4d8;'>🎨 Stable Diffusion XL Web UI - Tab Mode</h2>")
    gr.HTML(f"<script>{js}</script>")

    webhook_url = gr.Textbox(label="📬 Discord Webhook", placeholder="https://discord.com/api/webhooks/...", interactive=True)

    with gr.Tabs():
        with gr.Tab("🎨 Text to Image"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(label="🎯 Prompt")
                    negative_prompt = gr.Textbox(label="🚫 Negative Prompt")
                    width = gr.Slider(512, 2048, value=768, step=64, label="Width")
                    height = gr.Slider(512, 2048, value=1024, step=64, label="Height")
                    steps = gr.Slider(10, 100, value=30, step=1, label="Steps")
                    guidance = gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance Scale")
                    seed = gr.Slider(0, 2**32-1, value=0, step=1, label="Seed (0 = random)")
                    num_images = gr.Slider(1, 10, value=1, step=1, label="Number of Images")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            upscale_checkbox = gr.Checkbox(label="Apply Upscaler", value=False)
                        with gr.Column(scale=1):
                            watermark_checkbox = gr.Checkbox(label="Apply Watermark", value=False)
                    
                    generate_btn = gr.Button("🚀 Generate Text to Image")
                    shutdown_btn = gr.Button("❌ Xoá Web")

                with gr.Column(scale=1):
                    gallery = gr.Gallery(label="🖼️ Preview", columns=1, rows=1, object_fit="contain", height=512)
                    log_output = gr.Textbox(label="📜 Log", lines=12)

            generate_btn.click(
                fn=generate_images,
                inputs=[
                    prompt, negative_prompt, width, height, steps, guidance, gr.State(0),
                    seed, gr.State(None), gr.State(False), num_images, webhook_url,
                    upscale_checkbox, watermark_checkbox
                ],
                outputs=[gallery, log_output]
            )

        with gr.Tab("🖼️ Image to Image"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_i2i = gr.Textbox(label="🎯 Prompt", elem_classes=["prompt_input"], lines=3)
                    negative_prompt_i2i = gr.Textbox(label="🚫 Negative Prompt", elem_classes=["prompt_input"], lines=3)
                    
                    i2i_token_counter = gr.HTML(value="<div class='token-counter'><span>Prompt tokens: 0</span></div>")
                    i2i_negative_token_counter = gr.HTML(value="<div class='token-counter'><span>Negative prompt tokens: 0</span></div>")
                    
                    init_image = gr.Image(label="📤 Upload Image", type="pil")
                    strength = gr.Slider(0, 1, value=0.6, step=0.05, label="Strength")
                    steps_i2i = gr.Slider(10, 100, value=30, step=1, label="Steps")
                    guidance_i2i = gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance Scale")
                    seed_i2i = gr.Slider(0, 2**32-1, value=0, step=1, label="Seed (0 = random)")
                    num_images_i2i = gr.Slider(1, 10, value=1, step=1, label="Number of Images")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            upscale_checkbox_i2i = gr.Checkbox(label="Apply Upscaler", value=False)
                        with gr.Column(scale=1):
                            watermark_checkbox_i2i = gr.Checkbox(label="Apply Watermark", value=False)
                            
                    generate_btn_i2i = gr.Button("🚀 Generate Image to Image", variant="primary")

                with gr.Column(scale=1):
                    gallery_i2i = gr.Gallery(label="🖼️ Preview", columns=1, rows=1, object_fit="contain", height=512)
                    log_output_i2i = gr.Textbox(label="📜 Log", lines=12)

            # Add token counters for img2img
            prompt_i2i.input(
                fn=get_token_count,
                inputs=prompt_i2i,
                outputs=i2i_token_counter
            )
            
            negative_prompt_i2i.input(
                fn=get_token_count,
                inputs=negative_prompt_i2i,
                outputs=i2i_negative_token_counter
            )

            generate_btn_i2i.click(
                fn=generate_images,
                inputs=[
                    prompt_i2i, negative_prompt_i2i, gr.State(768), gr.State(1024), steps_i2i,
                    guidance_i2i, strength, seed_i2i, init_image, gr.State(True), num_images_i2i, webhook_url,
                    upscale_checkbox_i2i, watermark_checkbox_i2i
                ],
                outputs=[gallery_i2i, log_output_i2i]
            )

        with gr.Tab("🎬 Image to Video"):
            with gr.Row():
                i2v_input_image = gr.Image(type="pil", label="Input Image")
                i2v_output_video = gr.Video(label="Output Video")

            with gr.Row():
                i2v_prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", elem_classes=["prompt_input"], lines=3)
                i2v_negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here...", elem_classes=["prompt_input"], lines=3)

            i2v_token_counter = gr.HTML(value="<div class='token-counter'><span>Prompt tokens: 0</span></div>")
            i2v_negative_token_counter = gr.HTML(value="<div class='token-counter'><span>Negative prompt tokens: 0</span></div>")

            with gr.Row():
                with gr.Column():
                    i2v_num_inference_steps = gr.Slider(minimum=1, maximum=150, value=50, step=1, label="Steps")
                    i2v_guidance_scale = gr.Slider(minimum=1, maximum=20, value=6, step=0.1, label="Guidance Scale")
                    i2v_num_frames = gr.Slider(minimum=10, maximum=100, value=49, step=1, label="Number of Frames")
                    i2v_fps = gr.Slider(minimum=1, maximum=30, value=8, step=1, label="FPS")
                
                with gr.Column():
                    i2v_log_output = gr.Textbox(label="📜 Log", lines=12)
            
            i2v_generate = gr.Button("Generate Video", variant="primary")
            
            # Wire up the token counters
            i2v_prompt.input(
                fn=get_token_count,
                inputs=i2v_prompt,
                outputs=i2v_token_counter
            )
            
            i2v_negative_prompt.input(
                fn=get_token_count,
                inputs=i2v_negative_prompt,
                outputs=i2v_negative_token_counter
            )
            
            # Wire up the generate button
            i2v_generate.click(
                fn=generate_video,
                inputs=[
                    i2v_prompt, i2v_negative_prompt, i2v_input_image,
                    i2v_num_inference_steps, i2v_guidance_scale, i2v_num_frames, i2v_fps
                ],
                outputs=[i2v_output_video, i2v_log_output]
            )
        with gr.Tab("🔍 ControlNet"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_cn = gr.Textbox(label="🎯 Prompt")
                    negative_prompt_cn = gr.Textbox(label="🚫 Negative Prompt")
                    
                    with gr.Row():
                        width_cn = gr.Slider(512, 2048, value=768, step=64, label="Width")
                        height_cn = gr.Slider(512, 2048, value=1024, step=64, label="Height")
                    
                    controlnet_type = gr.Dropdown(
                        label="🧠 ControlNet Type",
                        choices=[
                            "None",
                            "canny",
                            "openpose",
                            "depth",
                            "mlsd"
                        ],
                        value="None",
                        interactive=True
                    )
                    
                    with gr.Group():
                        controlnet_image = gr.Image(label="📤 Control Image", type="pil")
                        process_btn = gr.Button("🔄 Process Control Image")
                        controlnet_preview = gr.Image(label="📊 Processed Control Image", type="pil")
                    
                    with gr.Row():
                        with gr.Column():
                            controlnet_scale = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="ControlNet Scale")
                            steps_cn = gr.Slider(10, 100, value=30, step=1, label="Steps")
                    
                    with gr.Row():
                        with gr.Column():
                            guidance_cn = gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance Scale")
                            seed_cn = gr.Slider(0, 2**32-1, value=0, step=1, label="Seed (0 = random)")
                    
                    num_images_cn = gr.Slider(1, 4, value=1, step=1, label="Number of Images")
                    load_controlnet_btn = gr.Button("📥 Load ControlNet")
                    generate_btn_cn = gr.Button("🚀 Generate with ControlNet")
                
                with gr.Column(scale=1):
                    gallery_cn = gr.Gallery(label="🖼️ Preview", columns=1, rows=1, object_fit="contain", height=512)
                    log_output_cn = gr.Textbox(label="📜 Log", lines=12)
            
            # ControlNet event handlers
            load_controlnet_btn.click(
                fn=load_controlnet,
                inputs=[controlnet_type],
                outputs=[log_output_cn]
            )
            
            process_btn.click(
                fn=process_controlnet_image,
                inputs=[controlnet_image, controlnet_type],
                outputs=[controlnet_preview]
            )
            
            generate_btn_cn.click(
                fn=generate_with_controlnet,
                inputs=[
                    prompt_cn, negative_prompt_cn, width_cn, height_cn, steps_cn, 
                    guidance_cn, controlnet_preview, controlnet_scale, seed_cn,
                    num_images_cn, webhook_url
                ],
                outputs=[gallery_cn, log_output_cn]
            )
        
        with gr.Tab("🖼️ Gallery"):
            with gr.Row():
                search_term = gr.Textbox(label="🔍 Tìm kiếm ảnh", placeholder="Nhập từ khóa tìm kiếm...")
                search_btn = gr.Button("🔍 Tìm kiếm")
            
            with gr.Row():
                search_results = gr.Textbox(label="Kết quả tìm kiếm", interactive=False)
            
            with gr.Row():
                gallery_browser = gr.Gallery(
                    label="Thư viện hình ảnh", 
                    columns=4, 
                    rows=3, 
                    object_fit="contain", 
                    height=600
                )
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Làm mới thư viện")
            
            # Gallery event handlers
            search_btn.click(
                fn=search_images,
                inputs=[search_term],
                outputs=[gallery_browser, search_results]
            )
            
            refresh_btn.click(
                fn=lambda: (scan_images_directory(), "Đã làm mới thư viện"),
                inputs=[],
                outputs=[gallery_browser, search_results]
            )
        
        with gr.Tab("☁️ Google Drive"):
            gr.Markdown("### 📂 Google Drive Integration")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🔄 Mount and Setup")
                    mount_drive_btn = gr.Button("🔗 Mount Google Drive")
                    create_folders_btn = gr.Button("📁 Create Directories in Drive")
                    drive_status = gr.Textbox(label="Status", interactive=False, lines=3)
                    
                    gr.Markdown("#### 🔄 Sync Files")
                    sync_to_drive_btn = gr.Button("🔄 Sync All Generated Files to Drive")
                    sync_status = gr.Textbox(label="Sync Status", interactive=False, lines=3)
                    
                with gr.Column():
                    gr.Markdown("#### 📥 Download From Drive")
                    drive_url = gr.Textbox(label="Google Drive URL or File ID", placeholder="https://drive.google.com/file/d/...")
                    drive_filetype = gr.Dropdown(
                        label="File Type",
                        choices=[
                            "LoRA Safetensors",
                            "VAE", 
                            "Other"
                        ],
                        value="LoRA Safetensors"
                    )
                    download_from_drive_btn = gr.Button("📥 Download From Drive")
                    download_status = gr.Textbox(label="Download Status", interactive=False, lines=3)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📤 Load From Drive (for Colab)")
                    gr.Markdown("""
                    If you're running in Google Colab, you can load models directly from your Drive.
                    The folders should be organized as:
                    - /content/drive/MyDrive/SDXL_WebUI/loras/ - for LoRA models
                    - /content/drive/MyDrive/SDXL_WebUI/vae/ - for VAE models
                    """)
                    
                    drive_lora_path = gr.Textbox(
                        label="LoRA Path in Drive",
                        placeholder="/content/drive/MyDrive/SDXL_WebUI/loras/my_lora.safetensors",
                        interactive=True
                    )
                    
                    load_drive_lora_btn = gr.Button("🔄 Load LoRA from Drive")
                    
                    drive_vae_path = gr.Textbox(
                        label="VAE Path in Drive",
                        placeholder="/content/drive/MyDrive/SDXL_WebUI/vae/my_vae.safetensors",
                        interactive=True
                    )
                    
                    load_drive_vae_btn = gr.Button("🔄 Load VAE from Drive")
                    drive_load_status = gr.Textbox(label="Load Status", interactive=False, lines=3)
            
            # Event handlers for Google Drive tab
            mount_drive_btn.click(
                fn=mount_google_drive,
                inputs=[],
                outputs=[drive_status]
            )
            
            create_folders_btn.click(
                fn=create_drive_directories,
                inputs=[],
                outputs=[drive_status]
            )
            
            sync_to_drive_btn.click(
                fn=sync_generated_to_drive,
                inputs=[],
                outputs=[sync_status]
            )
            
            download_from_drive_btn.click(
                fn=lambda url, filetype: download_from_drive_url(
                    url,
                    "loras" if filetype == "LoRA Safetensors" else "vae" if filetype == "VAE" else "downloads",
                    None
                ),
                inputs=[drive_url, drive_filetype],
                outputs=[download_status]
            )
            
            load_drive_lora_btn.click(
                fn=lambda path: load_lora_from_drive(path),
                inputs=[drive_lora_path],
                outputs=[drive_load_status]
            )
            
            load_drive_vae_btn.click(
                fn=lambda path: load_vae_from_drive(path),
                inputs=[drive_vae_path],
                outputs=[drive_load_status]
            )
        
        with gr.Tab("⚙️ Settings"):
            with gr.Row():
                with gr.Column():
                    # Token settings
                    gr.Markdown("### 🔢 Token Settings")
                    token_limit_slider = gr.Slider(
                        minimum=77, maximum=500, value=150, step=1,
                        label="Token Display Limit (Default SDXL: 77)"
                    )
                    token_limit_info = gr.Markdown(
                        """
                        **Note**: While you can increase the displayed token limit, SDXL's native limit is 77 tokens.  
                        Tokens beyond the first 77 may be ignored or have reduced influence on generation.
                        """
                    )
                    update_token_limit_btn = gr.Button("📝 Update Token Limit")
                
                with gr.Column():
                    # VAE Settings
                    gr.Markdown("### 🧩 VAE Settings")
                    vae_selector = gr.Dropdown(
                        label="Select VAE",
                        choices=[
                            "None",
                            "stabilityai/sd-vae-ft-mse",
                            "stabilityai/sdxl-vae",
                            "Linaqruf/anime-vae",
                            "stablediffusionapi/anything-v5-vae",
                            "hakurei/waifu-diffusion-vae",
                            "Mahou/Pastel-Mix-VAE",
                            "gsdf/CounterfeitXL-VAE",
                            "openai/consistency-decoder",
                            "sayakpaul/sd-vae-ft-ema-diffusers",
                            "iZELX1/SunshineMix-VAE",
                            "stabilityai/sd-x2-latent-upscaler-vae",
                            "madebyollin/sdxl-vae-fp16-fix",
                            "furryai/vae-ft-mse",
                            "johnslegers/EroticVAE",
                            "NagisaZj/VAE_Anime",
                            "ringhyacinth/novelai-vae-fp16-fix",
                            "VanillaMilk/EroChan-VAE",
                            "BunnyMagic/ShinyXLVAE",
                            "UhriG/SakumiVAE"
                        ],
                        value="None",
                        interactive=True
                    )
                    load_vae_btn = gr.Button("📥 Load VAE")
                    vae_status = gr.Textbox(label="VAE Status", interactive=False)
            
            with gr.Row():
                with gr.Column():
                    # Scheduler Settings
                    gr.Markdown("### 🔄 Scheduler Settings")
                    scheduler_selector = gr.Dropdown(
                        label="Select Scheduler",
                        choices=list(scheduler_list.keys()),
                        value=current_scheduler,
                        interactive=True
                    )
                    set_scheduler_btn = gr.Button("🔄 Change Scheduler")
                    scheduler_status = gr.Textbox(label="Scheduler Status", interactive=False)
                
                with gr.Column():
                    # Upscaler Settings
                    gr.Markdown("### 🔍 Upscaler Settings")
                    upscaler_selector = gr.Dropdown(
                        label="Select Upscaler Model",
                        choices=list(upscaler_models.keys()),
                        value="None",
                        interactive=True
                    )
                    outscale_factor = gr.Slider(
                        minimum=1, maximum=4, value=2, step=0.25,
                        label="Scale Factor"
                    )
                    load_upscaler_btn = gr.Button("📥 Load Upscaler")
                    upscaler_status = gr.Textbox(label="Upscaler Status", interactive=False)
        
            with gr.Row():
                with gr.Column():
                    # Hugging Face Settings
                    gr.Markdown("### 🤗 Hugging Face Settings")
                    huggingface_token_input = gr.Textbox(
                        label="Hugging Face Token (HF_TOKEN)", 
                        placeholder="Nhập HF_TOKEN của bạn ở đây",
                        type="password",
                        value=huggingface_token
                    )
                    set_hf_token_btn = gr.Button("💾 Lưu Token")
                    hf_token_status = gr.Textbox(label="Trạng thái", interactive=False)
                    gr.Markdown(
                        """
                        **Lưu ý**: Token Hugging Face cần thiết để tải một số mô hình ControlNet.
                        Bạn có thể đăng ký và lấy token tại [Hugging Face](https://huggingface.co/settings/tokens)
                        """
                    )

                with gr.Column():
                    # Ngrok Settings
                    gr.Markdown("### 🌐 Ngrok Settings")
                    ngrok_token_dropdown = gr.Dropdown(
                        label="Chọn Ngrok Token Có Sẵn",
                        choices=list(ngrok_tokens.keys()),
                        value="Default"
                    )
                    ngrok_token_input = gr.Textbox(
                        label="Custom Ngrok Auth Token", 
                        placeholder="Nhập token tùy chỉnh của bạn ở đây nếu chọn 'Custom'",
                        type="password",
                        value=""
                    )
                    ngrok_port_input = gr.Number(
                        label="Port Number",
                        value=7860,
                        precision=0
                    )
                    set_ngrok_token_btn = gr.Button("💾 Lưu Ngrok Token")
                    start_ngrok_btn = gr.Button("🚀 Khởi động Ngrok Tunnel")
                    ngrok_status = gr.Textbox(label="Trạng thái Ngrok", interactive=False)
                    gr.Markdown(
                        """
                        **Note**: Ngrok allows you to expose your local server to the internet.
                        Register at [Ngrok](https://dashboard.ngrok.com/) to get your auth token.
                        """
                    )

            with gr.Row():
                with gr.Column():
                    # LoRA Search
                    gr.Markdown("### 🔍 LoRA Settings")
                    civitai_api_key_input = gr.Textbox(
                        label="CivitAI API Key", 
                        placeholder="Enter your CivitAI API key here",
                        type="password",
                        value=civitai_api_key
                    )
                    set_api_key_btn = gr.Button("💾 Save API Key")
                    lora_search_query = gr.Textbox(
                        label="Search LoRAs on CivitAI", 
                        placeholder="Enter search term"
                    )
                    search_lora_btn = gr.Button("🔍 Search")
                    lora_search_results = gr.Textbox(label="Search Results")
                
                with gr.Column():
                    # LoRA Selection and Download
                    gr.Markdown("### 📥 LoRA Download")
                    lora_selector = gr.Dropdown(
                        label="Select LoRA",
                        choices=[("None", "None")],
                        value="None",
                        interactive=True
                    )
                    download_lora_btn = gr.Button("📥 Download Selected LoRA")
                    lora_download_status = gr.Textbox(label="Download Status")
                    lora_weight = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.8, step=0.05,
                        label="LoRA Weight"
                    )
                    
                with gr.Column():
                    # Watermark Settings
                    gr.Markdown("### 🖼️ Watermark Settings")
                    watermark_toggle = gr.Checkbox(
                        label="Enable Watermark",
                        value=watermark_enabled,
                        interactive=True
                    )
                    watermark_text_input = gr.Textbox(
                        label="Watermark Text",
                        placeholder="Text to show as watermark",
                        value=watermark_text
                    )
                    watermark_opacity_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=watermark_opacity, step=0.05,
                        label="Watermark Opacity"
                    )
                    set_watermark_btn = gr.Button("🖼️ Apply Watermark Settings")
                    watermark_status = gr.Textbox(label="Watermark Status", interactive=False)

        with gr.Tab("🔧 Enhance"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔍 Upscale & Enhance Image")
                    enhance_image_input = gr.Image(label="Upload Image to Enhance", type="pil")
                    
                    with gr.Row():
                        with gr.Column():
                            enhance_upscaler = gr.Dropdown(
                                label="Select Upscaler",
                                choices=list(upscaler_models.keys()),
                                value="RealESRGAN_x4plus_anime"
                            )
                            enhance_scale = gr.Slider(
                                minimum=1, maximum=4, value=2, step=0.25,
                                label="Scale Factor"
                            )
                        
                        with gr.Column():
                            enhance_watermark_toggle = gr.Checkbox(
                                label="Add Watermark",
                                value=False
                            )
                            enhance_watermark_text = gr.Textbox(
                                label="Watermark Text",
                                value=watermark_text
                            )
                            enhance_watermark_opacity = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.3, step=0.05,
                                label="Watermark Opacity"
                            )
                    
                    enhance_btn = gr.Button("✨ Enhance Image", variant="primary")
                
                with gr.Column():
                    enhance_output = gr.Image(label="Enhanced Image")
                    enhance_log = gr.Textbox(label="Enhancement Log", lines=8)
            
            # Event handlers for upscaler and settings
            enhance_btn.click(
                fn=lambda img, model, scale, add_wm, wm_text, wm_opacity: process_enhance_image(img, model, scale, add_wm, wm_text, wm_opacity),
                inputs=[
                    enhance_image_input, enhance_upscaler, enhance_scale,
                    enhance_watermark_toggle, enhance_watermark_text, enhance_watermark_opacity
                ],
                outputs=[enhance_output, enhance_log]
            )
            
            # Kết nối cho scheduler và upscaler
            set_scheduler_btn.click(
                fn=set_scheduler,
                inputs=[scheduler_selector],
                outputs=[scheduler_status]
            )
            
            load_upscaler_btn.click(
                fn=load_upscaler,
                inputs=[upscaler_selector],
                outputs=[upscaler_status]
            )
            
            # Kết nối cho watermark
            set_watermark_btn.click(
                fn=toggle_watermark,
                inputs=[watermark_toggle, watermark_text_input, watermark_opacity_slider],
                outputs=[watermark_status]
            )
            
            # Event handlers for Settings tab
            set_hf_token_btn.click(
                fn=set_huggingface_token,
                inputs=[huggingface_token_input],
                outputs=[hf_token_status]
            )
            
            set_ngrok_token_btn.click(
                fn=set_ngrok_token,
                inputs=[ngrok_token_dropdown, ngrok_token_input],
                outputs=[ngrok_status]
            )
            
            start_ngrok_btn.click(
                fn=setup_ngrok,
                inputs=[ngrok_port_input],
                outputs=[ngrok_status]
            )

    set_api_key_btn.click(
        fn=set_civitai_api_key,
        inputs=[civitai_api_key_input],
        outputs=[civitai_api_key_input]
    )
    
    load_vae_btn.click(
        fn=load_vae,
        inputs=[vae_selector],
        outputs=[vae_status]
    )
    
    search_lora_btn.click(
        fn=search_and_update_loras,
        inputs=[lora_search_query],
        outputs=[lora_selector, lora_search_results]
    )
    
    download_lora_btn.click(
        fn=download_selected_lora,
        inputs=[lora_selector],
        outputs=[lora_download_status]
    )
            
    update_token_limit_btn.click(
        fn=update_token_limit,
        inputs=[token_limit_slider],
        outputs=[token_limit_info]
    )

    shutdown_btn.click(fn=shutdown)

# Initial setup
# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("loras", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("controlnet", exist_ok=True)
create_csb_folders()

# Load tokenizer for token counting
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# Initial model load
pipe = load_pipeline(False, selected_model)

# Load VAE if specified
if selected_vae and selected_vae != "None":
    load_vae(selected_vae)

# Đăng nhập với Hugging Face token
huggingface_hub.login(token=huggingface_token, add_to_git_credential=False)

# Scan gallery on startup
scan_images_directory()

# Configure ngrok

def start_ngrok_automatically():
    global ngrok_token, ngrok_tokens
    
    try:
        # Nếu token rỗng, sử dụng token mặc định
        if not ngrok_token:
            ngrok_token = ngrok_tokens["Default"]
            
        # Thiết lập ngrok auth token
        ngrok.set_auth_token(ngrok_token)
        
        # Tạo tunnel
        port = 7860  # Default Gradio port
        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url
        
        print(f"✅ Ngrok tunnel khởi động: {public_url}")
        return public_url
    except Exception as e:
        print(f"❌ Lỗi khởi động ngrok tunnel: {str(e)}")
        return None

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

def add_watermark(image, text=None, opacity=0.3):
    """Thêm watermark vào hình ảnh"""
    global watermark_text, watermark_opacity
    
    if text is None:
        text = watermark_text
    
    if opacity is None:
        opacity = watermark_opacity
    
    if not text.strip():
        return image
    
    try:
        # Tạo bản sao để không ảnh hưởng đến hình ảnh gốc
        img = image.copy()
        
        # Tạo watermark layer
        watermark = Image.new('RGBA', img.size, (0, 0, 0, 0))
        
        # Chuẩn bị để vẽ lên watermark layer
        draw = ImageDraw.Draw(watermark)
        
        # Cố gắng load font, nếu không có thì dùng font mặc định
        try:
            font_size = int(min(img.width, img.height) / 20)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Tính toán vị trí và kích thước văn bản - tương thích với nhiều phiên bản PIL
        if hasattr(draw, 'textsize'):
            text_width, text_height = draw.textsize(text, font=font)
        else:
            bbox = font.getbbox(text)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Vị trí ở góc dưới bên phải
        position = (img.width - text_width - 10, img.height - text_height - 10)
        
        # Vẽ văn bản với độ mờ
        draw.text(position, text, font=font, fill=(255, 255, 255, int(255 * opacity)))
        
        # Chuyển đổi hình ảnh gốc sang RGBA nếu cần
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Kết hợp watermark với hình ảnh gốc
        result = Image.alpha_composite(img, watermark)
        
        # Chuyển trở lại mode gốc nếu cần
        if image.mode != 'RGBA':
            result = result.convert(image.mode)
        
        return result
    except Exception as e:
        print(f"Lỗi khi thêm watermark: {str(e)}")
        return image



# Sửa đổi hàm generate_images để áp dụng watermark và upscaler
def process_with_upscaler_and_watermark(image):
    """Xử lý ảnh với upscaler và watermark"""
    global upscaler_model, watermark_enabled
    
    result_message = ""
    
    # Upscale image nếu đã tải upscaler
    if upscaler_model is not None:
        image, upscale_msg = upscale_image(image)
        result_message += upscale_msg + "\n"
    
    # Thêm watermark nếu đã bật
    if watermark_enabled:
        image = add_watermark(image)
        result_message += f"✅ Đã thêm watermark. Text: '{watermark_text}'\n"
    
    return image, result_message

def process_enhance_image(image, model_name, scale_factor, add_watermark, watermark_text, watermark_opacity):
    """Xử lý hình ảnh với upscaler và watermark được chọn"""
    if image is None:
        return None, "❌ Không có hình ảnh nào được tải lên"
    
    logs = []
    result_image = image
    
    # Tải upscaler model nếu chưa có
    global upscaler_model, upscaler_models
    current_model = None
    
    if upscaler_model is not None:
        if upscaler_models[model_name]["model_path"] == upscaler_model.model.model_path:
            current_model = model_name
    
    if current_model != model_name:
        load_result = load_upscaler(model_name)
        logs.append(load_result)
    
    # Upscale hình ảnh
    if model_name != "None" and upscaler_model is not None:
        result_image, upscale_msg = upscale_image(result_image, scale_factor)
        logs.append(upscale_msg)
    
    # Thêm watermark nếu được chọn
    if add_watermark and watermark_text.strip():
        result_image = add_watermark(result_image, watermark_text, watermark_opacity)
        logs.append(f"✅ Đã thêm watermark: '{watermark_text}'")
    
    # Tạo thư mục và lưu kết quả
    output_dir = os.path.join("enhanced", datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%H%M%S_%f")
    filename = f"enhanced_{timestamp}.png"
    file_path = os.path.join(output_dir, filename)
    
    result_image.save(file_path)
    logs.append(f"✅ Đã lưu hình ảnh tại: {file_path}")
    
    return result_image, "\n".join(logs)

def get_local_ip():
    """Get the local network IP address"""
    try:
        # Create a socket connection to determine the local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        print(f"⚠️ Could not get local IP: {e}")
        return "127.0.0.1"

# Start the interface with server configuration
if __name__ == "__main__":
    # Check if we're in Kaggle
    in_kaggle = os.path.exists("/kaggle/input")
    
    # Start ngrok tunnel
    ngrok_url = start_ngrok_automatically()
    print(f"🌐 Remote access URL (ngrok): {ngrok_url}")
    
    # Print local access information
    local_ip = get_local_ip()
    print(f"💻 Local access URL: http://127.0.0.1:7860")
    print(f"🖥️ Local network URL: http://{local_ip}:7860")
    
    if in_kaggle:
        print("Running in Kaggle environment...")
        # In Kaggle, use the built-in share option
        demo.launch(server_name="0.0.0.0", share=True, show_error=True)
    else:
        # Default local run with ngrok already connected
        demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
else:
    # When imported as a module, don't auto-launch
    pass
