# SDXL WebUI

Giao diện web dễ sử dụng cho Stable Diffusion XL, hỗ trợ nhiều tính năng như Text-to-Image, Image-to-Image, ControlNet, Upscaling và nhiều tính năng khác.

## Tính năng

- **Text-to-Image**: Tạo hình từ mô tả văn bản
- **Image-to-Image**: Chỉnh sửa hình ảnh có sẵn
- **ControlNet**: Điều khiển quá trình tạo hình với hình ảnh tham chiếu
- **Video Generation**: Tạo video từ hình ảnh
- **Upscaling**: Nâng cao chất lượng hình ảnh
- **LoRA Support**: Hỗ trợ tải và sử dụng LoRA từ CivitAI và Hugging Face
- **Google Drive Integration**: Tự động lưu và đồng bộ với Google Drive

## Cách sử dụng trên Google Colab

1. Tạo một notebook Colab mới
2. Sao chép và dán code từ `colab_setup.py` vào cell đầu tiên và chạy
3. Sao chép và dán code từ `colab_runner.py` vào cell thứ hai, cập nhật `github_repo` và chạy

## Cài đặt trên máy tính cá nhân

```bash
git clone https://github.com/your-username/sdxl-webui.git
cd sdxl-webui
pip install -r requirements.txt
python webui.py
```

## Yêu cầu

- Python 3.8+
- CUDA compatible GPU with 8+ GB VRAM
- PyTorch 2.0+

## Lưu ý

- Cần token Hugging Face để tải một số model
- Cần token Ngrok để truy cập từ xa
- Có thể cấu hình gửi thông báo qua Telegram

## License

MIT