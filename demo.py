import torch
import gradio as gr
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import kornia
from model import HistColorNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HistColorNet().to(device)
model.load_state_dict(torch.load('best_model_lpips.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def colorize(img):
    # img: PIL Image (RGB) – даже если ч/б, будет 3 канала
    img_tensor = transform(img).unsqueeze(0).to(device)   # (1,3,256,256)
    # Конвертируем RGB -> Lab (kornia)
    lab = kornia.color.rgb_to_lab(img_tensor)             # (1,3,256,256)
    L = lab[:, 0:1, :, :]                                 # (1,1,256,256) в [0,100]
    # Нормализуем L в [-1,1] как требует модель
    L_norm = (L / 50.0) - 1.0
    # Вход для модели: три канала, все равны L_norm
    gray = L_norm.repeat(1, 3, 1, 1)
    with torch.no_grad():
        ab_pred = model(gray)                             # (1,2,256,256) в [-1,1]
    # Денормализуем для обратного преобразования Lab->RGB
    L_denorm = (L_norm + 1.0) * 50.0                      # обратно в [0,100]
    ab_denorm = ab_pred * 110.0                           # в [-110,110]
    lab_full = torch.cat([L_denorm, ab_denorm], dim=1)    # (1,3,256,256)
    rgb = kornia.color.lab_to_rgb(lab_full)               # (1,3,256,256) в [0,1]
    # Преобразуем в numpy для Gradio
    rgb_img = rgb.squeeze(0).permute(1,2,0).cpu().numpy()
    rgb_img = (rgb_img * 255).astype(np.uint8)
    return rgb_img

gr.Interface(
    fn=colorize,
    inputs=gr.Image(type="pil", label="Ч/б фото"),
    outputs=gr.Image(type="numpy", label="Цветное фото"),
    title="HistColorNet – колоризация"
).launch()