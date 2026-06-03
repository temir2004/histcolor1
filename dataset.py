import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset  # <-- ЭТО БЫЛО ПРОПУЩЕНО
from augmentations import historical_augmentation


# --- Функции для перевода RGB в LAB (без зависимости от версии torchvision) ---
def rgb_to_lab(img_rgb):
    """Перевод изображения из RGB в LAB"""
    img_rgb = img_rgb.float()
    # Матрица перевода RGB -> XYZ
    rgb_to_xyz = torch.tensor([[0.412453, 0.357580, 0.180423],
                               [0.212671, 0.715160, 0.072169],
                               [0.019334, 0.119193, 0.950227]], device=img_rgb.device)
    # Применяем преобразование
    img_xyz = torch.tensordot(img_rgb.permute(1, 2, 0), rgb_to_xyz.T, dims=1).permute(2, 0, 1)

    epsilon = 0.008856
    kappa = 903.3
    x, y, z = img_xyz[0], img_xyz[1], img_xyz[2]
    xr, yr, zr = 0.950456, 1.0, 1.088754

    fx = torch.where(x / xr > epsilon, torch.pow((x / xr), 1 / 3), (kappa * x / xr + 16) / 116)
    fy = torch.where(y / yr > epsilon, torch.pow((y / yr), 1 / 3), (kappa * y / yr + 16) / 116)
    fz = torch.where(z / zr > epsilon, torch.pow((z / zr), 1 / 3), (kappa * z / zr + 16) / 116)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    # Нормализация для устойчивости обучения
    a = torch.clamp(a / 110.0, -1, 1)
    b = torch.clamp(b / 110.0, -1, 1)
    L = L / 50.0 - 1.0

    return torch.stack([L, a, b])


class ColorizationDataset(Dataset):
    def __init__(self, root_dir, size=256, use_synthetic_degradation=True):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.use_synthetic_degradation = use_synthetic_degradation
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img_np = np.array(img)

        # Синтетическое старение
        if self.use_synthetic_degradation and np.random.rand() > 0.5:
            img_np = historical_augmentation(img_np)
            img = Image.fromarray(img_np)

        img_tensor = self.transform(img)  # [0, 1]
        img_lab = rgb_to_lab(img_tensor)  # [L, a, b]

        L = img_lab[0:1, :, :]  # канал яркости
        ab = img_lab[1:3, :, :]  # цветовые каналы
        gray = L.repeat(3, 1, 1)  # вход для сети (3 канала L)

        return gray, ab


def lab_to_rgb(L, ab):
    """
    L: (B,1,H,W) в диапазоне [-1,1]
    ab: (B,2,H,W) в диапазоне [-1,1]
    returns: (B,3,H,W) RGB в диапазоне [0,1]
    """
    # Денормализация
    L_denorm = (L + 1.0) * 50.0
    a_denorm = ab[:, 0:1, :, :] * 110.0
    b_denorm = ab[:, 1:2, :, :] * 110.0

    # Преобразование Lab -> RGB (упрощённое, для точности можно использовать kornia)
    # Здесь используем приближённое преобразование через XYZ (как в rgb_to_lab, но обратно)
    # Для простоты и стабильности предлагаю использовать kornia, если установлена.
    # Если нет, напишем вручную.
    try:
        import kornia
        lab = torch.cat([L_denorm, a_denorm, b_denorm], dim=1)
        rgb = kornia.color.lab_to_rgb(lab)
        return rgb  # диапазон [0,1]
    except ImportError:
        # Упрощённый вариант: конвертировать обратно через матрицы
        # Это сложно и может быть неточно, но для обучения сойдёт.
        # Рекомендую установить kornia: pip install kornia
        raise ImportError("Установите kornia: pip install kornia")