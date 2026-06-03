import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import kornia
from model import HistColorNet


# ---------- ДАТАСЕТ (такой же, как в train_fast.py, но можно уменьшить размер) ----------
class QuickDataset(Dataset):
    def __init__(self, folder, size=128):  # size=128 для ускорения, потом можно увеличить
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img_tensor = self.transform(img)
        lab = kornia.color.rgb_to_lab(img_tensor.unsqueeze(0)).squeeze(0)
        L = lab[0:1, :, :]
        L_norm = (L / 50.0) - 1.0  # [-1, 1]
        ab = lab[1:3, :, :] / 110.0  # нормализуем в [-1,1]
        gray = L_norm.repeat(3, 1, 1)  # вход: 3 канала L
        return gray, ab


# ---------- ФУНКЦИИ ПОТЕРЬ ----------
class HistogramLoss(torch.nn.Module):
    def __init__(self, bins=32):
        super().__init__()
        self.bins = bins

    def forward(self, pred_ab, target_ab):
        def get_hist(x):
            B, C, H, W = x.shape
            x_flat = x.view(B, C, -1)
            hist = torch.zeros(B, C, self.bins, device=x.device)
            for b in range(B):
                for c in range(C):
                    values = x_flat[b, c]
                    minv, maxv = values.min(), values.max()
                    if maxv - minv > 1e-5:
                        indices = ((values - minv) / (maxv - minv) * (self.bins - 1)).long()
                        hist[b, c] = torch.bincount(indices, minlength=self.bins).float()
                    else:
                        hist[b, c, 0] = len(values)
            return hist / (H * W)

        hist_pred = get_hist(pred_ab)
        hist_target = get_hist(target_ab)
        return torch.mean((hist_pred - hist_target) ** 2)


class PerceptualLossLPIPS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import lpips
        self.lpips = lpips.LPIPS(net='vgg').eval()
        for p in self.lpips.parameters():
            p.requires_grad = False

    def forward(self, pred_ab, target_ab, L):
        # L: (B,1,H,W) в [-1,1]; ab: (B,2,H,W) в [-1,1]
        L_denorm = (L + 1.0) * 50.0
        pred_ab_denorm = pred_ab * 110.0
        target_ab_denorm = target_ab * 110.0
        lab_pred = torch.cat([L_denorm, pred_ab_denorm], dim=1)
        lab_target = torch.cat([L_denorm, target_ab_denorm], dim=1)
        rgb_pred = kornia.color.lab_to_rgb(lab_pred)
        rgb_target = kornia.color.lab_to_rgb(lab_target)
        # LPIPS ожидает [-1,1], переведём
        rgb_pred = rgb_pred * 2 - 1
        rgb_target = rgb_target * 2 - 1
        return self.lpips(rgb_pred, rgb_target).mean()


def combined_loss(pred_ab, target_ab, L, perceptual_fn, hist_fn):
    l1 = torch.nn.functional.l1_loss(pred_ab, target_ab)
    perc = perceptual_fn(pred_ab, target_ab, L) if perceptual_fn is not None else 0
    hist = hist_fn(pred_ab, target_ab) if hist_fn is not None else 0
    return l1 + 0.1 * perc + 0.05 * hist


# ---------- ОСНОВНАЯ ЧАСТЬ ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Параметры
train_dir = 'DIV2K_valid_HR'  # путь к вашим данным
batch_size = 4  # можно увеличить если есть память
num_epochs = 5  # дообучаем 5 эпох (хватит для улучшения цвета)
learning_rate = 1e-5  # маленькая скорость для тонкой настройки

# Датасет и загрузчик
dataset = QuickDataset(train_dir, size=128)  # размер 128 для скорости
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Модель
model = HistColorNet().to(device)
# Загружаем ранее обученные веса
model.load_state_dict(torch.load('best_model.pth', map_location=device))
print("Загружены веса из best_model.pth")

# Оптимизатор
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Функции потерь
perceptual_fn = PerceptualLossLPIPS().to(device)
hist_fn = HistogramLoss().to(device)

# Обучение
model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for i, (gray, ab) in enumerate(loader):
        gray, ab = gray.to(device), ab.to(device)
        L = gray[:, 0:1, :, :]  # берём один канал яркости (все три одинаковые)

        optimizer.zero_grad()
        pred_ab = model(gray)
        loss = combined_loss(pred_ab, ab, L, perceptual_fn, hist_fn)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 20 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, step {i}, loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1} finished. Average loss: {avg_loss:.4f}")

    # Сохраняем модель после каждой эпохи
    torch.save(model.state_dict(), 'best_model_lpips.pth')
    print("Модель сохранена как best_model_lpips.pth")

print("Дообучение завершено.")