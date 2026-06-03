import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr(pred_rgb, target_rgb):
    # pred_rgb, target_rgb: numpy arrays (H,W,3) в диапазоне [0,1]
    return peak_signal_noise_ratio(target_rgb, pred_rgb, data_range=1.0)

def compute_ssim(pred_rgb, target_rgb):
    # multichannel SSIM
    return structural_similarity(target_rgb, pred_rgb, multichannel=True, data_range=1.0)

def evaluate_model(model, dataloader, device):
    model.eval()
    psnr_list, ssim_list = [], []
    with torch.no_grad():
        for gray, ab in dataloader:
            gray = gray.to(device)
            ab_pred = model(gray)
            # преобразуем Lab в RGB для метрик
            L = gray[:,0:1,:,:]  # но gray у нас 3 канала одинаковые, возьмём первый
            lab_pred = torch.cat([L, ab_pred], dim=1)
            lab_true = torch.cat([L, ab.to(device)], dim=1)
            rgb_pred = torch.clamp(transforms.functional.lab_to_rgb(lab_pred), 0, 1)
            rgb_true = torch.clamp(transforms.functional.lab_to_rgb(lab_true), 0, 1)
            # numpy
            for i in range(rgb_pred.shape[0]):
                p = rgb_pred[i].cpu().permute(1,2,0).numpy()
                t = rgb_true[i].cpu().permute(1,2,0).numpy()
                psnr_list.append(compute_psnr(p, t))
                ssim_list.append(compute_ssim(p, t))
    return np.mean(psnr_list), np.mean(ssim_list)