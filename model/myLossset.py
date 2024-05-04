import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import lpips

#################

def Gaussian_window(size, sigma):
    """
    Create my own gaussian window used for convolution
    """

    coords = torch.arange(size, dtype=torch.float32)
    coords = coords - size //2

    gaussian = torch.exp(-(coords**2)/(2*sigma**2))
    gaussian /= gaussian.sum()


    return gaussian.view(1,1,1,-1).repeat(1,1,size,1)


def SSIM(img1, img2, window_size=11, window_sigma=1.5, size_average=True, C1=0.01**2, C2=0.03**2):
    """
    Compute Structural Similarity Index between two images.
    """
    
    window = Gaussian_window(window_size, window_sigma).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=(window_size//2, window_size//2), groups=1)
    mu2 = F.conv2d(img2, window, padding=(window_size//2, window_size//2), groups=1)
    mu1_mu2 = mu1*mu2

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    
    sigma1_sq = F.conv2d(img1*img1, window, padding=(window_size//2, window_size//2), groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=(window_size//2, window_size//2), groups=1) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=(window_size//2, window_size//2), groups=1) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2))/ ( (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()

    else:
        return ssim_map.mean(1).mean(1).mean(1).item()
def rgb_to_grayscale(img):
    """
    Convert a RGB image to grayscale.
    This function applies the formula: 0.2989 * R + 0.5870 * G + 0.1140 * B
    Args:
        img (Tensor): The image tensor in [B, C, H, W] format.
    Returns:
        Tensor: Grayscale image tensor in [B, 1, H, W] format.
    """
    if img.size(1) == 3:  # Check if the image has 3 channels
        grayscale = 0.2989 * img[:, 0, :, :] + 0.5870 * img[:, 1, :, :] + 0.1140 * img[:, 2, :, :]
        grayscale = grayscale.unsqueeze(1)  # Add a channel dimension
        return grayscale
    else:
        return img

def SSIMD(img1, img2, window_size=11, window_sigma=1.5, size_average=True, C1=0.01**2, C2=0.03**2):
    """
    SSIM Dis similarity, SSIMD = (1 - SSIM ) / 2
    Args:
        img1, img2 (Tensor): The input images to compare.
        window_size (int): The size of the gaussian window.
        window_sigma (float): The standard deviation of the gaussian window.
        size_average (bool): If True, returns the mean SSIM over the batch. Otherwise, returns the SSIM per item in the batch.
        C1, C2 (float): Constants to stabilize the division.
    Returns:
        float: The SSIM dissimilarity value.
    """
    # Convert the images to grayscale if they have 3 channels
    img1 = rgb_to_grayscale(img1)
    img2 = rgb_to_grayscale(img2)

    # SSIM calculation (Assuming SSIM() is a predefined function or imported from a library)
    _ssim = SSIM(img1, img2, window_size, window_sigma, size_average, C1, C2)

    _ssimd = (1 - _ssim) / 2

    return _ssimd

def fLPIPS(img1, img2, net='alex'):
    """
    Compute LPIPS( learned perceptual image patch similarity) with library lpips4
    Don't use me 
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lpips_model = lpips.LPIPS(net).to(device)
    distance = lpips_model(img1, img2)

    return distance.item()




if __name__ == '__main__':
    torch.manual_seed(1234)
    img1 = torch.randn(1,1,256,256)
    img2 = torch.randn(1,1,256,256)
    ssim = SSIM(img1, img1)
    ssimd = SSIMD(img1, img1)
    distance = fLPIPS(img1,img2)
    print(f'SSIM:{ssim}, typeof:{type(ssim)}')
    print(f'SSIMD:{ssimd}, typeof:{type(ssimd)}')
    print(f"lpips:{distance}, typeof:{type(distance)}")
