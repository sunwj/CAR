import numpy as np
import torch
from scipy import signal
from PIL import Image


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
    Acknowledgement : https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python (Author@ali_m)
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def calc_ssim(X, Y, sigma=1.5, K1=0.01, K2=0.03, R=255):
    '''
    X : y channel (i.e., luminance) of transformed YCbCr space of X
    Y : y channel (i.e., luminance) of transformed YCbCr space of Y
    Please follow the setting of psnr_ssim.m in EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution CVPRW2017).
    Official Link : https://github.com/LimBee/NTIRE2017/tree/db34606c2844e89317aac8728a2de562ef1f8aba
    The authors of EDSR use MATLAB's ssim as the evaluation tool,
    thus this function is the same as ssim.m in MATLAB with C(3) == C(2)/2.
    '''
    gaussian_filter = matlab_style_gauss2D((11, 11), sigma)

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    window = gaussian_filter

    ux = signal.convolve2d(X, window, mode='same', boundary='symm')
    uy = signal.convolve2d(Y, window, mode='same', boundary='symm')

    uxx = signal.convolve2d(X * X, window, mode='same', boundary='symm')
    uyy = signal.convolve2d(Y * Y, window, mode='same', boundary='symm')
    uxy = signal.convolve2d(X * Y, window, mode='same', boundary='symm')

    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy

    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D
    mssim = S.mean()

    return mssim


def cal_psnr(img_1, img_2, benchmark=False):
    assert img_1.shape[0] == img_2.shape[0] and img_1.shape[1] == img_2.shape[1]
    img_1 = np.float64(img_1)
    img_2 = np.float64(img_2)

    diff = (img_1 - img_2) / 255.0
    if benchmark:
        gray_coeff = np.array([65.738, 129.057, 25.064]).reshape(1, 1, 3) / 255.0
        diff = diff * gray_coeff
        diff = diff[:, :, 0] + diff[:, :, 1] + diff[:, :, 2]

    mse = np.mean(diff ** 2)
    psnr = -10.0 * np.log10(mse)

    return psnr


def load_img(img_file):
    img = Image.open(img_file).convert('RGB')
    img = np.array(img)
    h, w, _ = img.shape
    img = img[:h // 8 * 8, :w // 8 * 8, :]
    img = np.array(img) / 255.
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0).cuda()

    return img
