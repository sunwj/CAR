import os, argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image

import torch
import torch.nn as nn

import utils
from EDSR.edsr import EDSR
from modules import DSN
from adaptive_gridsampler.gridsampler import Downsampler
from skimage.color import rgb2ycbcr


parser = argparse.ArgumentParser(description='Content Adaptive Resampler for Image downscaling')
parser.add_argument('--model_dir', type=str, default='./models', help='path to the pre-trained model')
parser.add_argument('--img_dir', type=str, help='path to the HR images to be downscaled')
parser.add_argument('--scale', type=int, help='downscale factor')
parser.add_argument('--output_dir', type=str, help='path to store results')
parser.add_argument('--benchmark', type=bool, default=True, help='report benchmark results')
args = parser.parse_args()


SCALE = args.scale
KSIZE = 3 * SCALE + 1
OFFSET_UNIT = SCALE
BENCHMARK = args.benchmark

kernel_generation_net = DSN(k_size=KSIZE, scale=SCALE).cuda()
downsampler_net = Downsampler(SCALE, KSIZE).cuda()
upscale_net = EDSR(32, 256, scale=SCALE).cuda()

kernel_generation_net = nn.DataParallel(kernel_generation_net, [0])
downsampler_net = nn.DataParallel(downsampler_net, [0])
upscale_net = nn.DataParallel(upscale_net, [0])

kernel_generation_net.load_state_dict(torch.load(os.path.join(args.model_dir, '{0}x'.format(SCALE), 'kgn.pth')))
upscale_net.load_state_dict(torch.load(os.path.join(args.model_dir, '{0}x'.format(SCALE), 'usn.pth')))
torch.set_grad_enabled(False)


def validation(img, name, save_imgs=False, save_dir=None):
    kernel_generation_net.eval()
    downsampler_net.eval()
    upscale_net.eval()

    kernels, offsets_h, offsets_v = kernel_generation_net(img)
    downscaled_img = downsampler_net(img, kernels, offsets_h, offsets_v, OFFSET_UNIT)
    downscaled_img = torch.clamp(downscaled_img, 0, 1)
    downscaled_img = torch.round(downscaled_img * 255)

    reconstructed_img = upscale_net(downscaled_img / 255.0)

    img = img * 255
    img = img.data.cpu().numpy().transpose(0, 2, 3, 1)
    img = np.uint8(img)

    reconstructed_img = torch.clamp(reconstructed_img, 0, 1) * 255
    reconstructed_img = reconstructed_img.data.cpu().numpy().transpose(0, 2, 3, 1)
    reconstructed_img = np.uint8(reconstructed_img)

    downscaled_img = downscaled_img.data.cpu().numpy().transpose(0, 2, 3, 1)
    downscaled_img = np.uint8(downscaled_img)

    orig_img = img[0, ...].squeeze()
    downscaled_img = downscaled_img[0, ...].squeeze()
    recon_img = reconstructed_img[0, ...].squeeze()

    if save_imgs and save_dir:
        img = Image.fromarray(orig_img)
        img.save(os.path.join(save_dir, name + '_orig.png'))

        img = Image.fromarray(downscaled_img)
        img.save(os.path.join(save_dir, name + '_down.png'))

        img = Image.fromarray(recon_img)
        img.save(os.path.join(save_dir, name + '_recon.png'))

    psnr = utils.cal_psnr(orig_img[SCALE:-SCALE, SCALE:-SCALE, ...], recon_img[SCALE:-SCALE, SCALE:-SCALE, ...], benchmark=BENCHMARK)

    orig_img_y = rgb2ycbcr(orig_img)[:, :, 0]
    recon_img_y = rgb2ycbcr(recon_img)[:, :, 0]
    orig_img_y = orig_img_y[SCALE:-SCALE, SCALE:-SCALE, ...]
    recon_img_y = recon_img_y[SCALE:-SCALE, SCALE:-SCALE, ...]

    ssim = utils.calc_ssim(recon_img_y, orig_img_y)

    return psnr, ssim


if __name__ == '__main__':
    img_list = glob(os.path.join(args.img_dir, '**', '*.png'), recursive=True)
    assert len(img_list) > 0

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    psnr_list = list()
    ssim_list = list()
    for img_file in tqdm(img_list):
        name = os.path.basename(img_file)
        name = os.path.splitext(name)[0]

        img = utils.load_img(img_file)

        psnr, ssim = validation(img, name, save_imgs=True, save_dir=args.output_dir)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print('Mean PSNR: {0:.2f}'.format(np.mean(psnr_list)))
    print('Mean SSIM: {0:.4f}'.format(np.mean(ssim_list)))
