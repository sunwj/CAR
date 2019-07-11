# CAR-pytorch
**The code is under refactoring and will be available soon ...**

Pytorch implementation of paper **"Learned Image Downscaling for Upscaling using Content Adaptive Resampler"**

![](figs/overview.png)

## Installation

	# get CAR-pytorch source
	git clone https://github.com/sunwj/CAR.git
	cd CAR
	
	# compile the code of the resampler
	cd adaptive_gridsampler
	python3 setup.py build_ext --inplace

### Python requirements
Currently, the code only supports python3 and machine with NVIDIA GPU installed

* numpy
* scipy
* pytorch (== 1.1.0)
* Pillow
* tqdm

### Pre-trained models
You can download the pre-trained models for 2x and 4x downscaling and super-resolution from [here](https://xxxx).

## Inference
	python3 run.py --scale 4 --img_dir path_to_images --model_dir path_to_pretrained_models \
	--output_dir path_to_output

## Sample results
![](figs/qualitative.png)

## Acknowlegements
EDSR code is provided by [thstkdgus35/EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch).
