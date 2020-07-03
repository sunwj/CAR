# CAR-pytorch

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
Currently, the code only supports python3 and machine with NVIDIA GPU (and the CUDA development toolkit) installed

* numpy
* scipy
* pytorch (== 1.3.1)
* Pillow
* tqdm

### Pre-trained models
You can download the pre-trained models for 2x and 4x downscaling and super-resolution from [here](https://mega.nz/#!XzIm3YhT!jbIOOOGBOiKtv3VAOD782Mz7nK1L_kma-BzR-RhboW4).

## Inference
	python3 run.py --scale 4 --img_dir path_to_images --model_dir path_to_pretrained_models \
	--output_dir path_to_output

## Sample results
![](figs/qualitative.png)

You can download HR images of benchmark datasets, i.e., the Set5, Set14, B100 and Urban100 from [here](https://mega.nz/#!znBRCSJA!_qwJMP5VDe3yleiK8m0QXrpHLee9AS8vzT03lAOorP0).

If you find our work useful in your research or publication, please cite our work:

Wanjie Sun, Zhenzhong Chen. **"Learned Image Downscaling for Upscaling using Content Adaptive Resampler"**. arXiv preprint arXiv:1907.12904, 2019.

```
@article{sun2020learned,
  title={Learned image downscaling for upscaling using content adaptive resampler},
  author={Sun, Wanjie and Chen, Zhenzhong},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={4027--4040},
  year={2020},
  publisher={IEEE}
}
```

## Acknowlegements
EDSR code is provided by [thstkdgus35/EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch).
