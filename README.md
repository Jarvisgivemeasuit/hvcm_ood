# HVCM
## Requirements
- python 3.8.13
- cuda 11.1
- numpy 1.22.4
- Pillow 9.4.0
- progress 1.6
- scikit_learn 1.2.2
- scipy 1.7.3
- torch 1.13.1
- torchvision 0.14.1

## Dataset Preparation for Large-scale Experiment
### In-distribution dataset
Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in ```./hvcm_imagenet/data/train``` and ```./hvcm_imagenet/data/val```, respectively.

### Out-of-distribution datasets
we follow [MOS](https://github.com/deeplearning-wisc/large_scale_ood) and use Texture, iNaturalist, Places365 and SUN,  and de-duplicated concepts overlapped with ImageNet-1k. To further explore the limitation of our approach, we follow [VIM](https://github.com/haoqiwang/vim) and use ImageNet-O and OpenImage-O. 

For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset, which can be download via the following links:
```
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```
For Textures, we use the entire dataset, which can be downloaded from their [original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

ImageNet-O and OpenImage-O can be Download from [VIM](https://github.com/haoqiwang/vim).

Please put all downloaded OOD datasets into ```./hvcm_imagenet/data/ood_data/```. 



## Dataset Preparation for CIFAR Experiment
### In-distribution dataset
The downloading process will start immediately upon running.

### Out-of-distribution datasets
we follow [Energy](https://github.com/wetliu/energy_ood) and [KNN](https://github.com/deeplearning-wisc/knn-ood) and use Texture, SVHN, Place365, iSUN, LSUN-Crop, LSUN-Resize. 

We provide links and instructions to download each dataset:
- [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of ```./hvcm_cigar10/data/svhn```. Then run ```python select_svhn_data.py``` to generate test subset.
- [Texture](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of ```./hvcm_cigar10/data/dtd```
- [Places365](http://data.csail.mit.edu/places/places365/test_256.tar)download it and place it in the folder of ```./hvcm_cifar10/data/places/test_256```. We randomly sample 10,000 images from the original test dataset by running ```python sample_places.py```.
- [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): 
download it and place it in the folder of ```./hvcm_cigar10/data/LSUN/```.
- [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)download it and place it in the folder of ```./hvcm_cigar10/data/LSUN-R/```.
- [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): 
download it and place it in the folder of ```./hvcm_cigar10/data/iSUN/```.

## Demo
1. Demo code for ImageNet Experiment
```
cd hvcm_imagenet
sh main.sh
```
Run ```sh ind_acc.sh``` for in-distribution accuracy.

Run ```sh get_gau.sh``` for GMM components.

Run```sh ood_maha.sh``` for OOD detection.

2. Demo code for CIFAR Experiment
```
cd hvcm_cifar10
sh train.sh
```

Run ```sh ind_acc.sh``` for in-distribution accuracy.

Run ```sh get_gau.sh``` for GMM components.

Run```sh ood_maha.sh``` for OOD detection.

