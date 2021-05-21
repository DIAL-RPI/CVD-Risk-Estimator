# Tri2D-Net for CVD Risk Estimation

[![DOI](https://zenodo.org/badge/256093026.svg)](https://zenodo.org/badge/latestdoi/256093026)

Tri2D-Net is the **first** deep learning network trained for directly estimating **overall** cardiovascular disease (CVD) risks on low dose computed tomography (LDCT). The corresponding [paper](https://www.nature.com/articles/s41467-021-23235-4) has been published on Nature Communications.

## Prerequisites

- Python 3.7
- PyTorch 1.4
- Computing device with GPU


## Getting started
### Installation

- (Optional) Install [Anaconda3](https://www.anaconda.com/download/) for managing Python and packages
- Install [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)
- Install [PyTorch](http://pytorch.org/)

Noted that our code is tested based on [PyTorch 1.4](https://pytorch.org/get-started/previous-versions/)

### Data 
#### Availability
This model was trained on the [National Lung Screening Trial (NLST)](https://biometry.nci.nih.gov/cdas/learn/nlst/images/) dataset. The NLST is made publicly available by the National Cancer Institute.

#### Preprocess
- **Heart Detection**: [RetinaNet](https://github.com/yhenon/pytorch-retinanet) was used in our study for heart detection.
- **Resize & Normalization**: The detected heart region was resized into 128x128x128. The image was normalized with a range of -300HU~500HU.

### Get Trained Model

**BEFORE RUNNING THE CODE, PLEASE DOWNLOAD THE NETWORK CHECKPOINT FIRST.**

The trained model can be downloaded through [this link](https://1drv.ms/u/s!AurT2TsSKdxQvz1aHvmxTlkDNkTz?e=8rCnJl). Please download the checkpoint to the `./checkpoint` folder.


### CVD Risk Prediction

To predict CVD Risk from an image, run:
```bash
python pred.py
```
- `--path` path of the input image. #Default: `./demos/Positive_CAC_1.npy`
- `--iter` iteration of the checkpoint to load. #Default: 8000

#### Input

The model takes a normalized 128x128x128 `numpy.ndarray` as an input, i.e., each item in the `ndarray` ranges 0~1.

#### Output

A real number in \[0, 1\] indicates the estimated CVD risk.

#### Demo

We uploaded 4 demos in the `./demo` folder, including one CVD negative case and three CVD positive case. One of the CVD positive subjects died because of CVD in the trial. 

The name of the file indicates its label and the CAC grade evaluated by our radiologists.


## Citation
Please cite these papers in your publications if the code helps your research:
```
@Article{chao2021deep,
  author  = {Chao, Hanqing and Shan, Hongming and Homayounieh, Fatemeh and Singh, Ramandeep and Khera, Ruhani Doda and Guo, Hengtao and Su, Timothy and Wang, Ge and Kalra, Mannudeep K. and Yan, Pingkun},
  title   = {Deep learning predicts cardiovascular disease risks from lung cancer screening low dose computed tomography},
  journal = {Nature Communications},
  year    = {2021},
  volume  = {12},
  number  = {1},
  pages   = {2963},
  url     = {https://doi.org/10.1038/s41467-021-23235-4},
}
```
Link to paper:
- [Deep Learning Predicts Cardiovascular Disease Risks from Lung Cancer Screening Low Dose Computed Tomography](https://www.nature.com/articles/s41467-021-23235-4)


## License
The source code of Tri2D-Net is licensed under a MIT-style license, as found in the [LICENSE](LICENSE) file.
This code is only freely available for non-commercial use, and may be redistributed under these conditions.
For commercial queries, please contact [Dr. Pingkun Yan](https://dial.rpi.edu/people/pingkun-yan).
