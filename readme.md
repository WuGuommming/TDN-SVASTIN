# TDN version of SVASTIN
### SVASTIN: Sparse Video Adversarial Attack via Spatio-Temporal Invertible Neural Networks
https://github.com/Brittany-Chen/SVASTIN

### TDN: Temporal Difference Networks for Efficient Action Recognition (CVPR 2021)
https://github.com/MCG-NJU/TDN

## Warning

**Personal Use.** **Poor Implementation.** **Terrible Encapsulation.** **Shit Code.**

## Requisites

* PyTorch >= 2.1.1
* Python3 >= 3.10.13
* [pywavelets](https://github.com/KeKsBoTer/torch-dwt) >= 1.5.0
* [mmaction2](https://github.com/open-mmlab/mmaction2) >= 1.2.0
* NVIDIA GPU + CUDA CuDNN
* decord
* scikit-learn
* TensorboardX
* tqdm

## Preparation

### Dataset

The TDN dataset code and config (RGB)

### Models

Use the TDN model with resnet-101 backbone, RGB modality and 8 segments. Or you need to modify codes.

## Run

You can run ```TDN-main.py``` directly.

Prepare ```hmdb51_train_split_1_rawframes.txt``` just like the train/valid list file of hmdb51/ucf101 (such as ```hmdb51_target_split_1_rawframes.txt```). But make sure this target list includes all classes.


