Category-anchor Guided Unsupervised Domain Adaptation for Semantic Segmentation

Qiming Zhang*, Jing Zhang*, Wei Liu, Dacheng Tao

[paper](https://arxiv.org/abs/1910.13049)


## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)
- [Notes](#note)

## Introduction

This respository contains the CAG-UDA method as described in the NeurIPS 2019 paper ["Category-anchor Guided Unsupervised Domain Adaptation for Semantic Segmentation"](https://arxiv.org/abs/1910.13049).

## Requirements

The code is implemented based on [Pytorch 0.4.1](https://pytorch.org/) with CUDA 9.0, Python 3.6.7. The code is trained using a NVIDIA Tesla V100 with 16 GB memory. Please see the 'requirements.txt' file for other requirements.

## Usage

Assuming you are in the CAG-UDA master folder.

1. Preparation:
* Download the [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) dataset as the source domain, and the [Cityscapes](https://www.cityscapes-dataset.com/) dataset as the target domain.
* Then put them into a folder (dataset/GTA5 for example). Please carefully check the directory of the folder whether containing invalid characters.
* Please notice that images in GTA5 have slightly different resolutions, which has been resolved in our code.
* Download pretrained models [here](https://www.dropbox.com/sh/ysnwdj70l3t4hxu/AADPx3ecwAlv4syrKDAIS7vpa?dl=0) and put them in the 'pretrained/' folder. There are four models for warmup, stage 1, stage 2, and stage 3 respectively.

2. Setup the config file with directory 'config/adaptation_from_city_to_gta.yml'.
* Set the dataset path in the config file (data:source:rootpath and data:target:rootpath).
* Set the pretrained model path to 'training:resume' and 'training:Pred_resume' in the config file. 'Pred_resume' model is used to assign pseudo-labels..
* To better understand the meaning of each parameter in the config file, please see 'config/readme'.

3. Training
* To run the code:
~~~~
python train.py
~~~~
* During the training, the generated files (log file) will be written in the folder 'runs/..'.

4. Evaluation
* Set the config file for test (configs/test_from_city_to_gta.yml):
    (1). Set the dataset path as illustrated before.
    (2). Set the model path in 'test:path:'.
* Run:
~~~~
python test.py
~~~~
to see the results.

4. Constructing anchors
* Setting the config file 'configs/CAC_from_gta_to_city.yml' as illustrated before.
* Run:
~~~~
python cac.py
~~~~
* The anchor file would be in 'run/cac_from_gta_to_city/..'
<!-- 
to train the neural network from GTA5 to Cityscapes:
    config file: config/adaptation_from_city_to_gta.yml
    1. set the dataset path in the config file (data:source:rootpath, 'dataset/GTA5' for example)
    2. set the model path to 'training:resume' and 'training:Pred_resume' in the config file (pretrained/from_gta5_to_cityscapes_on_deeplab101_best_model_warmup.pkl for example as training from warmup)
    3. run 'train.py'

to evaluate the model on Cityscapes validation set:
    config file: config/test_from_city_to_gta.yml
    1. set the dataset path in the config file (data:source:rootpath)
    2. set the model path in the config file (test:path)
    2. run 'test.py' -->




## License

[MIT](LICENSE)

The code is heavily borrowed from the repository (https://github.com/meetshah1995/pytorch-semseg).

If you use this code and find it usefule, please cite:
~~~~
@inproceedings{zhang2019category,
  title={Category Anchor-Guided Unsupervised Domain Adaptation for Semantic Segmentation},
  author={Zhang, Qiming and Zhang, Jing and Liu, Wei and Tao, Dacheng},
  booktitle={Advances in Neural Information Processing Systems},
  pages={433--443},
  year={2019}
}
~~~~

## Notes
The category anchors are stored in the file 'category_anchors'. It is calculated as the mean value of features with respect to each category from the source domain.

Contact: qzha2506@uni.sydney.edu.au / qmzhangzz@gmail.com