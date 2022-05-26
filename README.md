# Deep Neural Networks Learn Meta-Structures from Noisy Labels in Semantic Segmentation 

## Introduction
This is the official code of 

[Deep Neural Networks Learn Meta-Structures from Noisy Labels in Semantic Segmentation](https://www.aaai.org/AAAI22Papers/AAAI-12729.LuoY.pdf) (AAAI-2022 Oral). 

[Elucidating Meta-Structures of Noisy Labels in Semantic Segmentation by Deep Neural Networks](https://arxiv.org/pdf/2205.00160v1.pdf) (under review)

---

### Data preparation
You need to download the [ER and MITO](https://ieee-dataport.org/documents/fluorescence-microscopy-images-cbmi) datasets.

Your directory tree should be look like this:
````
$SEG_ROOT/datasets
├── er
│   ├── test
│   │   ├── images
│   │   ├── labels
│   ├── train
│   │   ├── images
│   │   ├── labels
│   ├── val
│   │   ├── images
│   │   ├── labels
├── mito
│   ├── test
│   │   ├── images
│   │   ├── labels
│   ├── train
│   │   ├── images
│   │   ├── labels
│   ├── val
│   │   ├── images
│   │   ├── labels
├── txt
│   ├── er
│   │   ├── train
│   │   │   ├── train_gt.txt
│   │   │   ├── train_noisyLabel.txt
│   │   └── val.txt
│   ├── mito
│   │   ├── train
│   │   │   ├── train_gt.txt
│   │   │   ├── train_noisyLabel.txt
│   │   └── val.txt
````

### Training

To train model, you should save the datapath into a **__.txt** file and put it into the **txt** dictionary, then run **main.py** for training.


### Testing

To test the segmentation performance, you should first run **evaluation/inference.py** to save the outputs of testing sets in **train_log** (Use parameters `train_dir` and `test_ckpt_epoch` to change the path of pre-trained models).


Then, you can run **evaluation/inference.py** to get different metrics scores such as IOU, F1 and others on testing set. (Use parameter `test_data_dir` to change the testing datapath **__.txt**. Use parameter `prd_dir` to change the saved predictions path of testing sets).

---

## Reference
[1] Deep Neural Networks Learn Meta-Structures from Noisy Labels in Semantic Segmentation. 
Yaoru Luo, Guole Liu, Yuanhao Guo, Ge Yang
Accepted by AAAI-22. [download](https://www.aaai.org/AAAI22Papers/AAAI-12729.LuoY.pdf)

---

## Contributing 
Code for this projects developped at CBMI Group (Computational Biology and Machine Intelligence Group).

CBMI at National Laboratory of Pattern Recognition, INSTITUTE OF AUTOMATION, CHINESE ACADEMY OF SCIENCES.

