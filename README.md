# GaitNet
--------------------------------------------------------------------------------
* GaitNet: An E nd-to-end Network for Gait based Human Identification 
* Code Version 1.0                                                             
* By Chunfeng Song                                                             
* E-mail: chunfeng.song@nlpr.ia.ac.cn                                          
---------------------------------------------------------------------------------

i.    Overview
ii.   Copying
iii.  Use

i. OVERVIEW
-----------------------------
This code implements the paper:

>Chunfeng Song, Yongzhen Huang, Yan Huang, Ning Jia, Liang Wang, GaitNet: An end-to-end network for gait based human identification, Pattern Recognition, 2019

If you find this work is helpful for your research, please cite our paper [[PDF]](https://www.sciencedirect.com/science/article/pii/S0031320319302912).

ii. COPYING
-----------------------------
We share this code only for research use. We neither warrant 
correctness nor take any responsibility for the consequences of 
using this code. If you find any problem or inappropriate content
in this code, feel free to contact us (chunfeng.song@nlpr.ia.ac.cn).

iii. USE
-----------------------------
This code should work on Caffe with Python layer (pycaffe). You can install Caffe from: https://github.com/BVLC/caffe

(1) Data Preparation.

Download the gait datasets and their masks: CASIA-b([apply link](http://www.cbsr.ia.ac.cn/china/Gait%20Databases%20CH.asp)), Outdoor-Gait ([Baidu Yun](https://pan.baidu.com/s/1oW6u9olOZtQTYOW_8wgLow) with extract code (tjw0) OR [Google Drive](https://drive.google.com/drive/folders/1XRWq40G3Zk03YaELywxuVKNodul4TziG?usp=sharing)), and SZU RGB-D Gait ([apply link](https://faculty.sustech.edu.cn/yusq/))

Note: All images should be pre-cropped guided by the corresponding segmentations.

(2) Model Training.

Here, we take CASIA-b as an example. The other two datasets are the same.
>cd ./experiments/casiab

First eidt the 'CAFFE_ROOT' in 'train_net.sh', and 'im_path', 'gt_path' and 'dataset' in the prototxt files. 

Then, we can train the GaitNet model with the commands in 'train_net.sh'. For each step, it will take roughly 24 hours for single Titan X.
>sh train_net.sh

(3) Evaluation.

Run the code in './eval/eval-casiab/outdoor/szu.py'. If you did not train this model, just want run the inference, you could download the pre-trained model from [Baidu Yun](https://pan.baidu.com/s/111N5wcsZ09jjA9rpMrM1Qw) with extract code (ne65).
