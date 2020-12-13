Dataset Preparation.
---

Download the gait datasets and their masks: CASIA-b([apply link](http://www.cbsr.ia.ac.cn/china/Gait%20Databases%20CH.asp)), Outdoor-Gait ([Baidu Yun](https://pan.baidu.com/s/1oW6u9olOZtQTYOW_8wgLow) with extract code (tjw0) OR [Google Drive](https://drive.google.com/drive/folders/1XRWq40G3Zk03YaELywxuVKNodul4TziG?usp=sharing)), and SZU RGB-D Gait ([apply link](https://faculty.sustech.edu.cn/yusq/))

>*Note: All images should be pre-cropped guided by the corresponding segmentations.

Make sure that all the datasets are saveing in the following structure:

CASIA-B:
>./data  
>./data/CASIA-B  
>./data/CASIA-B/crop-im   
>./data/CASIA-B/crop-seg   

Outdoor-Gait:
>./data  
>./data/Outdoor-Gait  
>./data/Outdoor-Gait/im   
>./data/Outdoor-Gait/seg   

SZU-Gait:
>./data  
>./data/SZU-Gait  
>./data/SZU-Gait/crop-im   
>./data/SZU-Gait/crop-seg   
