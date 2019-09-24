# Video Captioning with pytorch

## Requirements
* python 3.x
* pytorch >= 1.0

You can download the required python packages by `pip install -r requirements.txt`  

## Dataset
**MSR-VTTT**

You have to save features from 3DCNN which have spatial and temporal dimensions beforehand.  
First,  Please download the dataset from the link on [this page](https://github.com/VisionLearningGroup/caption-guided-saliency/issues/6)  
Then, you can extract features in MSR-VTT [using the codes from this repository](https://github.com/yiskw713/video_feature_extractor).


## Directory Structure

```
root/ ── libs/
      ├─ data/
      ├─ model/
      ├─ result/
      ├─ utils/
      ├.gitignore
      ├ README.md
      ├ requirements.txt
      ├ test.py
      ├ train.py
      └ hoge.py

dataset_dir/ ─── feature_dir/
              └─ anno_file (.json)
```



## How to use
First of all, please run `python utils/build_vocab.py $PATH_TO_ANNO_FILE` to generate vocablary.  
Then, run `python train.py ./result/xxx/config.yaml --resume` for training.  
  
Codes for evaluation and visualization(CAM) are coming soon.


## References
* [pytorch tutorial image captioning](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)