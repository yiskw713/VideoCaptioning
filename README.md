# Video Captioning with pytorch

## Requirements
* python 3.x
* pytorch >= 1.0

You can download the required python packages by `pip install -r requirements.txt`  

## Dataset
**MSR-VTT**

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
      ├─.gitignore
      ├─ README.md
      ├─ requirements.txt
      ├─ test.py
      ├─ train.py
      └─ generate_cam.py

dataset_dir/ ─── feature_dir/
              ├─ hdf5_dir/ (video dir) 
              └─ anno_file (.json)
```



## How to use
### Setting vocabulary
First of all, please run `python utils/build_vocab.py $PATH_TO_ANNO_FILE` to generate vocablary.  
  
### Training
Then, run `python train.py ./result/xxx/config.yaml --resume` for training.  
You can train models on your own setting. Please make `config.yaml` like the below example:
```
# for decoder
embed_size: 256
hidden_size: 512
num_layers: 1

criterion: crossentropy

writer_flag: True      # if you use tensorboardx or not

batch_size: 64

# the number of input feature channels, size
in_channels: 2048
align_size: [10, 7, 7]

add_noise: True       # data augumentation
stddev: 0.01           # stddev of noise

num_workers: 1
max_epoch: 300

optimizer: Adam
scheduler: None

learning_rate: 0.001
lr_patience: 10       # Patience of LR scheduler
momentum: 0.9         # momentum of SGD
dampening: 0.0        # dampening for momentum of SGD
weight_decay: 0.0001  # weight decay
nesterov: True        # enables Nesterov momentum
final_lr: 0.1         # final learning rate for AdaBound
poly_power: 0.9       # for polunomial learning scheduler

dataset: MSR-VTT
dataset_dir: /media/cvrg/ssd2t2/msr-vtt/
feature_dir: features/r50_k700_16f
hdf5_dir: hdf5
ann_file: videodatainfo_2017.json
vocab_path: ./data/vocab.pkl

result_path: ./result/cfg1/
```
  
### Caption Generation
Run `python eval.py ./result/xxx/config.yaml test` to save predicted captions to csv file.

### CAM Visualization
Run `python generate_cam.py ./result/xxx/config.yaml test gradcam` to generate and save cams(.png).

### Convert .png to .mp4 (CAM files)
Run `python utils/convert_png2vid.py ./result/xxx/gradcam`.

## To do
- [ ] Add metric codes in eval.py.

## References
* [pytorch tutorial image captioning](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)
