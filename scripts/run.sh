#!/bin/bash

# 暂停 5 小时
# sleep $((5 * 3600))

python train_net.py --num-gpus 4  --config-file configs/ade20k/panoptic-segmentation/maskformer2_R50_bs16_160k.yaml 
