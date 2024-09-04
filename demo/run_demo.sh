#!/bin/bash

# if [ "$#" -ne 1 ]; then
#     echo "Usage: $0 <filename>"
#     exit 1
# fi

# 从命令行获取文件名
filename=$1
catname=$2
# 定义基础路径
base_path="../datasets/ADEChallengeData2016/images/validation/"

# 定义输出路径
# output_path="/inspurfs/group/yangsb/zhuyuchen/2stage_analyse/v2better/mask2former_2s_v2/${catname}/${filename%.*}"
output_path="./${filename%.*}"
# 执行命令
python demo.py --config-file /public/home/zhuyuchen530/projects/cvpr24/2sM2F_cont/configs/ade20k/panoptic-segmentation/maskformer2_R50_bs16_160k.yaml \
 --input "${base_path}${filename}" --output "${output_path}" \
 --opts MODEL.WEIGHTS "/public/home/zhuyuchen530/projects/cvpr24/2sM2F_cont/output/ps/100-5/step4/model_final.pth"

echo "命令执行完成"