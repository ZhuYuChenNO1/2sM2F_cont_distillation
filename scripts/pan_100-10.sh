#!/bin/bash

# 暂停 5 小时
# sleep $((1 * 3600))

python train_continual.py --resume --num-gpus 4 --dist-url auto --config-file configs/ade20k/panoptic-segmentation/100-10.yaml \
  CONT.TASK 2 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 1000000 MODEL.MASK_FORMER.USE_TEXT_EMBEDDING False OUTPUT_DIR ./output/ps/100-10_psd0.8/step2 \
  CONT.WEIGHTS /public/home/zhuyuchen530/projects/cvpr24/2sM2F_cont_distillation/output/ps/100-5_passSelfCross/step2/model_final.pth
# for t in 2 3 4 5 6; do
#   python train_continual.py --num-gpus 4 --dist-url auto --config-file configs/ade20k/panoptic-segmentation/100-10_inc.yaml \
#   CONT.TASK ${t} SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 10000 MODEL.MASK_FORMER.USE_TEXT_EMBEDDING False OUTPUT_DIR ./output/ps/100-10_distill_freezelabel/step${t}
# done
# for t in 2; do
#   python train_continual.py --num-gpus 8 --dist-url auto --config-file configs/ade20k/panoptic-segmentation/100-10_inc.yaml \
#   CONT.TASK ${t} SOLVER.BASE_LR 0.00002 SOLVER.MAX_ITER 10000 MODEL.MASK_FORMER.USE_TEXT_EMBEDDING False OUTPUT_DIR ./output/ps/100-10_all_clsembed/step${t}
# done

# for t in 2 ; do
#   python train_continual.py --config-file configs/ade20k/panoptic-segmentation/100-10.yaml \
#   CONT.TASK ${t} SsOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 10000 OUTPUT_DIR ./output/ps/100-10/step${t}
# done
