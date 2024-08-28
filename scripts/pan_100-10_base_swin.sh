python train_continual.py --num-gpus 8 --config-file configs/ade20k/panoptic-segmentation/100-10_swin.yaml\
    CONT.TASK 1 SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 160000 MODEL.MASK_FORMER.USE_TEXT_EMBEDDING True OUTPUT_DIR ./output/ps/100-10_testclip_pomptegineering_all+swin/step1
# for t in 2; do
#   python train_continual.py --num-gpus 4 --config-file configs/ade20k/panoptic-segmentation/100-10_inc.yaml \
#   CONT.TASK ${t} SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 10000 MODEL.MASK_FORMER.USE_TEXT_EMBEDDING False OUTPUT_DIR ./output/ps/100-10_ceWeight/step${t}
# done
# for t in 2 ; do
#   python train_continual.py --config-file configs/ade20k/panoptic-segmentation/100-10.yaml \
#   CONT.TASK ${t} SsOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 10000 OUTPUT_DIR ./output/ps/100-10/step${t}
# done