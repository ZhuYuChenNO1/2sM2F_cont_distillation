# python train_continual.py --num-gpus 4 --config-file configs/ade20k/panoptic-segmentation/100-50.yaml \
# CONT.TASK 1 SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 160000 OUTPUT_DIR ./output/ps/100-50/step1

python train_continual.py --num-gpus 4 --eval-only --config-file configs/ade20k/panoptic-segmentation/100-50.yaml \
CONT.TASK 1 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 50000 OUTPUT_DIR ./output/ps/100-50/step1 \

