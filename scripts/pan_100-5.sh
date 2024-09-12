python train_continual.py --dist-url auto --num-gpus 1 --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
CONT.TASK 2 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR ./output/ps/100-5_distill_freezelabel_tt/step2 \
CONT.WEIGHTS model_final.pth


# for t in 3 4 5 6 7 8 9 10 11; do
#   python train_continual.py --num-gpus 4 --resume --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
#   CONT.TASK ${t} SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR ./output/ps/100-5_distill_freezelabel/step${t}
# done
