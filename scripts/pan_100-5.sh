python train_continual.py --num-gpus 8 --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
CONT.TASK 2 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR ./output/ps/100-5/step2 \
CONT.WEIGHTS output/ps/100-10_all_clsembed/step1/model_final.pth


for t in 3 4 5 6 7 8 9 10 11; do
  python train_continual.py --num-gpus 8 --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
  CONT.TASK ${t} SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR ./output/ps/100-5/step${t}
done
