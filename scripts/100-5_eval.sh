for t in 4 5; do
  python train_continual.py --num-gpus 4 --eval-only --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
  CONT.TASK ${t} SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR ./output/ps/100-5/step${t}
done
