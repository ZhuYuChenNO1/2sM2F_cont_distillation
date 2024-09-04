for t in 2 3 4 5; do
  python train_continual.py --dist-url auto --eval-only --num-gpus 4 --config-file configs/ade20k/panoptic-segmentation/100-10_inc.yaml \
  CONT.TASK ${t} SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 10000 MODEL.MASK_FORMER.USE_TEXT_EMBEDDING False OUTPUT_DIR ./output/ps/100-10_all_clsembed_psdthresh_lr0.5_10k/step${t}
done