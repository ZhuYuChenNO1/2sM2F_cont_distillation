for t in 2; do
  python train_continual.py --num-gpus 4 --eval-only --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
  CONT.TASK ${t} MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD 0.25 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR output/ps/fake3_psd3/step${t}
done
