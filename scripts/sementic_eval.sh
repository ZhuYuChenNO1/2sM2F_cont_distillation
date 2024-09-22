for t in 1; do
  python train_continual.py --num-gpus 4 --eval-only --config-file configs/ade20k/semantic-segmentation/100-5.yaml \
  CONT.TASK ${t} MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD 0.25 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR output/ss/100-5/step${t}
done
