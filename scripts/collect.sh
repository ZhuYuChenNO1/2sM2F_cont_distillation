python train_continual.py --dist-url auto --num-gpus 4 --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
CONT.TASK 11 SOLVER.BASE_LR 0.0 SOLVER.MAX_ITER 2500 CONT.COLLECT_QUERY_MODE True OUTPUT_DIR ./output/ps/fake3/step11 \
CONT.WEIGHTS output/ps/fake3/step11/model_final.pth