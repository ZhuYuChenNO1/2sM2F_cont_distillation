python train_continual.py --dist-url auto --num-gpus 8 --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
CONT.TASK 3 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR ./output/ps/100-5_a40/step3 \
# CONT.WEIGHTS /public/home/zhuyuchen530/projects/cvpr24/2sM2F_cont/output/ps/100-10_all_clsembed/step1/model_final.pth
