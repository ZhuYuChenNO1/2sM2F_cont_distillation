# python train_continual.py --num-gpus 4 --config-file configs/ade20k/panoptic-segmentation/100-50.yaml \
# CONT.TASK 1 SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 160000 OUTPUT_DIR ./output/ps/100-50/step1

python train_continual.py --num-gpus 4 --config-file configs/ade20k/panoptic-segmentation/100-50.yaml \
CONT.TASK 2 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 50000 OUTPUT_DIR ./output/ps/100-50_distill_freezelabel/step2 \
CONT.WEIGHTS /root/projets/2sM2F_cont_baseline_medtoken/output/ps/100-10_distill_freezelabel/step1/model_final.pth
