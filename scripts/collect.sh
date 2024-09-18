python train_continual.py --dist-url auto --num-gpus 4 --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
CONT.TASK 1 SOLVER.BASE_LR 0.0 SOLVER.MAX_ITER 2500 CONT.COLLECT_QUERY_MODE False OUTPUT_DIR ./output/ps/test_ddp/step1 \
MODEL.WEIGHTS /public/home/zhuyuchen530/projects/cvpr24/2sM2F_cont_distillation/output/ps/100-5_passSelfCross_fakequery_psdWeight/step1/model_final.pth