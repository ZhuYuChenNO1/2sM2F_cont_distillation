<<<<<<< HEAD
<<<<<<< HEAD
# python train_continual.py --dist-url auto --num-gpus 4 --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
# CONT.TASK 2 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR ./output/ps/100-5_distill/step2 \
# CONT.WEIGHTS /public/home/zhuyuchen530/projects/cvpr24/2sM2F_cont/output/ps/100-10_all_clsembed/step1/model_final.pth
=======
python train_continual.py --dist-url auto --num-gpus 4 --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
CONT.TASK 2 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR ./output/ps/100-5_distill_freezelabel/step2 \
=======
python train_continual.py --dist-url auto --num-gpus 1 --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
CONT.TASK 2 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR ./output/ps/100-5_distill_freezelabel_tt/step2 \
>>>>>>> 2290bfd9ed6dcbb2d1cbbb5b399c49781c692c17
CONT.WEIGHTS model_final.pth
>>>>>>> 4284b9902c277fcf9f4228fa851f2535c250e7ca


<<<<<<< HEAD
for t in 3 4 5 6 7 8 9 10 11; do
  python train_continual.py --num-gpus 4 --resume --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
<<<<<<< HEAD
  CONT.TASK ${t} SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR ./output/ps/100-5_distill/step${t}
=======
  CONT.TASK ${t} SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR ./output/ps/100-5_distill_freezelabel/step${t}
>>>>>>> 4284b9902c277fcf9f4228fa851f2535c250e7ca
done
=======
# for t in 3 4 5 6 7 8 9 10 11; do
#   python train_continual.py --num-gpus 4 --resume --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
#   CONT.TASK ${t} SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR ./output/ps/100-5_distill_freezelabel/step${t}
# done
>>>>>>> 2290bfd9ed6dcbb2d1cbbb5b399c49781c692c17
