for t in 1 2 3 4 ; do
  python train_continual.py --dist-url auto --num-gpus 4 --eval-only --config-file configs/ade20k/semantic-segmentation/100-5.yaml \
  CONT.TASK ${t} OUTPUT_DIR /inspurfs/group/yangsb/zhuyuchen/fake3_exp/ss/100-5_1011_klallF_crazy000/step${t}
done
