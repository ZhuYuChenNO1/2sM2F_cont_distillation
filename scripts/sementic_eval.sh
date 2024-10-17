for t in 2; do
  python train_continual.py --dist-url auto --num-gpus 8 --eval-only --config-file configs/ade20k/semantic-segmentation/100-5.yaml \
  CONT.TASK ${t} OUTPUT_DIR /public/home/zhuyuchen530/projects/cvpr24/fake3/output/ss/100-5_0_01_all/step${t}
done
