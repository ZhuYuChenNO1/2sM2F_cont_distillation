python train_net.py --num-gpus 8  --config-file configs/ade20k/panoptic-segmentation/maskformer2_R50_bs16_160k.yaml\
 --eval-only MODEL.WEIGHTS outputs/ade20k_panoptic_SCE_twoS_v2_selfattnfirst_100q_lr/model_final.pth
