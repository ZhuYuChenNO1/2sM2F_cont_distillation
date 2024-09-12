import os
import sys
import runpy

# os.chdir('WORKDIR')
# args = 'python test.py 4 5'
args = 'python train_continual.py --dist-url auto --num-gpus 1 --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
CONT.TASK 2 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR ./output/ps/100-5_distill_freezelabel_tt/step2 \
CONT.WEIGHTS model_final.pth'
# args = 'python tools/train_net.py --config-file projects/deformable_detr/configs/deformable_detr_r50_two_stage_50ep.py --num-gpus 2' 
# args = 'python tools/train_net.py --config-file projects/deformable_detr/configs/deformable_detr_r50_two_stage_90k_cocolvis.py --num-gpus 1 train.init_checkpoint=model_final.pth'

args = args.split()
if args[0] == 'python':
    """pop up the first in the args""" 
    args.pop(0)
if args[0] == '-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path
sys.argv.extend(args[1:])
fun(args[0], run_name='__main__')
