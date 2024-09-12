import os
import sys
import runpy

# os.chdir('WORKDIR')
# args = 'python test.py 4 5'
args = 'python visualize_data.py --source annotation --config-file configs/ade20k/panoptic-segmentation/maskformer2_R50_bs16_160k.yaml --output-dir ./try_debug'
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
