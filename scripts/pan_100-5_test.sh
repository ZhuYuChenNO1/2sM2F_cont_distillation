#!/bin/bash
# python train_continual.py --dist-url auto --num-gpus 4 --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
# CONT.TASK 1 SOLVER.BASE_LR 0.0 SOLVER.MAX_ITER 2500 CONT.LIB_SIZE 80 CONT.COLLECT_QUERY_MODE False OUTPUT_DIR ./output/ps/test_ddp_lib80/step1 \
# MODEL.WEIGHTS /public/home/zhuyuchen530/projects/cvpr24/2sM2F_cont_distillation/output/ps/100-5_passSelfCross_fakequery_psdWeight/step1/model_final.pth
# 定义一个包含所有迭代次数的数组
itratioin=(142 75 154 150 143 234 225 367 154 237)

# 遍历任务 t
for t in 4 5 6 7 8 9 10 11; do
    # 计算索引
    index=$((t - 2))  # 数组索引从 0 开始，因此 t=3 对应数组中的索引 2

    # 确保索引在有效范围内
    if [ $index -ge 0 ] && [ $index -lt ${#itratioin[@]} ]; then
        iter=${itratioin[$index]}
        
        # 运行第一个 Python 命令
        python train_continual.py --dist-url auto --num-gpus 1 --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
            CONT.TASK ${t} SOLVER.BASE_LR 0.0 SOLVER.MAX_ITER $iter CONT.COLLECT_QUERY_MODE True OUTPUT_DIR ./output/ps/test_ddp_lib80_maskstuff/step${t} \
            # CONT.WEIGHTS /public/home/zhuyuchen530/projects/cvpr24/2sM2F_cont/output/ps/100-10_all_clsembed/step1/model_final.pth

        # 运行第二个 Python 命令
        python train_continual.py --dist-url auto --num-gpus 8 --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
            CONT.TASK ${t} SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 CONT.LIB_SIZE 80 CONT.COLLECT_QUERY_MODE False OUTPUT_DIR ./output/ps/test_ddp_lib80_maskstuff/step${t} \
            # CONT.WEIGHTS /public/home/zhuyuchen530/projects/cvpr24/2sM2F_cont_distillation/output/ps/100-5_passSelfCross/step$((t-1))/model_final.pth
    else
        echo "Index $index out of range for itratioin array"
    fi
done

