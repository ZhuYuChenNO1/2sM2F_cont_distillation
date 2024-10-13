# python train_continual.py --dist-url auto --resume --num-gpus 8 --config-file configs/ade20k/panoptic-segmentation/50-50.yaml \
# CONT.TASK 1 SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 160000 OUTPUT_DIR ./output/ps/50-50/step1

itration=(1650 1174)

for t in 2 3; do
    # 计算索引
    index=$((t - 2))  # 数组索引从 0 开始，因此 t=3 对应数组中的索引 2

    # 确保索引在有效范围内
    if [ $index -ge 0 ] && [ $index -lt ${#itration[@]} ]; then
        iter=${itration[$index]}
        
        # 运行第一个 Python 命令
        python train_continual.py --dist-url auto --num-gpus 1 --config-file configs/ade20k/panoptic-segmentation/50-50.yaml \
            CONT.TASK ${t} SOLVER.BASE_LR 0.0 SOLVER.MAX_ITER $iter CONT.COLLECT_QUERY_MODE True OUTPUT_DIR ./output/ps/50-50/step${t} \
            # CONT.WEIGHTS /public/home/zhuyuchen530/projects/cvpr24/2sM2F_cont/output/ps/100-10_all_clsembed/step1/model_final.pth

        # 运行第二个 Python 命令
        python train_continual.py --dist-url auto --num-gpus 4 --config-file configs/ade20k/panoptic-segmentation/50-50.yaml \
            CONT.TASK ${t} SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 50000 CONT.LIB_SIZE 80 CONT.COLLECT_QUERY_MODE False OUTPUT_DIR ./output/ps/50-50/step${t} \
            # CONT.WEIGHTS /public/home/zhuyuchen530/projects/cvpr24/2sM2F_cont_distillation/output/ps/100-5_passSelfCross/step$((t-1))/model_final.pth
    else
        echo "Index $index out of range for itratioin array"
    fi
done