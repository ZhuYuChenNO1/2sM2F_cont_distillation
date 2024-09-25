from detectron2.config import CfgNode as CN


def add_continual_config(cfg):
    cfg.CONT = CN()
    cfg.CONT.TOT_CLS = 150
    cfg.CONT.BASE_CLS = 100
    cfg.CONT.INC_CLS = 10
    cfg.CONT.SETTING = 'overlapped'
    cfg.CONT.TASK = 1
    cfg.CONT.WEIGHTS = None
    cfg.CONT.OLD_WEIGHTS = None
    cfg.CONT.MED_TOKENS_WEIGHT = 1.0
    cfg.CONT.MEMORY = False
    cfg.CONT.PSD_LABEL_THRESHOLD = 0.35
    cfg.CONT.COLLECT_QUERY_MODE = False
    cfg.CONT.CUMULATIVE_PSDNUM = False
    cfg.CONT.WEIGHTED_SAMPLE = True
    cfg.CONT.LIB_SIZE = 80
    cfg.CONT.VQ_NUMBER = 3
    cfg.CONT.FREEZE_LABEL = False
    cfg.CONT.KL_ALL = True
    cfg.CONT.KL_WEIGHT = 2.0
    cfg.CONT.DISTRIBUTION_ALPHA = 0.5
