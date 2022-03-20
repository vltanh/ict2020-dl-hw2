from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances

import os
import sys
import datetime

METHOD = sys.argv[1]

register_coco_instances("chicken", {},
                        "list/train.json", "data/Dataset/train/img")

cfg = get_cfg()
if METHOD == 'ins':
    cfg.merge_from_file(
        "./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
elif METHOD == 'det':
    cfg.merge_from_file(
        "detectron2_repo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
else:
    sys.exit()

cfg.DATASETS.TRAIN = ("chicken",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4

if METHOD == 'seg':
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
elif METHOD == 'det':
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"

cfg.SOLVER.IMS_PER_BATCH = 12
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = (
    1000
)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

train_id = METHOD
train_id += '-' + datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
if METHOD == 'seg':
    cfg.OUTPUT_DIR = 'weights/seg/' + train_id + '/'
elif METHOD == 'det':
    cfg.OUTPUT_DIR = 'weights/det/' + train_id + '/'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
