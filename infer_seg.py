from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from PIL import Image
import numpy as np

import os
import sys

INPUT = sys.argv[1]
SEG_MODEL = sys.argv[2]

# Load model
cfg = get_cfg()
cfg.merge_from_file(
    "./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

cfg.MODEL.WEIGHTS = SEG_MODEL

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9

predictor = DefaultPredictor(cfg)

for fn in os.listdir(INPUT):
    _id = os.path.splitext(fn)[0]

    im_pil = Image.open(INPUT + '/' + fn)
    im = np.array(im_pil)
    outputs = predictor(im)

    os.makedirs(f'result_seg', exist_ok=True)

    v = Visualizer(im)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    Image.fromarray(v.get_image()).save(f'result_seg/{_id}.png', )
