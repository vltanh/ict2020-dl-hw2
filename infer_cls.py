import csv
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from PIL import Image
import numpy as np
import torchvision.transforms as tvtf
import torch.nn.functional as F

from src.utils import crop_object
from torchan.utils.getter import get_instance

import os
import sys
import torch

INPUT = sys.argv[1]
DET_MODEL = sys.argv[2]
CLS_MODEL = sys.argv[3]

DET_THRESHOLD = 0.9

OUTDIR = 'result_cls'

# DETECTION
cfg = get_cfg()
cfg.merge_from_file(
    "detectron2_repo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

cfg.MODEL.WEIGHTS = DET_MODEL

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

# CLASSIFICATION
device = torch.device('cuda')
config = torch.load(CLS_MODEL, map_location='cuda')
model = get_instance(config['config']['model']).to(device)
model.load_state_dict(config['model_state_dict'])

tfs = tvtf.Compose([
    tvtf.Resize((224, 224)),
    tvtf.ToTensor(),
    tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
])

out = []
for fn in os.listdir(INPUT):
    _id = os.path.splitext(fn)[0]

    im_pil = Image.open(INPUT + '/' + fn)

    # DETECT
    im = np.array(im_pil)

    outputs = predictor(im)

    boxes = outputs['instances'].pred_boxes.tensor.detach().cpu()
    scores = outputs['instances'].scores.detach().cpu()

    sortidx = boxes[:, 0].argsort()
    boxes = boxes[sortidx]
    scores = scores[sortidx]

    # CLASSIFY
    with torch.no_grad():
        for i, (bbox, score) in enumerate(zip(boxes, scores)):
            if score > DET_THRESHOLD:
                patch = crop_object(im_pil, bbox)
                patch = tfs(patch)
                output = model(patch.unsqueeze(0).cuda())
                probs = F.softmax(output, dim=1)
                confs, preds = torch.max(probs, dim=1)
                out.append([fn, preds.detach().cpu().tolist()[0]])

    # LOG
    os.makedirs(f'{OUTDIR}/det/{_id}/good', exist_ok=True)
    os.makedirs(f'{OUTDIR}/det/{_id}/bad', exist_ok=True)
    for i, (bbox, score) in enumerate(zip(boxes, scores)):
        patch = crop_object(im_pil, bbox)
        if score > DET_THRESHOLD:
            patch.save(f'{OUTDIR}/det/{_id}/good/{i}.png')
        else:
            patch.save(f'{OUTDIR}/det/{_id}/bad/{i}.png')

    v = Visualizer(im)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    Image.fromarray(v.get_image()).save(f'{OUTDIR}/{_id}/{_id}.png')

with open(f'{OUTDIR}/classification.txt', 'w') as f:
    w = csv.writer(f)
    w.writerow(['filename', 'prediction'])
    w.writerows(out)
