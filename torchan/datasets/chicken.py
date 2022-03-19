import os
from PIL import Image
import torchvision.transforms as tvtf

from .autoaugment import ImageNetPolicy


class ChickenDataset:
    def __init__(self, root, is_train=True):
        normal = [root + '/normal/' + x
                  for x in os.listdir(root + '/normal')
                  if x.endswith('.jpg')]
        defect = [root + '/defect/' + x
                  for x in os.listdir(root + '/defect')
                  if x.endswith('.jpg')]

        self.data = normal + defect
        self.label = [0 for _ in range(len(normal))] + \
            [1 for _ in range(len(defect))]

        if is_train:
            self.transforms = tvtf.Compose([
                tvtf.Resize((224, 224)),
                ImageNetPolicy(),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = tvtf.Compose([
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            ])

    def __getitem__(self, i):
        fp = self.data[i]
        label = self.label[i]
        im = Image.open(fp).convert('RGB')
        im = self.transforms(im)
        return im, label

    def __len__(self):
        return len(self.data)
