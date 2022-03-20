# **Homework 2 - Competition**

## **Info**

Name: The-Anh Vu-Le

Student ID: 20C13002

Course: Introduction to Deep Learning

Email: anh.vu2020@ict.jvn.edu.vn

## **Detection**

### **Setup**

```
conda install shapely scikit-image numpy Pillow
pip install -e detectron2_repo
```

### **Usage**

To generate the COCO-formatted JSON
```
python mask2coco.py data/Dataset/train/mask
```


```
python train_detect.py det
```

If you want to train Instance Segmentation, then replace `det` with `ins`.

## **Classification**

### **Setup**

```
conda install tqdm yaml
conda install -c pytorch pytorch torchvision cudatoolkit
pip install efficientnet_pytorch
```

### **Usage**

```
python train_cls.py --config configs/chicken.yaml --gpus 0
```

## **Inference**

To perform instance segmentation on the images in `data/Dataset/test/` using the weights stored in `output/ins-1000/model_final.pth`
```
python infer_seg.py data/Dataset/test/ output/ins-1000/model_final.pth
```

To perform detection then classification on the images in `data/Dataset/test/` using the weights stored in `output/det-1000/model_final.pth` (detection) and `runs/chicken_ResNet50-2022_03_19-12_29_20/best_metric_F1.pth` (classification)
```
python infer_cls.py data/Dataset/test/ output/det-1000/model_final.pth runs/chicken_ResNet50-2022_03_19-12_29_20/best_metric_F1.pth
```