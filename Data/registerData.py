from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.evaluation import  COCOEvaluator, inference_on_dataset
import detectron2

import os
import glob
import json
from PIL import Image
import numpy as np
import shutil

import rasterio
from rasterio import features
import shapely
from shapely.geometry import Point, Polygon

def mask_to_polygons_layer(mask):
    all_polygons = []
    for shape, value in features.shapes(mask.astype(np.int16), mask=(mask >0), transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        return shapely.geometry.shape(shape)
        all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons
def segment2bbox(segmentList):
    polygon = np.array(segmentList).reshape(-1, 2)
    minX = float(np.min(polygon[:, 0]))
    maxX = float(np.max(polygon[:, 0]))
    minY = float(np.min(polygon[:, 1]))
    maxY = float(np.max(polygon[:, 1]))
    return [minX, minY, maxX, maxY]

def segment2kp(segmentList):
    kp2D = np.array(segmentList).reshape(-1, 2)
    kpFull = np.zeros((30, 3))
    kpFull[:kp2D.shape[0], 0:2] = kp2D
    kpFull[:kp2D.shape[0], 2] = 2
    kpRes = kpFull.reshape(-1, 1)
    kpRes = kpRes.tolist()
    # kpRes = np.transpose(kpRes)
    # kpRes = kpRes.squeeze()
    return kpRes

def covertOneImage(imgName, annoName):
    with open(annoName) as f:
      annoData = json.load(f)
    resDict = {}
    resDict['file_name'] = imgName
    resDict['width'] = annoData['width']
    resDict['height'] = annoData['height']
    resDict['image_id'] = annoData['filename']
    objs = []
    label = 0
    for kindAnno in annoData['labels']:
        for instanceAnno in kindAnno['annotations']:
            obj = {
                "category_id": label,
                'bbox_mode':BoxMode.XYXY_ABS,
                'bbox': segment2bbox(instanceAnno['segmentation']),
                'segmentation':[instanceAnno['segmentation']],
                # 'keypoints': segment2kp(instanceAnno['segmentation'],)
            }
            objs.append(obj)
        label += 1
    resDict['annotations'] = objs
    return resDict


def myDataFuncTrain():
    annoRoot = os.path.join(dataRoot, 'annotations')
    trainImgRoot = os.path.join(dataRoot, 'train')
    finalDict = []
    for imgName in os.listdir(trainImgRoot):
        annoName = imgName + '-annotated.json'
        annoName = os.path.join(annoRoot, annoName)
        if not os.path.exists(annoName):
            continue
        imgName = os.path.join(trainImgRoot, imgName)
        resDict = covertOneImage(imgName, annoName)
        finalDict.append(resDict)
    return finalDict

def myDataFuncTest():
    annoRoot = os.path.join(dataRoot, 'annotations')
    testImgRoot = os.path.join(dataRoot, 'test')
    finalDict = []
    for imgName in os.listdir(testImgRoot):
        annoName = imgName + '-annotated.json'
        annoName = os.path.join(annoRoot, annoName)
        if not os.path.exists(annoName):
            continue
        imgName = os.path.join(testImgRoot, imgName)
        resDict = covertOneImage(imgName, annoName)
        finalDict.append(resDict)
    return finalDict

def splitTrainTestRand(dataRoot, trainImgRoot, testImgRoot):
    rawImgRoot = os.path.join(dataRoot, 'raw')
    # trainImgRoot = os.path.join(dataRoot, 'train')
    # testImgRoot = os.path.join(dataRoot, 'test')
    if not os.path.exists(trainImgRoot):
        os.makedirs(trainImgRoot)
    if not os.path.exists(testImgRoot):
        os.makedirs(testImgRoot)
    rawImgList = glob.glob(rawImgRoot + '/*.png')
    randInd = np.random.permutation(len(rawImgList))

    testInd = range(0, int(len(rawImgList)/5))
    trainInd = range(int(len(rawImgList) / 5), int(len(rawImgList)))

    for ind in testInd:
        shutil.copy2(rawImgList[randInd[ind]], testImgRoot)
    for ind in trainInd:
        shutil.copy2(rawImgList[randInd[ind]], trainImgRoot)

    print('Done')




dataRoot = '/home/yu/Documents/IncubitChallenge/Data'
annoRoot = os.path.join(dataRoot, 'annotations')
trainImgRoot = os.path.join(dataRoot, 'train')
testImgRoot = os.path.join(dataRoot, 'test')
# splitTrainTestRand(dataRoot, trainImgRoot, testImgRoot)


DatasetCatalog.register("TrainSet", myDataFuncTrain)
MetadataCatalog.get('TrainSet').set(thing_classes = ['Houses', 'Buildings', 'Sheds/Garages'])
DatasetCatalog.register("TestSet", myDataFuncTest)
MetadataCatalog.get('TestSet').set(thing_classes = ['Houses', 'Buildings', 'Sheds/Garages'])

# trainVisRoot = '/home/yu/Documents/IncubitChallenge/Data1/trainVis'
# testVisRoot = '/home/yu/Documents/IncubitChallenge/Data1/testVis'
# import random
# import cv2
# import matplotlib.pyplot as plt
# from detectron2.utils.visualizer import Visualizer
# dataset_dicts = myDataFuncTest()
# meta_dicts = MetadataCatalog
# for ii in range(0, len(dataset_dicts)):
#     img = cv2.imread(dataset_dicts[ii]["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get('TestSet'),  scale=0.5)
#     vis = visualizer.draw_dataset_dict(dataset_dicts[ii])
#     # plt.figure(figsize=(40, 40))
#     # plt.imshow(vis.get_image()[:, :, ::-1])
#     # plt.show()
#     picName = str.split(dataset_dicts[ii]["file_name"], '/')[-1]
#     picName = os.path.join(testVisRoot, picName)
#     plt.imsave(picName, vis.get_image()[:, :, ::-1])



from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
cfg = get_cfg()
cfg.merge_from_file("/home/yu/Documents/Facebook/detectron2/detectron2/model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("TrainSet",)
cfg.DATASETS.TEST = ("TestSet",)   # 没有测试集
cfg.DATALOADER.NUM_WORKERS = 24
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"  # 模型微调
cfg.SOLVER.IMS_PER_BATCH = 6
cfg.SOLVER.MAX_ITER = 4000
cfg.SOLVER.STEPS= [200,]

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  #仅有一类(ballon)
# cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
# cfg.MODEL.RPN.IN_FEATURES = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
# trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 #测试的阈值
cfg.DATASETS.TEST = ("TestSet", )
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("TestSet", output_dir='./output')

val_loader = build_detection_test_loader(cfg, "TestSet")
inference_on_dataset(predictor.model, val_loader, evaluator)



#随机选部分图片，可视化
predVisRoot = '/home/yu/Documents/IncubitChallenge/Data1/predVis'

meta_dicts = MetadataCatalog.get('TestSet')
import random
import cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
dataset_dicts = myDataFuncTest()
for d in myDataFuncTest():
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=meta_dicts,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW # 去除非气球区域的像素颜色.
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    res = outputs["instances"].to("cpu")

    # plt.figure(figsize=(40, 40))
    # plt.imshow(v.get_image()[:, :, ::-1])
    # plt.show()
    import rasterio

    picName = str.split(d["file_name"], '/')[-1]
    picName = os.path.join(predVisRoot, picName)
    plt.imsave(picName, v.get_image()[:, :, ::-1])




