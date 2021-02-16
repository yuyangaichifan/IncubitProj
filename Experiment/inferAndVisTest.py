import json
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.evaluation import  COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file
import os
from Data.preProcessData import myDataFuncTrain, myDataFuncTest
DatasetCatalog.register("TestSetBatch", myDataFuncTest)
MetadataCatalog.get('TestSetBatch').set(thing_classes = ['Houses', 'Buildings', 'Sheds/Garages'])

dataRoot = '/home/yu/Documents/IncubitChallenge/Data_New'
visRoot = os.path.join(dataRoot, 'visRes')
predVisRoot = os.path.join(visRoot, 'predRes')
predPlVisRoot = os.path.join(visRoot, 'predPolyRes')
predMkVisRoot = os.path.join(visRoot, 'predMaskRes')
if not os.path.exists(predVisRoot):
    os.mkdir(predVisRoot)
if not os.path.exists(predPlVisRoot):
    os.mkdir(predPlVisRoot)
if not os.path.exists(predMkVisRoot):
    os.mkdir(predMkVisRoot)

cfg = get_cfg()
cfg.merge_from_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.DATALOADER.NUM_WORKERS = 24
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 #测试的阈值
cfg.DATASETS.TEST = ("TestSetBatch", )
predictor = DefaultPredictor(cfg)

import cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask
dataset_dicts = myDataFuncTest()
meta_dicts = MetadataCatalog.get('TestSetBatch')
from  rdp import rdp
for ii in range(0, len(dataset_dicts)):
    img = cv2.imread(dataset_dicts[ii]["file_name"])
    vPred = Visualizer(img[:, :, ::-1],
                   metadata=meta_dicts,
                   scale=1.0,
                   instance_mode=ColorMode.IMAGE_BW
                   )
    vPoly = Visualizer(img[:, :, ::-1],
                       metadata=meta_dicts,
                       scale=1.0,
                       instance_mode=ColorMode.IMAGE_BW
                       )
    vMask = Visualizer(img[:, :, ::-1],
                       metadata=meta_dicts,
                       scale=1.0,
                       instance_mode=ColorMode.IMAGE_BW
                       )
    outputs = predictor(img)
    res = outputs["instances"].to("cpu")
    vPred.output = vPred.draw_instance_predictions(res)

    picName = str.split(dataset_dicts[ii]["file_name"], '/')[-1]
    picName = os.path.join(predVisRoot, picName)
    plt.imsave(picName, vPred.output.get_image()[:, :, ::-1])

    height = res.image_size[0]
    width = res.image_size[1]
    masks = [GenericMask(x.numpy(), height, width).mask_to_polygons(x.numpy()) for x in res.pred_masks]
    masksBin = [GenericMask(x.numpy(), height, width).mask for x in res.pred_masks]
    labels = [int(x.numpy()) for x in res.pred_classes]

    count = 0
    colorMaps = [[0., 0., 1.], [1., 0., 0.], [1.0, 0.3, .5]]
    for mask in masks:
        colorIn = colorMaps[labels[count]]
        maskRaw = mask[0][0].reshape(-1, 2)
        maskSim = rdp(maskRaw, epsilon=6)
        vPoly.output = vPoly.draw_polygon(maskSim, colorIn, alpha=0.3)
        vMask.output = vMask.draw_binary_mask(masksBin[count], colorIn, alpha=0.3)
        count += 1
    picName = str.split(dataset_dicts[ii]["file_name"], '/')[-1]
    picName = os.path.join(predPlVisRoot, picName)
    plt.imsave(picName, vPoly.output.get_image()[:, :, ::-1])

    picName = str.split(dataset_dicts[ii]["file_name"], '/')[-1]
    picName = os.path.join(predMkVisRoot, picName)
    plt.imsave(picName, vMask.output.get_image()[:, :, ::-1])
