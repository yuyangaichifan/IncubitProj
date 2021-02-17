import json
import  matplotlib.patches as patches
import os
import glob
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, mapping
from detectron2.structures import BoxMode
import cv2
import shutil
import json
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.evaluation import  COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
import argparse

def segment2bbox(segmentList):
    polygon = np.array(segmentList).reshape(-1, 2)
    minX = float(np.min(polygon[:, 0]))
    maxX = float(np.max(polygon[:, 0]))
    minY = float(np.min(polygon[:, 1]))
    maxY = float(np.max(polygon[:, 1]))
    return [minX, minY, maxX, maxY]

def showAnnoCoco(annoName, visRoot):
    batchVisRoot = os.path.join(visRoot, 'batchVis')
    if not os.path.exists(batchVisRoot):
        os.mkdir(batchVisRoot)
    with open(annoName) as f:
      annoData = json.load(f)
    img = Image.open(annoData['file_name'])
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    ax = plt.gca()
    for tmpAnno in annoData['annotations']:
        polygon = np.array(tmpAnno['segmentation'])
        polygon = polygon.reshape(-1, 2)
        polyStruct = patches.Polygon(polygon, True, edgecolor='k', alpha=0.4, linewidth=4)
        ax.add_patch(polyStruct)
        bbox = tmpAnno['bbox']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.title(annoData['file_name'])
    plt.savefig(os.path.join(batchVisRoot, str.split(annoData['file_name'], '/')[-1]))

def handleBatchAnno(batchImgNameFull, batchAnnoNameFull, annoData, startH, endH, startW, endW, tmpImg):
    resDict = {}
    resDict['file_name'] = batchImgNameFull
    resDict['height'] = tmpImg.shape[0]
    resDict['width'] = tmpImg.shape[1]
    resDict['image_id'] = str.split(batchImgNameFull, '/')[-1]
    objs = []
    label = 0
    for kindAnno in annoData['labels']:
        for instanceAnno in kindAnno['annotations']:
            segPolygon = np.array(instanceAnno['segmentation']).reshape(-1, 2)
            p1 = Polygon(segPolygon)
            p2 = Polygon([[startW, startH], [startW, endH], [endW, endH], [endW, startH]])
            if p1.intersects(p2):
                try:
                    InterPolygon = p1.intersection(p2)
                    InterPolygonCord = np.array(mapping(InterPolygon)['coordinates'])[0]
                    InterPolygonCord[:, 0] = InterPolygonCord[:, 0] - startW
                    InterPolygonCord[:, 1] = InterPolygonCord[:, 1] - startH
                    InterPolygonCord = InterPolygonCord.reshape(-1, 1).squeeze().tolist()

                    obj = {
                        "category_id": label,
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'bbox': segment2bbox(InterPolygonCord),
                        'segmentation': [InterPolygonCord],
                        # 'keypoints': segment2kp(instanceAnno['segmentation'],)
                    }

                except:
                    continue
                else:
                    objs.append(obj)

        label += 1
    resDict['annotations'] = objs
    with open(batchAnnoNameFull, 'w') as fp:
        json.dump(resDict, fp)

def splitOneImage(imgNameFull, batchImgRoot, batchAnnoRoot, annoName, divNum, visRoot=None):
    with open(annoName) as f:
      annoData = json.load(f)
    img = plt.imread(imgNameFull)

    hImg, wImg = img.shape[0], img.shape[1]
    stepH, stepW = int(hImg/divNum), int(wImg/divNum)
    imgName = str.split(imgNameFull, '/')[-1]
    for ii in range(divNum):
        startH = ii * stepH
        endH = hImg if ii == divNum - 1 else  stepH*(ii + 1) -1
        for jj in range(divNum):
            startW = jj * stepW
            endW = wImg if jj == divNum - 1 else stepW * (jj + 1) -1

            tmpImg = img[startH:endH+1, startW:endW+1, :]
            batchID = ii*divNum + jj
            batchImgNameFull = os.path.join(batchImgRoot, imgName[:-4] + '_' + str(batchID)+ '.png')
            plt.imsave(batchImgNameFull, tmpImg)
            batchAnnoNameFull = os.path.join(batchAnnoRoot, imgName[:-4] + '_' + str(batchID) + '.json')
            handleBatchAnno(batchImgNameFull, batchAnnoNameFull, annoData, startH, endH, startW, endW, tmpImg)

def myDataFuncTrain():
    dataRoot = '/home/yu/Documents/IncubitChallenge/Data_New'
    annoRoot = os.path.join(dataRoot, 'batchAnno')
    trainImgRoot = os.path.join(dataRoot, 'trainBatch')
    finalDict = []
    for imgName in os.listdir(trainImgRoot):
        annoName = imgName[:-4] + '.json'
        annoName = os.path.join(annoRoot, annoName)
        if not os.path.exists(annoName):
            continue
        imgName = os.path.join(trainImgRoot, imgName)
        with open(annoName) as f:
            resDict = json.load(f)
        for ii in range(len(resDict['annotations'])):
            resDict['annotations'][ii]['bbox_mode'] = BoxMode.XYXY_ABS
        finalDict.append(resDict)
    return finalDict

def myDataFuncTest():
    dataRoot = '/home/yu/Documents/IncubitChallenge/Data_New'
    annoRoot = os.path.join(dataRoot, 'batchAnno')
    testImgRoot = os.path.join(dataRoot, 'testBatch')
    finalDict = []
    for imgName in os.listdir(testImgRoot):
        annoName = imgName[:-4] + '.json'
        annoName = os.path.join(annoRoot, annoName)
        if not os.path.exists(annoName):
            continue
        imgName = os.path.join(testImgRoot, imgName)
        with open(annoName) as f:
            resDict = json.load(f)
        for ii in range(len(resDict['annotations'])):
            resDict['annotations'][ii]['bbox_mode'] = BoxMode.XYXY_ABS
        finalDict.append(resDict)
    return finalDict


def createDataFunc(imgRoot, annoRoot):
    def myDataFunc():
        finalDict = []
        for imgName in os.listdir(imgRoot):
            annoName = imgName[:-4] + '.json'
            annoName = os.path.join(annoRoot, annoName)
            if not os.path.exists(annoName):
                continue
            with open(annoName) as f:
                resDict = json.load(f)
            for ii in range(len(resDict['annotations'])):
                resDict['annotations'][ii]['bbox_mode'] = BoxMode.XYXY_ABS
            finalDict.append(resDict)
        return finalDict

    return myDataFunc


def splitTrainTestRand(rawImgRoot, trainImgRoot, testImgRoot):
    rawImgList = glob.glob(rawImgRoot + '/*.png')
    randInd = np.random.permutation(len(rawImgList))
    testInd = range(0, int(len(rawImgList)/5))
    trainInd = range(int(len(rawImgList) / 5), int(len(rawImgList)))
    for ind in testInd:
        shutil.copy2(rawImgList[randInd[ind]], testImgRoot)
    for ind in trainInd:
        shutil.copy2(rawImgList[randInd[ind]], trainImgRoot)


# def visGT(dataRoot):
#     myDataFuncTrain = createDataFunc(dataRoot)
#     DatasetCatalog.register("TrainSetBatch", myDataFuncTrain)
#     MetadataCatalog.get('TrainSetBatch').set(thing_classes=['Houses', 'Buildings', 'Sheds/Garages'])
#     DatasetCatalog.register("TestSetBatch", myDataFuncTest)
#     MetadataCatalog.get('TestSetBatch').set(thing_classes=['Houses', 'Buildings', 'Sheds/Garages'])
#
#     trainVisRoot = os.path.join(visRoot, 'trainVisGT')
#     testVisRoot = os.path.join(visRoot, 'testVisGT')
#     if not os.path.exists(trainVisRoot):
#         os.mkdir(trainVisRoot)
#     if not os.path.exists(testVisRoot):
#         os.mkdir(testVisRoot)
#     dataset_dicts = myDataFuncTest()
#     for ii in range(0, len(dataset_dicts)):
#         img = cv2.imread(dataset_dicts[ii]["file_name"])
#         visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get('TestSetBatch'), scale=1.0)
#         vis = visualizer.draw_dataset_dict(dataset_dicts[ii])
#         picName = str.split(dataset_dicts[ii]["file_name"], '/')[-1]
#         picName = os.path.join(testVisRoot, picName)
#         plt.imsave(picName, vis.get_image()[:, :, ::-1])
#
#     dataset_dicts = myDataFuncTrain()
#     for ii in range(0, len(dataset_dicts)):
#         img = cv2.imread(dataset_dicts[ii]["file_name"])
#         visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get('TrainSetBatch'), scale=1.0)
#         vis = visualizer.draw_dataset_dict(dataset_dicts[ii])
#         picName = str.split(dataset_dicts[ii]["file_name"], '/')[-1]
#         picName = os.path.join(trainVisRoot, picName)
#         plt.imsave(picName, vis.get_image()[:, :, ::-1])

def createDirsExp(dataRoot, imgRawRoot, annoRawRoot):
    expRoot = os.path.join(dataRoot, 'Exp')
    if not os.path.exists(expRoot): os.mkdir(expRoot)
    imgFull = os.path.join(expRoot, 'imgFull')
    if not os.path.exists(imgFull): os.mkdir(imgFull)
    imgBatch = os.path.join(expRoot, 'imgBatch')
    if not os.path.exists(imgBatch): os.mkdir(imgBatch)
    trainBatch = os.path.join(expRoot, 'trainBatch')
    if not os.path.exists(trainBatch): os.mkdir(trainBatch)
    testBatch = os.path.join(expRoot, 'testBatch')
    if not os.path.exists(testBatch): os.mkdir(testBatch)
    annoFull = os.path.join(expRoot, 'annoFull')
    if not os.path.exists(annoFull): os.mkdir(annoFull)
    annoBatch = os.path.join(expRoot, 'annoBatch')
    if not os.path.exists(annoBatch): os.mkdir(annoBatch)
    # copy raw images
    # rawImgList = glob.glob(imgRawRoot + '/*.png')
    # for file in rawImgList:
    #     shutil.copy2(file, imgFull)
    # copy raw anno
    rawAnnoList = glob.glob(annoRawRoot + '/*.json')
    for file in rawAnnoList:
        shutil.copy2(file, annoFull)
        imgName = os.path.join(imgRawRoot, str.split(file[:-15], '/')[-1])
        shutil.copy2(imgName, imgFull)
    # split image into batches
    for imgName in os.listdir(imgFull):
        annoName = imgName + '-annotated.json'
        annoName = os.path.join(annoFull, annoName)
        if not os.path.exists(annoName):
            continue
        imgName = os.path.join(imgFull, imgName)
        splitOneImage(imgName, imgBatch, annoBatch, annoName, 2)
    splitTrainTestRand(imgBatch, trainBatch, testBatch)
    print("Exp root done!")

def createDirsDemo(dataRoot, imgRawRoot, annoRawRoot):
    demoRoot = os.path.join(dataRoot, 'Demo')
    if not os.path.exists(demoRoot): os.mkdir(demoRoot)
    imgFull = os.path.join(demoRoot, 'imgFull')
    if not os.path.exists(imgFull): os.mkdir(imgFull)
    imgBatch = os.path.join(demoRoot, 'imgBatch')
    if not os.path.exists(imgBatch): os.mkdir(imgBatch)
    trainBatch = os.path.join(demoRoot, 'trainBatch')
    if not os.path.exists(trainBatch): os.mkdir(trainBatch)
    annoFull = os.path.join(demoRoot, 'annoFull')
    if not os.path.exists(annoFull): os.mkdir(annoFull)
    annoBatch = os.path.join(demoRoot, 'annoBatch')
    if not os.path.exists(annoBatch): os.mkdir(annoBatch)
    # copy raw images

    rawAnnoList = glob.glob(annoRawRoot + '/*.json')
    for file in rawAnnoList:
        shutil.copy2(file, annoFull)
        imgName = os.path.join(imgRawRoot, str.split(file[:-15], '/')[-1])
        shutil.copy2(imgName, imgFull)
    # split image into batches
    for imgName in os.listdir(imgFull):
        annoName = imgName + '-annotated.json'
        annoName = os.path.join(annoFull, annoName)
        if not os.path.exists(annoName):
            continue
        imgName = os.path.join(imgFull, imgName)
        splitOneImage(imgName, imgBatch, annoBatch, annoName, 2)
    for imgName in os.listdir(imgBatch):
        imgName = os.path.join(imgBatch, imgName)
        shutil.copy2(imgName, trainBatch)
    print("Demo root done!")

def main(dataRoot):
    imgRawRoot = os.path.join(dataRoot, 'raw')
    annoRawRoot = os.path.join(dataRoot, 'annotations')
    createDirsExp(dataRoot, imgRawRoot, annoRawRoot)
    createDirsDemo(dataRoot, imgRawRoot, annoRawRoot)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='new')
    parser.add_argument("--dataRoot", type=str)
    args = parser.parse_args()
    main(args.dataRoot)


