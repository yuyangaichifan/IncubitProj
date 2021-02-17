import sys
sys.path.append("./")
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
import os, cv2, json
from rdp import rdp
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask
import matplotlib.pyplot as plt
import argparse

def predOneImg(predictor, imgNameFull, annoResName, visRoot, thing_classes, divNum=2, isVis = 0):

    img = cv2.imread(imgNameFull)
    hImg, wImg = img.shape[0], img.shape[1]
    stepH, stepW = int(hImg/divNum), int(wImg/divNum)
    imgName = str.split(imgNameFull, '/')[-1]
    resDict = {'filename':imgName, 'labels':[]}
    for ii in range(len(thing_classes)):
        resDict['labels'].append({'name':thing_classes[ii], 'annotations':[]})
    if isVis == 1:
        vPoly = Visualizer(img[:, :, ::-1],
                           scale=1.0,
                           instance_mode=ColorMode.IMAGE_BW
                           )
    segmentCount = 0
    for ii in range(divNum):
        startH = ii * stepH
        endH = hImg if ii == divNum - 1 else  stepH*(ii + 1) -1
        for jj in range(divNum):
            print(ii, jj)
            startW = jj * stepW
            endW = wImg if jj == divNum - 1 else stepW * (jj + 1) -1
            tmpImg = img[startH:endH+1, startW:endW+1, :]
            outputs = predictor(tmpImg)
            res = outputs["instances"].to("cpu")
            height = res.image_size[0]
            width = res.image_size[1]
            masks = [GenericMask(x.numpy(), height, width).mask_to_polygons(x.numpy()) for x in res.pred_masks]
            # masks1 = [x[0][0].reshape(-1, 2)+[startW, startH] for x in masks]
            labels = [int(x.numpy()) for x in res.pred_classes]
            colorMaps = [[0., 0., 1.], [1., 0., 0.], [1.0, 0.3, .5]]
            for ll in range(len(labels)):
                tmpLabel = thing_classes[labels[ll]]
                if len(masks[ll][0]) == 0:
                    continue
                tmpsegmentNP = masks[ll][0][0].reshape(-1, 2)+[startW, startH]
                tmpsegmentRDP = rdp(tmpsegmentNP, epsilon=6)
                tmpsegment = tmpsegmentRDP.reshape(-1, 1).squeeze().tolist()
                tmpInst = {'id':segmentCount, 'segmentation':tmpsegment}
                for cc in range(len(resDict['labels'])):
                    if resDict['labels'][cc]['name'] == tmpLabel:
                        resDict['labels'][cc]['annotations'].append(tmpInst)
                        if isVis ==  1:
                            vPoly.output = vPoly.draw_polygon(tmpsegmentRDP, colorMaps[cc], alpha=0.3)
                            # plt.imshow(vPoly.output.get_image()[:, :, ::-1])
                            # plt.show()
                segmentCount += 1

    with open(annoResName, 'w') as fp:
        json.dump(resDict, fp)
    if isVis == 1:
        picName = os.path.join(visRoot, imgName)
        plt.imsave(picName, vPoly.output.get_image()[:, :, ::-1])

def main(dataRoot, taskName, isVis):
    cfg = get_cfg()
    cfg.merge_from_file("Cfg/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("TrainSetBatch",)
    cfg.DATALOADER.NUM_WORKERS = 24
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.OUTPUT_DIR = './' + taskName + '_output'
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 测试的阈值
    predictor = DefaultPredictor(cfg)

    thing_classes = ['Houses', 'Buildings', 'Sheds/Garages']

    imgPath = os.path.join(dataRoot, taskName, 'imgInfer')
    resPath = os.path.join(dataRoot, taskName, 'resInfer')
    visRoot = os.path.join(dataRoot, taskName, 'visInfer')
    if not os.path.exists(imgPath):
        os.mkdir(imgPath)
    if not os.path.exists(resPath):
        os.mkdir(resPath)
    if not os.path.exists(visRoot):
        os.mkdir(visRoot)

    for imgName in os.listdir(imgPath):
        imgNameFull = os.path.join(imgPath, imgName)
        resNameFull = os.path.join(resPath, imgName[:-4]+'.json')
        print(imgName)

        predOneImg(predictor, imgNameFull, resNameFull, visRoot, thing_classes, divNum=2, isVis=isVis)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='new')
    parser.add_argument("--taskName", type=str)
    parser.add_argument("--dataRoot", type=str)
    parser.add_argument("--isVis", type=int, default=0)
    args = parser.parse_args()

    main(args.dataRoot, args.taskName, args.isVis)

