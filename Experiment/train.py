import sys
import os
import argparse
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import  COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from Data.preProcessData import createDataFunc
sys.path.append(".")
def trainModel(trainDir, annoRoot, taskName, testDir=None):
    trainFunc = createDataFunc(trainDir, annoRoot)
    DatasetCatalog.register("TrainSetBatch", trainFunc)
    MetadataCatalog.get('TrainSetBatch').set(thing_classes=['Houses', 'Buildings', 'Sheds/Garages'])
    cfg = get_cfg()
    cfg.merge_from_file("Cfg/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("TrainSetBatch",)
    cfg.DATALOADER.NUM_WORKERS = 24
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"  # 模型微调
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.MAX_ITER = 6000
    cfg.SOLVER.STEPS = [200, ]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 仅有一类(ballon)
    cfg.OUTPUT_DIR = './' + taskName + '_output'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    if testDir == None:
        return
    else:
        testFunc = createDataFunc(testDir, annoRoot)
        DatasetCatalog.register("TestSetBatch", testFunc)
        MetadataCatalog.get('TestSetBatch').set(thing_classes=['Houses', 'Buildings', 'Sheds/Garages'])
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 测试的阈值
        cfg.DATASETS.TEST = ("TestSetBatch",)
        predictor = DefaultPredictor(cfg)
        evaluator = COCOEvaluator("TestSetBatch", output_dir='./output')
        val_loader = build_detection_test_loader(cfg, "TestSetBatch")
        inference_on_dataset(predictor.model, val_loader, evaluator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='new')
    parser.add_argument("--taskName", type=str)
    parser.add_argument("--dataRoot", type=str)
    args = parser.parse_args()


    if args.taskName == 'Exp':
        trainDir = os.path.join(args.dataRoot, args.taskName, 'trainBatch')
        annoRoot = os.path.join(args.dataRoot, args.taskName, 'annoBatch')
        testDir = os.path.join(args.dataRoot, args.taskName, 'testBatch')
        trainModel(trainDir, annoRoot, args.taskName, testDir=testDir)
    elif args.taskName == 'Demo':
        trainDir = os.path.join(args.dataRoot, args.taskName, 'trainBatch')
        annoRoot = os.path.join(args.dataRoot, args.taskName, 'annoBatch')
        trainModel(trainDir, annoRoot, args.taskName)
    else:
        print("Task name must be 'Exp' or 'Demo'! ")


