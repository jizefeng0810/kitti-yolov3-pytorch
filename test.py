from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils.logger import logger_init
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--model_config_path", type=str, default="config/yolov3-kitti.cfg", help="path to model config file")
    parser.add_argument("--data_config_path", type=str, default="config/kitti.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/kitti_best.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/kitti.names", help="path to class label file")
    parser.add_argument("--log_path", type=str, default="./all_test_log.txt", help="path to log some information")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.45, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
    return parser.parse_args()

if __name__=='__main__':
    opt = args_parser()
    logger = logger_init(opt.log_path)
    logger.info(opt)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() and opt.use_cuda else 'cpu')
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() and opt.use_cuda else torch.FloatTensor

    # Get data configuration
    data_config = parse_data_config(opt.data_config_path)
    test_path = data_config["valid"]
    num_classes = int(data_config["classes"])

    # Initiate model
    model = Darknet(opt.model_config_path).to(DEVICE)
    model.load_weights(opt.weights_path)
    model.eval()    # 测试必备

    logger.info("Compute mAP...")
    dataset = ListDataset(test_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    all_detections = []
    all_annotations = []
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        imgs = Variable(imgs.type(Tensor))

        # 模型输出 + 非极大抑制
        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, num_classes, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

        # 对输出按得分排序
        for output, annotations in zip(outputs, targets):
            all_detections.append([np.array([]) for _ in range(num_classes)])   # num_class个空数组
            if output is not None:
                # Get predicted boxes, confidence scores and labels
                pred_boxes = output[:, :5].cpu().numpy()
                scores = output[:, 4].cpu().numpy()
                pred_labels = output[:, -1].cpu().numpy()
                # Order by confidence
                sort_i = np.argsort(scores)  # 排序后返回索引值
                pred_labels = pred_labels[sort_i]  # 获得排序后的label
                pred_boxes = pred_boxes[sort_i]  # 获得排序后的boxes
                for label in range(num_classes):  # record同一类所有boxes
                    all_detections[-1][label] = pred_boxes[pred_labels == label]

            # 对ground truth重复上述操作
            all_annotations.append([np.array([]) for _ in range(num_classes)])  # num_class个空数组
            if any(annotations[:, -1] > 0):
                annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()
                _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

                # Reformat to x1, y1, x2, y2 and rescale to image dimensions
                annotation_boxes = np.empty_like(_annotation_boxes)
                annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
                annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
                annotation_boxes *= opt.img_size

                for label in range(num_classes):
                    all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

    # 计算每个类的AP
    average_precisions = {}
    for label in range(num_classes):
        true_positives = [] # 正例数量
        scores = []
        num_annotations = 0

        for i in tqdm.tqdm(range(len(all_annotations)), desc=f"Computing AP for class '{label}'"):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]

            num_annotations += annotations.shape[0] # 该类标注数目
            detected_annotations = []

            for *bbox, score in detections:
                scores.append(score)

                if annotations.shape[0] == 0:   # 该图像无目标则继续循环
                    true_positives.append(0)
                    continue

                overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)    # 计算预测box与标注box的IOU，则得分
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= opt.iou_thres and assigned_annotation not in detected_annotations:
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    true_positives.append(0)

        # no annotations -> AP for this class is 0
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        true_positives = np.array(true_positives)
        false_positives = np.ones_like(true_positives) - true_positives
        # sort by score
        indices = np.argsort(-np.array(scores)) # 大到小排序
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)    # 计算负正例数
        true_positives = np.cumsum(true_positives)      # 计算真正例数

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)   # 计算ap
        average_precisions[label] = average_precision

    logger.info("Average Precisions:")
    for c, ap in average_precisions.items():
        logger.info(f"+ Class '{c}' - AP: {ap}")
    mAP = np.mean(list(average_precisions.values()))
    logger.info(f"mAP: {mAP}")
