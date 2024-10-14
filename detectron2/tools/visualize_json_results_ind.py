#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import tqdm

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json, load_sem_seg


def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


def main() -> None:
    global args, dataset_id_map
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument(
        "--input", required=True, help="JSON file produced by the model"
    )
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument(
        "--dataset", help="name of the dataset", default="coco_2017_val"
    )
    parser.add_argument(
        "--conf-threshold", default=0.5, type=float, help="confidence threshold"
    )
    args = parser.parse_args()

    logger = setup_logger()
    register_coco_instances("my_dataset_test", {}, "../data/Endoscapes2023/processed/annotations/instances_test2017_vis.json", "../data/Endoscapes2023/processed/test2017")

    # register_coco_instances("my_dataset_test", {}, "data/CholecSeg8k/processed/out_ann_file_test_noback.json", "data/CholecSeg8k/processed/images/test")
    # register_coco_instances("my_dataset_test", {}, "data/Endovision18/out_ann_file_test_thread.json", "data/Endovision18/processed_Test_Data/images/test")
    # register_coco_instances("my_dataset_test", {}, "../data/Endovision18/processed/seq/out_ann_file_test_seq_1.json", "../data/Endovision18/processed/seq/Test_Data/images/test/seq_1")
    # register_coco_instances("my_dataset_test", {}, "data/Endovision18/out_ann_file_test_nosmall.json", "data/Endovision18/processed_Test_Data/images/test")
    # register_coco_instances("my_dataset_test", {}, "../data/CholecSeg8k/out_ann_file_train_noback.json", "../data/CholecSeg8k/processed_scene/images/train")
    # class_name_mapping=[
    #             'Black Background',
    #             'Abdominal Wall',
    #             'Liver',
    #             'Gastrointestinal Tract',
    #             'Fat',
    #             'Grasper',
    #             'Connective Tissue',
    #             'Blood',
    #             'Cystic Duct',
    #             'L-hook Electrocautery',
    #              'Gallbladder',
    #              'Hepatic Vein',
    #              'Liver Ligament']
    # metadata = {}
    # sem_seg_root = "../data/CholecSeg8k/processed_scene/labels/test"
    # image_root = "../data/CholecSeg8k/processed_scene/images/test"
    # semantic_name = "Chole_stuffonly_test"
    # DatasetCatalog.register(semantic_name, lambda: load_sem_seg(sem_seg_root, image_root, gt_ext="png", image_ext="png"))
    # MetadataCatalog.get(semantic_name).set(sem_seg_root=sem_seg_root,image_root=image_root,evaluator_type="sem_seg",ignore_label=255,**metadata)
    # MetadataCatalog.get(semantic_name).stuff_classes = class_name_mapping

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        # print(p["image_id"])
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)
    # print(dicts)
    for dic in tqdm.tqdm(dicts):
        # print(dic["file_name"])
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        back = np.zeros(img.shape)
        basename = os.path.basename(dic["file_name"])

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        vis = Visualizer(back, metadata)
        vis_pred = vis.draw_instance_predictions(predictions, alpha = 1.0,jittering=False).get_image()

        vis = Visualizer(back, metadata)
        vis_gt = vis.draw_dataset_dict(dic,1.0).get_image()

        concat = np.concatenate((img, vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])


if __name__ == "__main__":
    main()  # pragma: no cover
