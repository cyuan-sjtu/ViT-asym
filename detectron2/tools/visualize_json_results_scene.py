#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import tqdm
from pycocotools import mask as mask_utils
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json, load_sem_seg
# class_name_mapping=[
#                 'Black Background',
#                 'Abdominal Wall',
#                 'Liver',
#                 'Gastrointestinal Tract',
#                 'Fat',
#                 'Grasper',
#                 'Connective Tissue',
#                 'Blood',
#                 'Cystic Duct',
#                 'L-hook Electrocautery',
#                  'Gallbladder',
#                  'Hepatic Vein',
#                  'Liver Ligament']
class_name_mapping=[
                    'background-tissue',
                    'instrument-shaft',
                    'instrument-clasper',
                    'instrument-wrist',
                    'kidney-parenchyma',
                    'covered-kidney',
                    'thread',
                    'clamps',
                    'suturing-needle',
                    'suction-instrument',
                    'intestine',
                    'ultrasound-probe']
def create_instances(predictions, image_size):
    ret = Instances(image_size)

    chosen = np.asarray([i for i in range(0,len(predictions))])

    labels = np.asarray([class_name_mapping[predictions[i]["category_id"]] for i in chosen])

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

    # metadata = {}
    # sem_seg_root = "../data/CholecSeg8k/processed_scene_padding/labels/test"
    # image_root = "../data/CholecSeg8k/processed_scene_padding/images/test"
    # semantic_name = "Chole_stuffonly_test"
    # DatasetCatalog.register(semantic_name, lambda: load_sem_seg(sem_seg_root, image_root, gt_ext="png", image_ext="png"))
    # MetadataCatalog.get(semantic_name).set(sem_seg_root=sem_seg_root,image_root=image_root,evaluator_type="sem_seg",ignore_label=255,**metadata)
    # MetadataCatalog.get(semantic_name).stuff_classes = class_name_mapping


    metadata = {}
    sem_seg_root = "../data/Endovision18/processed_scene/labels/test"
    image_root = "../data/Endovision18/processed_scene/images/test"
    semantic_name = "Chole_stuffonly_test"
    DatasetCatalog.register(semantic_name, lambda: load_sem_seg(sem_seg_root, image_root, gt_ext="png", image_ext="png"))
    MetadataCatalog.get(semantic_name).set(sem_seg_root=sem_seg_root,image_root=image_root,evaluator_type="sem_seg",ignore_label=255,**metadata)
    MetadataCatalog.get(semantic_name).stuff_classes = class_name_mapping


    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        # print(p["image_id"])
        pred_by_image[os.path.basename(p["file_name"])[:-4]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    # if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

    #     def dataset_id_map(ds_id):
    #         return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    # elif "lvis" in args.dataset:
    #     # LVIS results are in the same format as COCO results, but have a different
    #     # mapping from dataset category id to contiguous category id in [0, #categories - 1]
    #     def dataset_id_map(ds_id):
    #         return ds_id - 1

    # else:
    #     raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)
    # print(dicts)
    for dic in tqdm.tqdm(dicts):
        # print(dic["file_name"])
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])
        seg = np.zeros(img.shape[:2])
        ann = pred_by_image[os.path.basename(dic["file_name"])[:-4]]
        for i in range(0,len(ann)):
            seg = seg + mask_utils.decode(ann[i]["segmentation"])*ann[i]["category_id"]
        seg = seg.astype(np.int64)
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_sem_seg(seg, area_threshold=0, alpha=0.5).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])


if __name__ == "__main__":
    main()  # pragma: no cover
