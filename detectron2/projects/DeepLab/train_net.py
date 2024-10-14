#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
DeepLab Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
from detectron2.data.datasets.coco import load_sem_seg
from detectron2.data import DatasetCatalog, MetadataCatalog
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, DatasetMapper, MetadataCatalog
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    DefaultTrainer,
    launch,
)
from detectron2.evaluation import (
    CityscapesSemSegEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "sem_seg":
            return SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        if evaluator_type == "cityscapes_sem_seg":
            return CityscapesSemSegEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(
                cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg)
            )
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    #     # CholecSeg8k
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
    # metadata_train = {}
    # sem_seg_root_train = "../../../../data/CholecSeg8k/processed_scene_padding/labels/train"
    # image_root_train = "../../../../data/CholecSeg8k/processed_scene_padding/images/train"
    # semantic_name_train = "Chole_stuffonly_train"
    # DatasetCatalog.register(semantic_name_train, lambda: load_sem_seg(sem_seg_root_train, image_root_train, gt_ext="png", image_ext="png"))
    # MetadataCatalog.get(semantic_name_train).set(sem_seg_root=sem_seg_root_train,evaluator_type="sem_seg",ignore_label=255,**metadata_train)
    # MetadataCatalog.get(semantic_name_train).stuff_classes = class_name_mapping

    # metadata_test = {}
    # sem_seg_root_test = "../../../../data/CholecSeg8k/processed_scene_padding/labels/test"
    # image_root_test = "../../../../data/CholecSeg8k/processed_scene_padding/images/test"
    # semantic_name_test = "Chole_stuffonly_test"
    # DatasetCatalog.register(semantic_name_test, lambda: load_sem_seg(sem_seg_root_test, image_root_test, gt_ext="png", image_ext="png"))
    # MetadataCatalog.get(semantic_name_test).set(sem_seg_root=sem_seg_root_test,image_root=image_root_test,evaluator_type="sem_seg",ignore_label=255,**metadata_test)
    # MetadataCatalog.get(semantic_name_test).stuff_classes = class_name_mapping

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
    metadata_train = {}
    sem_seg_root_train = "../../../../data/Endovision18/processed_scene/labels/train"
    image_root_train = "../../../../data/Endovision18/processed_scene/images/train"
    semantic_name_train = "Chole_stuffonly_train"
    DatasetCatalog.register(semantic_name_train, lambda: load_sem_seg(sem_seg_root_train, image_root_train, gt_ext="png", image_ext="png"))
    MetadataCatalog.get(semantic_name_train).set(sem_seg_root=sem_seg_root_train,evaluator_type="sem_seg",ignore_label=255,**metadata_train)
    MetadataCatalog.get(semantic_name_train).stuff_classes = class_name_mapping

    metadata_test = {}
    sem_seg_root_test = "../../../../data/Endovision18/processed_scene/seq/Test_Data/labels/test/seq_1"
    image_root_test = "../../../../data/Endovision18/processed_scene/seq/Test_Data/images/test/seq_1"
    semantic_name_test = "Chole_stuffonly_test"
    DatasetCatalog.register(semantic_name_test, lambda: load_sem_seg(sem_seg_root_test, image_root_test, gt_ext="png", image_ext="png"))
    MetadataCatalog.get(semantic_name_test).set(sem_seg_root=sem_seg_root_test,image_root=image_root_test,evaluator_type="sem_seg",ignore_label=255,**metadata_test)
    MetadataCatalog.get(semantic_name_test).stuff_classes = class_name_mapping

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
