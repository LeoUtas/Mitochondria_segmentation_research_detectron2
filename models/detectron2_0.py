import sys, os
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer, DefaultPredictor
from time import time

setup_logger()

# ________________ HANDLE THE PATH THING ________________ #
# get the absolute path of the script's directory
script_path = os.path.dirname(os.path.abspath(__file__))
# get the parent directory of the script's directory
parent_path = os.path.dirname(script_path)
sys.path.append(parent_path)

from utils_data import *
from utils_model import *
from utils_model_inference import *

# from logger import logging


# ________________ MAKE TRAIN DATA READY FOR TRAINING ________________ #
train_path, _, test_path = make_data_path(parent_path)

path_to_train_JSON = os.path.join(train_path, "train_images", "train.json")
path_to_train_images = os.path.join(train_path, "train_images")
register_coco_instances("train", {}, path_to_train_JSON, path_to_train_images)
train_metadata = MetadataCatalog.get("train")
train_dataset_dicts = DatasetCatalog.get("train")

path_to_test_JSON = os.path.join(test_path, "test_images", "test.json")
path_to_test_images = os.path.join(test_path, "test_images")
register_coco_instances("test", {}, path_to_test_JSON, path_to_test_images)
test_metadata = MetadataCatalog.get("test")
test_dataset_dicts = DatasetCatalog.get("test")


# # ________________ VISUALIZE THE ANNOTATED TRAIN DATA ________________ #
# annotated_train_visualization = visualize_image_with_annotation(
#     train_dataset_dicts, train_metadata
# )
# file_name = "annotated_train_image.png"
# path_to_save_visualization = os.path.join(parent_path, "input", "viz")
# save_plot(annotated_train_visualization, file_name, path_to_save_visualization)


start_time = time()


# ****** --------- ****** #
test_name = "test18"
note = "train-test reversed images\n"
# ****** --------- ****** #


# model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
# model = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

learning_rate = 0.001
max_iterations = 2000
# extract the number of classes from train_metadata
number_classes = get_num_classes(str(train_metadata))
number_classes = number_classes
batch_size_per_image = 128  # Default is 512, using 256 or 128 for this dataset.
score_thresh = 0.5  # for evaluation
train_dataset_name = "train"
# val_dataset_name = "val"
num_workers = 2
ims_per_batch = 2


# ________________ CONFIGURE THE DETECTRON2 READY FOR TRAINING ________________ #
configurator = ModelConfiguration(
    parent_path,
    test_name,
    model,
    learning_rate,
    max_iterations,
    number_classes,
    batch_size_per_image,
    train_dataset_name,
    # val_dataset_name,
    num_workers,
    ims_per_batch,
)
configurator.setup()
cfg = configurator.get_config()

# # ________________ custom trainer for augmentation if needed ________________ #
# # Config for data augmentation
# cfg.MODEL.AUGS = configurator.augs
# trainer = CustomTrainer(cfg)

# ________________ trainer ________________ #
trainer = DefaultTrainer(cfg)


# ________________ hook for train and val losses if needed ________________ #
# # Create a custom validation loss object
# val_loss = ValidationLoss(cfg)

# # Register the custom validation loss object as a hook to the trainer
# trainer.register_hooks([val_loss])

# # Swap the positions of the evaluation and checkpointing hooks so that the validation loss is logged correctly
# trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

# cfg.SOLVER.STEPS = [1000]

trainer.resume_or_load(resume=False)
trainer.train()


execution_time = time() - start_time


# ________________ MAKE COCO METRICS EVALUATION ________________ #
# Inference should use the config with parameters that are used in training
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("test", output_dir=os.path.join(cfg.OUTPUT_DIR))
test_loader = build_detection_test_loader(cfg, "test")
COCOmetrics = inference_on_dataset(predictor.model, test_loader, evaluator)
COCOmetrics["info"] = {
    "model": model,
    "test_name": test_name,
    "learning_rate": learning_rate,
    "max_iterations": max_iterations,
    "batch_size_per_image": batch_size_per_image,
    "number_worker": num_workers,
    "images_per_batch": ims_per_batch,
    "score_thresh": score_thresh,
    # "SOLVER.STEPS": "[1000]",
    "note": note,
}


# ________________ SAVE COCO METRICS TO JSON ________________ #
file_path = os.path.join(parent_path, "output", "data")
file_name = f"{test_name}_COCOmetrics.json"
save_JSON(COCOmetrics, file_path, file_name)


# ________________ MAKE COCO METRICS TABLE AND SAVE TO .PNG ________________ #
file_path = os.path.join(parent_path, "output", "viz")
file_name = f"{test_name}_test0_COCOmetrics.png"
generate_COCOtable(COCOmetrics, file_path, file_name)


# ________________ SAVE LOSS METRICS TO JSON ________________ #
path_to_metrics = os.path.join(parent_path, "models", test_name)
path_to_save_JSON = os.path.join(parent_path, "output", "data")
record_loss_to_JSON(path_to_metrics, path_to_save_JSON, test_name, execution_time)


# ________________ VISUALIZE LOSS METRICS ________________ #
file_path = os.path.join(parent_path, "output", "data")
file_name = f"{test_name}_loss_metrics.json"
loss_metrics = load_JSON(file_path, file_name)
figure = visualize_losses(loss_metrics, figsize=(8, 6))


# ________________ SAVE LOSS METRICS VISUALIZATION TO PNG ________________ #
file_path = os.path.join(parent_path, "output", "viz")
file_name = f"{test_name}_loss_plot.png"
save_plot(figure, file_name, file_path)
