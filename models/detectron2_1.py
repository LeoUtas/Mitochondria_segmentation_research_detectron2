import sys, os
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
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


start_time = time()


# ****** --------- ****** #
test_name = "test20"
note = "train-test reversed images\nrandom init weights"
# ****** --------- ****** #


model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
# model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
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
solver_steps = [100]
number_classes = get_num_classes(str(train_metadata))


cfg = get_cfg()
cfg.OUTPUT_DIR = os.path.join(script_path, test_name)
cfg.merge_from_file(model_zoo.get_config_file(model))

# cfg.MODEL.DEVICE = "cpu"

cfg.DATASETS.TRAIN = "train"
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = num_workers
cfg.MODEL.WEIGHTS = ""
cfg.SOLVER.IMS_PER_BATCH = ims_per_batch  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = learning_rate  # pick a good LR
cfg.SOLVER.MAX_ITER = (
    max_iterations  # 1000 iterations seems good enough for this dataset
)
cfg.SOLVER.STEPS = solver_steps
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128  # Default is 512, using 256 for this dataset.
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = number_classes

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(
    cfg
)  # Create an instance of of DefaultTrainer with the given congiguration
trainer.resume_or_load(
    resume=False
)  # Load a pretrained model if available (resume training) or start training from scratch if no pretrained model is available


trainer.train()  # Start the training process


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
    "SOLVER.STEPS": solver_steps,
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
