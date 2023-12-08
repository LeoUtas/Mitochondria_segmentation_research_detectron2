import sys, os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog


# ________________ HANDLE THE PATH THING ________________ #
# get the absolute path of the script's directory
script_path = os.path.dirname(os.path.abspath(__file__))
# get the parent directory of the script's directory
parent_path = os.path.dirname(script_path)
sys.path.append(parent_path)
from utils_data import *
from utils_model import *
from logger import *


# ________________ DATA REGISTRATION FOR PREDICTION ________________ #
train_path, _, test_path = make_data_path(script_path)
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


# ________________ MAKE ANNOTATED IMAGES AND SAVE ________________ #
path_to_images = path_to_train_images
path_to_save_annotated_images = os.path.join(
    script_path, "input", "data", "train", "train_annotated_images"
)
metadata = train_metadata
dataset_dicts = train_dataset_dicts

time = measure_execution_time(
    make_annotated_images,
    path_to_images,
    path_to_save_annotated_images,
    metadata,
    dataset_dicts,
)
print(f"Time taken for make_annotated_images: {time} s")


path_to_images = path_to_test_images
path_to_save_annotated_images = os.path.join(
    script_path, "input", "data", "test", "test_annotated_images"
)
metadata = test_metadata
dataset_dicts = test_dataset_dicts

time = measure_execution_time(
    make_annotated_images,
    path_to_images,
    path_to_save_annotated_images,
    test_metadata,
    test_dataset_dicts,
)
print(f"Time taken for make_annotated_images: {time} s")
