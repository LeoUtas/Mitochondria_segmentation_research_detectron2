import sys, os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
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

# ________________ CONFIGURATION SETUP FOR PREDICTION ________________ #
cfg = get_cfg()
# model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
number_classes = get_num_classes(str(test_metadata))

# **** --------- **** #
chosen_test = "test15"
# **** --------- **** #

model_chosen_path = os.path.join(script_path, "models", chosen_test)
# Previous configuration setup...
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.MODEL.WEIGHTS = os.path.join(model_chosen_path, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = number_classes

# Create predictor
predictor = DefaultPredictor(cfg)


# ________________ MAKE SEGMENTATION IMAGES & CSV ________________ #
# Path to the input images folder
path_to_images = os.path.join(script_path, "input", "data", "test", "test_images")
# Path where the segmented images will be saved
path_to_save_segm = os.path.join(script_path, "output", "segm_images")
metadata = test_metadata
segmentation_csv_name = "segm_df.csv"

# make prediction CSV file
time = measure_execution_time(
    make_prediction,
    path_to_images,
    path_to_save_segm,
    predictor,
    metadata,
    segmentation_csv_name,
)
print(f"Time taken for make_segm_csv: {time} s")


# ________________ MAKE COMPARISON IMAGES AS HTML ________________ #
path_to_ground_truth = os.path.join(
    script_path, "input", "data", "test", "test_masks", "mitochondrion"
)
path_to_segm_image = path_to_save_segm
path_to_save_html = os.path.join(script_path, "output", "compare_images")
resize = 0.625

time = measure_execution_time(
    make_comparison_images,
    path_to_ground_truth,
    path_to_segm_image,
    path_to_save_html,
    resize,
)
print(f"Time taken for make_comparison: {time} s")


# ________________ MAKE GROUND TRUTH ANNOTATION DF AS A CSV ________________ #
path_to_annotation_JSON = os.path.join("input", "data", "test", "test_images")
annotation_JSON_name = "test.json"
ground_truth_csv_name = "annotations_df.csv"
path_to_save_annotation_DF = path_to_train_JSON
annotation_DF = make_annotation_df(path_to_annotation_JSON, annotation_JSON_name)
save_CSV(annotation_DF, path_to_annotation_JSON, ground_truth_csv_name)
print(f"annotation_DF was saved to {path_to_train_JSON}")


# ________________ MAKE COMPARE BAR PLOTS ________________ #
path_to_segmentation_csv = os.path.join(path_to_save_segm, segmentation_csv_name)
path_to_ground_truth_csv = os.path.join(path_to_annotation_JSON, ground_truth_csv_name)
path_to_save_bar_plots = os.path.join(script_path, "output", "compare_plots")
if not os.path.exists(path_to_save_bar_plots):
    os.makedirs(path_to_save_bar_plots)

unique_image_ids = get_unique_image_ids(path_to_ground_truth_csv)

for image_id in unique_image_ids:
    plotter = PlotMitochondria(
        path_to_segmentation_csv, path_to_ground_truth_csv, image_id
    )
    plotter.plot_and_save(image_id, path_to_save_bar_plots, figsize=(2.5, 1.2))
