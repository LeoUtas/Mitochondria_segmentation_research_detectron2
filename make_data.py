import sys, os, json

sys.path.append("../")  # add parent directory to the system path
from utils_data import *
from utils_model import *
from logger import logging


root_name = os.path.join("Mito_segmentation", "Research_detectron2")
train_path, _, test_path = make_data_path(root_name)
logging.info("train_path, _, test_path were configured")
# -------------------------------------------------------------------------------- #


def make_data():
    # load classes JSON
    classes_path = os.path.join("input", "data", "classes.json")
    with open(classes_path, "r") as file:
        classes = json.load(file)

    classes = {int(key): value for key, value in classes.items()}
    logging.info(f"classes dict was loaded")
    # -------------------------------------------------------------------------------- #

    # ________________ HANDLE DATA ________________ #
    DataHandler("train", train_path, classes)
    DataHandler("test", test_path, classes)
    logging.info(
        f"train and test images were extracted and saved to {train_path}, {test_path}"
    )

    # ________________ HANDLE ANNOTATIONS ________________ #
    filtered_classes = {key: value for key, value in classes.items() if key not in [0]}
    print(filtered_classes)

    filtered_classes = {
        idx + 1: value for idx, (_, value) in enumerate(filtered_classes.items())
    }
    print(filtered_classes)

    # Create category_ids by swapping keys and values
    category_ids = {value: key for key, value in filtered_classes.items()}
    print(category_ids)

    path_to_train_masks = os.path.join("input", "data", "train", "train_masks")
    path_to_test_masks = os.path.join("input", "data", "test", "test_masks")

    AnnotationHandler("train", path_to_train_masks, classes, category_ids)
    AnnotationHandler("test", path_to_test_masks, classes, category_ids)
    # -------------------------------------------------------------------------------- #


if __name__ == "__main__":
    make_data()
