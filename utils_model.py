import os, sys, torch, json, re, cv2, leafmap, csv
import detectron2.utils.comm as comm
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import (
    DatasetMapper,
    build_detection_train_loader,
    transforms as T,
)
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from skimage.measure import regionprops, label
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from exception import CustomException
from logger import logging


# ________________ GET NUMBER OF CLASSES FROM METADATA ________________ #
def get_num_classes(metadata_string: str) -> int:
    """
    Extracts the thing_classes from the metadata string and returns the number of classes.

    Parameters:
    - metadata_string: Metadata in string format

    return:
    - num_classes: Number of classes that will go into the detectron2 model

    """

    try:
        # Extract the substring for thing_classes using regular expressions
        thing_classes_match = re.search(r"thing_classes=\[([^\]]+)\]", metadata_string)

        if thing_classes_match:
            thing_classes_str = thing_classes_match.group(1)
            # Convert the substring to a list of classes
            thing_classes = [x.strip().strip("'") for x in thing_classes_str.split(",")]
            return len(thing_classes)
        else:
            return 0
    except Exception as e:
        raise CustomException(e, sys)


# ________________ MAKE CONFIGURATION FOR THE MODEL ________________ #
class ModelConfiguration:
    def __init__(
        self,
        parent_path,
        test_name,
        model,
        learning_rate,
        max_iterations,
        number_classes,
        batch_size_per_image,
        train_dataset_name="train",
        # val_dataset_name="val",
        num_workers=2,
        ims_per_batch=2,
    ):
        try:
            self.parent_path = parent_path
            self.test_name = test_name
            self.model = model
            self.learning_rate = learning_rate
            self.max_iterations = max_iterations
            self.number_classes = number_classes
            self.batch_size_per_image = batch_size_per_image
            self.num_workers = num_workers
            self.ims_per_batch = ims_per_batch
            # self.init_wesight = init_weight
            self.train_dataset_name = train_dataset_name
            # self.val_dataset_name = val_dataset_name
            self.cfg = get_cfg()
        except Exception as e:
            raise CustomException(e, sys)

    def setup(self):
        try:
            self.cfg.OUTPUT_DIR = os.path.join(
                self.parent_path, "models", self.test_name
            )
            if not os.path.exists(self.cfg.OUTPUT_DIR):
                os.makedirs(self.cfg.OUTPUT_DIR)

            # Add augmentation pipeline
            self.augs = [
                # T.ResizeShortestEdge(
                #     short_edge_length=(640, 672, 704, 736, 768, 800),
                #     max_size=1333,
                #     sample_style="choice",
                # ),
                # T.RandomFlip(),
                # T.RandomBrightness(intensity_min=0.5, intensity_max=1.5),
                # T.RandomRotation(
                #     angle=[90, 90]
                # # ),  # 90-degree rotation for demonstration; adjust as needed
                # T.RandomContrast(intensity_min=0.2, intensity_max=1.8),
                # T.RandomLighting(scale=0.9),
                # T.RandomSaturation(intensity_min=0.5, intensity_max=1.5),
                # T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            ]

            self.cfg.merge_from_file(model_zoo.get_config_file(self.model))

            self.cfg.DATASETS.TRAIN = (self.train_dataset_name,)
            self.cfg.DATASETS.VAL = ()
            self.cfg.DATASETS.TEST = ()
            self.cfg.DATALOADER.NUM_WORKERS = self.num_workers

            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model)
            # self.cfg.MODEL.WEIGHTS = ""
            # self.cfg.MODEL.WEIGHTS_INIT = "random"

            self.cfg.SOLVER.IMS_PER_BATCH = self.ims_per_batch
            self.cfg.SOLVER.BASE_LR = self.learning_rate
            self.cfg.SOLVER.MAX_ITER = self.max_iterations
            self.cfg.SOLVER.STEPS = []
            self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.batch_size_per_image
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.number_classes
            self.cfg.TEST.EVAL_PERIOD = 50  # Evaluate every 50 iterations

        except Exception as e:
            raise CustomException(e, sys)

    def get_config(self):
        return self.cfg


# ________________ MAKE CUSTOM TRAINER FOR USING AUGMENTATION ________________ #
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=cfg.MODEL.AUGS)
        return build_detection_train_loader(cfg, mapper=mapper)


# ________________ MAKE HOOK TO CAPTURE TRAIN AND VAL LOSSES ________________ #
class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        # Switch to the validation dataset
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        # Build the validation data loader iterator
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        try:
            # Get the next batch of data from the validation data loader
            data = next(self._loader)
            with torch.no_grad():
                # Compute the validation loss on the current batch of data
                loss_dict = self.trainer.model(data)

                # Check for invalid losses
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                # Reduce the loss across all workers
                loss_dict_reduced = {
                    "val_" + key: value.item()
                    for key, value in comm.reduce_dict(loss_dict).items()
                }
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                # Save the validation loss in the trainer storage
                if comm.is_main_process():
                    self.trainer.storage.put_scalars(
                        total_val_loss=losses_reduced, **loss_dict_reduced
                    )

        except Exception as e:
            raise CustomException(e, sys)


# ________________ RECORD TRAIN AND VAL LOSSES TO JSON ________________ #
def record_loss_to_JSON(path_to_metrics, path_to_save_JSON, test_name, execution_time):
    """
    Load JSON file for further analysis

    Parameters:
    - file_path: Path to the location where the expected file was saved
    - file_name: Name of the expected JSON file * must be a JSON file

    Return:
    - data: the loaded JSON file

    """

    try:
        full_path = os.path.join(path_to_metrics, "metrics.json")
        # Load the data from the JSON file
        with open(full_path, "r") as file:
            data = file.read()

        # Extract the required information using a robust approach
        loss_metrics = []
        for potential_json in data.split("}"):
            if potential_json.strip():
                try:
                    if not potential_json.strip():
                        continue

                    data = json.loads(potential_json + "}")
                    iteration = data.get("iteration")
                    total_loss = data.get("total_loss")
                    time = data.get("time")
                    execution_time = execution_time
                    # total_val_loss = data.get("total_val_loss")

                    if (
                        iteration is not None
                        and total_loss is not None
                        # and total_val_loss is not None
                    ):
                        loss_metrics.append(
                            {
                                "iteration": iteration,
                                "total_train_loss": total_loss,
                                # "total_val_loss": total_val_loss,
                                "train_time": time,
                                "execution_time": execution_time,
                            }
                        )
                except Exception as e:
                    raise CustomException(e, sys)

        # Save the loss_metrics to a JSON file
        with open(
            os.path.join(path_to_save_JSON, f"{test_name}_loss_metrics.json"), "w"
        ) as file:
            json.dump(loss_metrics, file)

        logging.info(f"loss_metrics was saved to {path_to_save_JSON}")
        print(f"loss_metrics was saved to {path_to_save_JSON}")

    except Exception as e:
        raise CustomException(e, sys)


# ________________ MAKE PREDICTION FOR MULTIPLE IMAGES & CSV ________________ #
def make_prediction(path_to_images, path_to_save_segm, predictor, metadata, csv_name):
    try:
        if not os.path.exists(path_to_save_segm):
            os.makedirs(path_to_save_segm)

        full_path_to_save_segm_csv = os.path.join(path_to_save_segm, csv_name)

        # Open the CSV file for writing
        with open(full_path_to_save_segm_csv, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the header row in the CSV file
            csv_writer.writerow(
                [
                    "File Name",
                    "Image ID",
                    "Class Name",
                    "Object Number",
                    "Area",
                    "Centroid",
                    "BoundingBox",
                ]
            )  # Add more columns as needed for other properties

            # Loop over the images in the input folder
            for image_file_name in os.listdir(path_to_images):
                if not image_file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                # Extract image ID from file name
                image_id_match = re.search(r"test_(\d+).png", image_file_name)
                if image_id_match:
                    image_id = int(image_id_match.group(1))
                else:
                    # Log a warning and skip this file if the image ID cannot be extracted
                    logging.warning(
                        f"Could not extract image ID from file name: {image_file_name}"
                    )
                    continue

                path_to_image = os.path.join(path_to_images, image_file_name)
                image = cv2.imread(path_to_image)

                # Perform prediction on the new image
                segm_image = predictor(image)

                # ________ MAKE csv SEGMENTATION _______ #
                # Convert the predicted mask to a binary mask
                mask = segm_image["instances"].pred_masks.to("cpu").numpy().astype(bool)

                # Get the predicted class labels
                class_labels = segm_image["instances"].pred_classes.to("cpu").numpy()

                # Use skimage.measure.regionprops to calculate object parameters
                labeled_mask = label(mask)
                props = regionprops(labeled_mask)

                # Write the object-level information to the CSV file
                for i, prop in enumerate(props):
                    object_number = i + 1  # Object number starts from 1
                    area = prop.area
                    centroid = prop.centroid
                    bounding_box = prop.bbox

                    # Check if the corresponding class label exists
                    if i < len(class_labels):
                        class_label = class_labels[i]
                        class_name = metadata.thing_classes[class_label]
                    else:
                        # If class label is not available (should not happen), use 'Unknown' as class name
                        class_name = "Unknown"

                    # Write the object-level information to the CSV file
                    csv_writer.writerow(
                        [
                            image_file_name,
                            image_id,
                            class_name,
                            object_number,
                            area,
                            centroid,
                            bounding_box,
                        ]
                    )  # Add more columns as needed for other properties

                # ________ MAKE IMAGE SEGMENTATION _______ #
                # Use `Visualizer` to draw the predictions on the image.
                image_visual = Visualizer(image[:, :, ::-1], metadata=metadata)
                visual_out = image_visual.draw_instance_predictions(
                    segm_image["instances"].to("cpu")
                )

                # Create the output file_name with _result extension
                result_file_name = os.path.splitext(image_file_name)[0] + ".png"
                path_to_save_segm_images_full = os.path.join(
                    path_to_save_segm, result_file_name
                )

                # Save the segmented image
                cv2.imwrite(
                    path_to_save_segm_images_full, visual_out.get_image()[:, :, ::-1]
                )

    except Exception as e:
        raise CustomException(e, sys)


# ________________ MAKE COMPARISON IMAGES AND SAVE TO .HTML ________________ #
# resize images before making comparison image
def resize_image(input_path, output_path, resize=0.5):
    try:
        with Image.open(input_path) as img:
            width, height = img.size
            new_size = (int(width * resize), int(height * resize))
            img = img.resize(new_size, Image.ANTIALIAS)
            img.save(output_path)

    except Exception as e:
        raise CustomException(e, sys)


# ________________ MAKE COMPARISON IMAGES AND SAVE TO .HTML ________________ #
def make_comparison_images(
    path_to_ground_truth, path_to_segm_image, path_to_save_html, resize=1.0
):
    try:
        if not os.path.exists(path_to_save_html):
            os.makedirs(path_to_save_html)

        # Loop over the images in the input folder
        for image_name in os.listdir(path_to_ground_truth):
            if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            full_path_to_ground_truth = os.path.join(path_to_ground_truth, image_name)
            full_path_to_segm_image = os.path.join(path_to_segm_image, image_name)
            image_name_0 = os.path.splitext(image_name)[0]
            full_path_to_save_html = os.path.join(path_to_save_html, image_name_0)

            if resize != 1.0:
                # Resize images
                resized_ground_truth = f"{full_path_to_ground_truth}_resized.png"
                resized_segm_image = f"{full_path_to_segm_image}_resized.png"
                resize_image(full_path_to_ground_truth, resized_ground_truth, resize)
                resize_image(full_path_to_segm_image, resized_segm_image, resize)
            else:
                resized_ground_truth = full_path_to_ground_truth
                resized_segm_image = full_path_to_segm_image

            leafmap.image_comparison(
                resized_ground_truth,
                resized_segm_image,
                label1="True labels",
                label2="Image Segmentation",
                starting_position=50,
                out_html=f"{full_path_to_save_html}.html",
            )

            # Optionally, remove resized images to save disk space
            if resize != 1.0:
                os.remove(resized_ground_truth)
                os.remove(resized_segm_image)

    except Exception as e:
        raise CustomException(e, sys) from e


# ________________ MAKE COMPARISON BAR PLOTS ________________ #
class PlotMitochondria:
    def __init__(self, path_to_segmentation_csv, path_to_ground_truth_csv, image_id=1):
        self.path_to_segmentation_csv = path_to_segmentation_csv
        self.path_to_ground_truth_csv = path_to_ground_truth_csv
        self.image_id = image_id
        self.segmentation_data, self.ground_truth_data = self.read_and_prepare_data()

    def read_and_prepare_data(self):
        try:
            segmentation_df = pd.read_csv(self.path_to_segmentation_csv)
            ground_truth_df = pd.read_csv(self.path_to_ground_truth_csv)

            segmentation_df.rename(
                columns={"area": "Area", "image ID": "Image ID"}, inplace=True
            )
            ground_truth_df.rename(
                columns={"area": "Area", "image ID": "Image ID"}, inplace=True
            )

            segmentation_data = segmentation_df[
                segmentation_df["Image ID"] == self.image_id
            ]
            ground_truth_data = ground_truth_df[
                ground_truth_df["Image ID"] == self.image_id
            ]

            return segmentation_data, ground_truth_data

        except Exception as e:
            raise CustomException(e, sys) from e

    def calculate_metrics(self):
        try:
            mean_area_segmentation = self.segmentation_data["Area"].mean()
            mean_area_ground_truth = self.ground_truth_data["Area"].mean()
            count_segmentation = self.segmentation_data.shape[0]
            count_ground_truth = self.ground_truth_data.shape[0]

            return (
                mean_area_segmentation,
                mean_area_ground_truth,
                count_segmentation,
                count_ground_truth,
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def create_plot_data(self, prediction_label, true_label):
        try:
            (
                mean_area_segmentation,
                mean_area_ground_truth,
                count_segmentation,
                count_ground_truth,
            ) = self.calculate_metrics()

            mean_area_data = pd.DataFrame(
                {
                    "Dataset": [prediction_label, true_label],
                    "Mean Area": [mean_area_segmentation, mean_area_ground_truth],
                }
            )

            mitochondria_counts = pd.DataFrame(
                {
                    "Dataset": [prediction_label, true_label],
                    "Count": [count_segmentation, count_ground_truth],
                }
            )

            return mean_area_data, mitochondria_counts

        except Exception as e:
            raise CustomException(e, sys) from e

    def add_text_annotations(self, ax, dataset_names, font_size):
        try:
            for i, p in enumerate(ax.patches):
                width = p.get_width()
                ax.text(
                    p.get_x() + width / 2,
                    p.get_y() + p.get_height() / 2,
                    dataset_names[i % len(dataset_names)],
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=font_size,
                )

        except Exception as e:
            raise CustomException(e, sys) from e

    def make_bar_plot(
        self, data, x, y, palette, xlabel, title, font_size=9, figsize=(5, 2)
    ):
        try:
            f, ax = plt.subplots(1, 1, figsize=figsize)
            sns.barplot(x=x, y=y, data=data, palette=palette, ax=ax, orient="h")
            ax.set(xlabel=xlabel, ylabel="", title=title)
            ax.yaxis.set_visible(False)
            sns.despine(left=True, bottom=True, ax=ax)
            self.add_text_annotations(ax, data[y].tolist(), font_size)
            plt.tight_layout()
            return f

        except Exception as e:
            raise CustomException(e, sys) from e

    def plot_and_save(
        self,
        image_id,
        path_to_save_bar_plots,
        prediction_label="Prediction",
        true_label="True Label",
        font_size=9,
        figsize=(5, 2),
    ):
        try:
            mean_area_data, mitochondria_counts = self.create_plot_data(
                prediction_label, true_label
            )
            bar_colors = {prediction_label: "#8B4513", true_label: "#2F4F4F"}

            formatted_image_id = str(image_id).zfill(
                3
            )  # Ensure image_id is at least 3 digits

            fig = self.make_bar_plot(
                mean_area_data,
                "Mean Area",
                "Dataset",
                bar_colors,
                "",
                "",  # plot title goes here
                font_size,
                figsize,
            )
            fig.savefig(
                os.path.join(
                    path_to_save_bar_plots, f"barplot_area_{formatted_image_id}.png"
                )
            )
            plt.close(fig)

            fig = self.make_bar_plot(
                mitochondria_counts,
                "Count",
                "Dataset",
                bar_colors,
                "",
                "",  # plot title goes here
                font_size,
                figsize,
            )
            fig.savefig(
                os.path.join(
                    path_to_save_bar_plots, f"barplot_count_{formatted_image_id}.png"
                )
            )
            plt.close(fig)

        except Exception as e:
            raise CustomException(e, sys)


# ________________ GET UNIQUE IMAGE IDS ________________ #
def get_unique_image_ids(csv_path):
    try:
        df = pd.read_csv(csv_path)
        unique_ids = df["Image ID"].unique()
        all_unique_ids = set(unique_ids) | set(unique_ids)
        return sorted(all_unique_ids)

    except Exception as e:
        raise CustomException(e, sys)
