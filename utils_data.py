import os, sys, random, json, glob, cv2, shutil
from pathlib import Path
from exception import CustomException
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import load_coco_json
from logger import logging


MASK_EXT = "png"
IMAGE_EXT = "png"


# ________________ MAKE PATHS ________________ #
def make_data_path(root_name):
    """
    Constructs and returns the paths for the training, testing, and validation datasets, given a root directory name.

    Parameters:
    - root_name (str): The name of the root directory under which the datasets reside.

    Returns:
    - train_path (pathlib.Path): Path object pointing to the training dataset directory.
    - val_path (pathlib.Path): Path object pointing to the validation dataset directory.
    - test_path (pathlib.Path): Path object pointing to the testing dataset directory.

    """

    try:
        current_path = os.getcwd()

        train_path = Path(
            os.path.join(
                (os.path.dirname(os.path.dirname(current_path))),
                root_name,
                "input",
                "data",
                "train",
            )
        )

        val_path = Path(
            os.path.join(
                (os.path.dirname(os.path.dirname(current_path))),
                root_name,
                "input",
                "data",
                "val",
            )
        )

        test_path = Path(
            os.path.join(
                (os.path.dirname(os.path.dirname(current_path))),
                root_name,
                "input",
                "data",
                "test",
            )
        )

        return train_path, val_path, test_path

    except Exception as e:
        raise CustomException(e, sys)


# ________________ VISUALIZE IMAGES ________________ #
def visualize_frames(image, title, number_frames=12, layout=(3, 4), figsize=(12, 12)):
    """
    Display specified number of frames from a multi-frame image. The selected frames to display are randomly sampled without replacement from the images

    Parameters:
    - image: The multi-frame image.
    - number_frames: The number of frames to display.
    - layout: A tuple (nrows, ncols) specifying the layout of the frames.
    - figsize: A tuple specifying the figure dimensions.
    - title: The title of the plot.

    Return:
    - fig as a plot object

    """

    try:
        # randomly select frame indices
        all_frames = list(range(image.n_frames))
        selected_frames = np.random.choice(
            all_frames, size=number_frames, replace=False
        )

        nrows, ncols = layout
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

        for index, frame_index in enumerate(selected_frames):
            image.seek(frame_index)
            ax = axs[index // ncols, index % ncols]

            # convert the image to a numpy array
            image_array = np.array(image)
            # normalize the pixel values to fit within the 0-255 range
            normalized_image_array = (
                (image_array - image_array.min())
                / (image_array.max() - image_array.min())
                * 255
            ).astype(np.uint8)

            ax.imshow(normalized_image_array, cmap="gray")
            ax.axis("off")
            ax.set_title(f"Frame {frame_index+1}")

        # hide any remaining unused subplots
        for index in range(number_frames, nrows * ncols):
            axs[index // ncols, index % ncols].axis("off")

        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.05)
        # plt.show()
        return fig

    except Exception as e:
        raise CustomException(e, sys)


# ________________ CONVERT TIFF FRAMES TO .PNG IMAGES ________________ #
def save_image_from_tiff(image, output_path, pre_fix):
    """
    Given a multi-layer TIFF image, this function will iterate through each frame and save a separate .png image for each frame.

    Parameters:
    - image: The multi-layer TIFF image.
    - file_name: The file name for saving the output image.
    - output_path: The path for saving the output image

    Returns
    - saved .png image

    """

    try:
        total_digits = len(str(image.n_frames))

        for frame_index in range(image.n_frames):
            image.seek(frame_index)

            image_array = np.array(image)
            normalized_image_array = (
                (image_array - image_array.min())
                / (image_array.max() - image_array.min())
                * 255
            ).astype(np.uint8)

            converted_image = Image.fromarray(normalized_image_array)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            file_name = (
                f"{pre_fix}_{str(frame_index + 1).zfill(total_digits)}.{IMAGE_EXT}"
            )

            converted_image.save(os.path.join(output_path, file_name))

    except Exception as e:
        raise CustomException(e, sys)


# ________________ SAVE THE PLOTS ________________ #
def save_plot(fig, file_name, file_path):
    """
    Save the given figure to a .jpg file.

    Parameters:
    - fig: The figure object to be saved.
    - file_name (str, optional): The name of the output file".
    - file_path (str, optional): The path where the output file should be saved".

    """

    try:
        # ensure the directory structure exists
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        full_path = os.path.join(file_path, file_name)

        fig.savefig(full_path, dpi=300, bbox_inches="tight", format="png")

    except Exception as e:
        raise CustomException(e, sys)


# ________________ SAVE THE DF ________________ #
def save_CSV(df, path_to_save_DF, file_name_DF):
    try:
        full_path_to_save_DF = os.path.join(path_to_save_DF, file_name_DF)
        df.to_csv(full_path_to_save_DF, index=False)

    except Exception as e:
        raise CustomException(e, sys)


# ________________ MAKE OUT FOLDERS ________________ #
def make_out_folder(output_path, classes):
    """
    Create output folders for each class.

    Parameters:
    - output_path (str): The base directory for output folders.
    - classes (dict): Dictionary of class keys and names.

    """

    try:
        for key in [x for x in classes.keys() if x != 0]:
            out_folder_path = os.path.join(output_path, str(key))

            if not os.path.exists(out_folder_path):
                os.makedirs(out_folder_path)
    except Exception as e:
        raise CustomException(e, sys)


# ________________ CONVERT MULTINARY MASK TO IMAGE AND SAVE TO FOLDER ________________ #
def save_image_from_mask(image, output_path, pre_fix):
    """
    Extract multinary masks from the image object and convert them into .png images for saving in corresponding folders

    Parameters:
    - image: The mask object that the function take in for the process
    - output_path: The expected path for output images extracted from the mask

    """

    try:
        total_digits = len(str(image.n_frames))

        for frame_index in range(image.n_frames):
            # seek the first frame
            image.seek(frame_index)

            # convert the first frame to a NumPy array
            image_array = np.array(image)

            unique_classes = np.unique(image_array)

            # loop through each unique class
            for class_item in [
                x for x in unique_classes if x != 0
            ]:  # to not include the backgound in the needed classes
                # create a binary mask for the current class
                mask = np.where(image_array == class_item, 255, 0).astype(np.uint8)

                # convert the binary mask to an Image object
                mask_image = Image.fromarray(mask)

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                # define the output folder based on the class value
                class_folder = os.path.join(output_path, str(class_item))

                file_name = (
                    f"{pre_fix}_{str(frame_index + 1).zfill(total_digits)}.{IMAGE_EXT}"
                )

                # save the mask to the appropriate folder
                mask_image.save(os.path.join(class_folder, file_name))

    except Exception as e:
        raise CustomException(e, sys)


# ________________ RENAME MASK FOLDERS ________________ #
def rename_folder(path_to_masks, classes):
    """
    Rename the folders by using matching names with keys in classes

    Parameters:
    - path_to_masks: Path to the masks folder
    - classes: Classes containing information about keys and categories

    """

    try:
        # List all directories inside train_masks
        dir_names = sorted(
            [
                dir_name
                for dir_name in os.listdir(path_to_masks)
                if os.path.isdir(os.path.join(path_to_masks, dir_name))
            ]
        )
        # Iterate and rename
        for dir_name in dir_names:
            # Check if dir_name can be converted to an integer
            try:
                int_dir_name = int(dir_name)
            except ValueError:
                continue  # Skip this directory if it can't be converted to an integer

            if int_dir_name in classes.keys():
                old_path = os.path.join(path_to_masks, dir_name)
                new_path = os.path.join(path_to_masks, classes[int_dir_name])

                # Check if new_path already exists
                if os.path.exists(new_path):
                    # Remove the directory and its contents
                    shutil.rmtree(new_path)

                # Rename the directory
                os.rename(old_path, new_path)

    except Exception as e:
        raise CustomException(e, sys)


# ________________ VISUALIZE IMAGES WITH ANNOTATIONS ________________ #
def visualize_image_with_annotation(
    train_dataset_dicts, train_metadata, image_id, n_cols=2, figsize=(10, 16)
):
    """
    Visualize annotations of a specified image in separate subplots for each category.

    Parameters:
    - train_dataset_dicts: The train data dict acquired by DatasetCatalog.get("train")
    - train_metadata: The metadata of train acquired by MetadataCatalog.get("train")
    - image_id: The ID of the image to be visualized.
    - n_cols: Number of columns in the layout
    - figsize: Figure size

    Return:
    - fig

    """
    # Find the specified image by image_id
    single_image = next(
        (d for d in train_dataset_dicts if d["image_id"] == image_id), None
    )
    if not single_image:
        raise ValueError(f"Image with ID {image_id} not found in the dataset.")

    # Extract all unique category IDs from the annotations for this image
    category_ids = list(
        set([anno["category_id"] for anno in single_image["annotations"]])
    )

    # Create a mapping from category_id to category_name based on the actual category IDs
    category_names = train_metadata.get("thing_classes")
    cat_id_to_name = {cat_id: category_names[cat_id] for cat_id in category_ids}

    # Create samples for each category for the single image
    samples = []
    for cat_id in category_ids:
        image_annotations = [
            a for a in single_image["annotations"] if a["category_id"] == cat_id
        ]
        d_filtered = single_image.copy()
        d_filtered["annotations"] = image_annotations
        samples.append(d_filtered)

    n_rows = len(category_ids) // n_cols + (len(category_ids) % n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(
        f"Visualization of Image with Annotations for image ID: {image_id}",
        fontsize=16,
        y=1.00,
    )

    if n_rows == 1 and n_cols == 1:
        # Handle the case when there's only one category and a layout of 1 image
        img = cv2.imread(samples[0]["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(samples[0])
        axs.imshow(vis.get_image()[:, :, ::-1])
        axs.set_title(cat_id_to_name[samples[0]["annotations"][0]["category_id"]])
        axs.axis("off")
    else:
        for index, d in enumerate(samples):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
            vis = visualizer.draw_dataset_dict(d)

            row = index // n_cols
            col = index % n_cols

            axs[row, col].imshow(vis.get_image()[:, :, ::-1])
            axs[row, col].set_title(cat_id_to_name[d["annotations"][0]["category_id"]])
            axs[row, col].axis("off")

    plt.tight_layout()
    return fig


# ________________ COMBINE THE DATA HANDLING FUNCTIONS TO A CLASS ________________ #
class DataHandler:
    def __init__(self, data_type, path, classes):
        assert data_type in ["train", "val", "test"], "Invalid data type provided!"
        self.data_type = data_type
        self.path = path
        self.classes = classes
        self._handle_data()

    def _handle_data(self):
        self._handle_images()
        self._handle_masks()

    def _handle_images(self):
        # Load the tiff file
        file_name = f"{self.data_type}-images.tif"
        image = Image.open(os.path.join(self.path, file_name))
        path_to_save_images = os.path.join(
            "input", "data", self.data_type, f"{self.data_type}_images"
        )

        # Save images from the tiff
        save_image_from_tiff(image, path_to_save_images, pre_fix=self.data_type)
        logging.info(
            f"{self.data_type} images were extracted and saved to {path_to_save_images}"
        )

    def _handle_masks(self):
        # Make the folders ready for storing mask images
        path_to_save_mask_images = os.path.join(
            "input", "data", self.data_type, f"{self.data_type}_masks"
        )
        make_out_folder(path_to_save_mask_images, self.classes)
        logging.info(f"output folder for {self.data_type} mask images was made ready")

        # Load the mask labels tiff
        file_name = f"{self.data_type}-groundtruth.tif"
        image = Image.open(os.path.join(self.path, file_name))

        # Save mask images from the tiff
        save_image_from_mask(image, path_to_save_mask_images, pre_fix=self.data_type)
        logging.info(
            f"{self.data_type} mask images were extracted and stored in {path_to_save_mask_images}"
        )


# ________________ ANNOTATION HANDLING PART ________________ #
MASK_EXT = "png"
ORIGINAL_EXT = "png"
image_id = 0
annotation_id = 0


# ________________ EXTRACT THE ANNOTATION INFO FROM MASK ________________ #
def images_annotations_info(mask_path, category_ids):
    """
    This function is to process the binary masks and generate images and annotations information.

    Parameters:
    - mask_path: Path to the directory containing binary masks

    Return:
    - Tuple containing images info, annotations info, and annotation count

    """
    global image_id, annotation_id
    annotations = []
    images = []

    try:
        # Iterate through categories and corresponding masks
        for category in category_ids.keys():
            for mask_image in sorted(
                glob.glob(os.path.join(mask_path, category, f"*.{MASK_EXT}"))
            ):
                original_file_name = (
                    f'{os.path.basename(mask_image).split(".")[0]}.{ORIGINAL_EXT}'
                )

                image_number = int(original_file_name.split("_")[-1].split(".")[0])
                mask_image_open = cv2.imread(mask_image)

                # Get image dimensions
                height, width, _ = mask_image_open.shape

                # Create or find existing image annotation
                if original_file_name not in map(lambda img: img["file_name"], images):
                    image = {
                        "id": image_number,
                        "width": width,
                        "height": height,
                        "file_name": original_file_name,
                    }
                    images.append(image)
                    # image_id += 1
                else:
                    image = [
                        element
                        for element in images
                        if element["file_name"] == original_file_name
                    ][0]

                # Find contours in the mask image
                gray = cv2.cvtColor(mask_image_open, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                contours = cv2.findContours(
                    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )[0]

                # Create annotation for each contour
                for contour in contours:
                    bbox = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)
                    segmentation = contour.flatten().tolist()

                    annotation = {
                        "iscrowd": 0,
                        "id": annotation_id,
                        "image_id": image["id"],
                        "category_id": category_ids[category],
                        "bbox": bbox,
                        "area": area,
                        "segmentation": [segmentation],
                    }

                    # Add annotation if area is greater than zero
                    if area > 0:
                        annotations.append(annotation)
                        annotation_id += 1

        return images, annotations, annotation_id

    except Exception as e:
        raise CustomException(e, sys)


# ________________ PROCESS AND SAVE ANNOTATIONS TO COCO JSON ________________ #
def process_masks(mask_path, dest_json, category_ids):
    """
    Get annotation information from the images_annotations_info() and save the annotation to coco JSON format

    Parameters:
    - mask_path: Path to where the mask located
    - dest_json: Path to where to save the COCO JSON
    - category_ids: The category ids from the images_annotations_info()

    Returns:
    - save COCO JSON to expected location

    """

    global image_id, annotation_id
    image_id = 0
    annotation_id = 0

    try:
        # Initialize the COCO JSON format with categories
        coco_format = {
            "info": {},
            "licenses": [],
            "images": [],
            "categories": [
                {"id": value, "name": key, "supercategory": key}
                for key, value in category_ids.items()
            ],
            "annotations": [],
        }

        # Create images and annotations sections
        (
            coco_format["images"],
            coco_format["annotations"],
            annotation_cnt,
        ) = images_annotations_info(mask_path, category_ids)

        # Save the COCO JSON to a file
        with open(dest_json, "w") as file:
            json.dump(coco_format, file, sort_keys=True, indent=4)

        print(
            "Created %d annotations for images in folder: %s"
            % (annotation_cnt, mask_path)
        )

    except Exception as e:
        raise CustomException(e, sys)


# ________________ COMBINE THE ANNOTATION HANDLING FUNCTIONS TO A CLASS ________________ #
class AnnotationHandler:

    """
    Exactract contour lines from mask images, convert the contour lines to annotations and save the annotation to a COCO JSON file

    """

    def __init__(self, dataset_type, path_to_masks, classes, category_ids):
        """
        Initialize the AnnotationHandler.

        Parameters:
        - dataset_type: A string, one of ["train", "val", "test"]
        - path_to_masks: Path to the masks folder
        - classes: Classes containing information about keys and categories
        - category_ids: Dictionary mapping category names to IDs
        """
        self.dataset_type = dataset_type
        self.path_to_masks = path_to_masks
        self.classes = classes
        self.category_ids = category_ids
        self._rename_folders()
        self._process_masks()

    def _rename_folders(self):
        """
        Rename folders from "0", ..., "6" to class names for ease of handling.
        """
        rename_folder(self.path_to_masks, self.classes)

    def _process_masks(self):
        """
        Process masks and generate COCO JSON annotations.
        """
        path_to_JSON = os.path.join(
            "input",
            "data",
            self.dataset_type,
            self.dataset_type + "_images",
            self.dataset_type + ".json",
        )
        process_masks(self.path_to_masks, path_to_JSON, self.category_ids)
        logging.info(
            f"Annotations for {self.dataset_type} were extracted and stored in {path_to_JSON}"
        )


# ________________ LOAD A JSON FILE ________________ #
def load_JSON(file_path, file_name):
    try:
        full_path = os.path.join(file_path, file_name)
        with open(full_path, "r") as file:
            data = json.load(file)

        return data
    except Exception as e:
        raise CustomException(e, sys)


# ________________ SAVE A JSON FILE ________________ #
def save_JSON(data, file_path, file_name):
    """ """
    try:
        full_path = os.path.join(file_path, file_name)
        with open(full_path, "w") as file:
            json.dump(data, file)

    except Exception as e:
        raise CustomException(e, sys)


# ________________ FILTER AND SAVE A JSON FILE (WHEN NEEDED) ________________ #
def filter_and_save_annotation(data, min_area, path_to_save_JSON):
    """ """
    try:
        # Filter out annotations with areas less than the specified threshold
        filtered_annotations = [
            annotation
            for annotation in data["annotations"]
            if annotation["area"] >= min_area
        ]

        # Update the "annotations" key in the data with the filtered annotations
        data["annotations"] = filtered_annotations

        # Save the updated data to the specified JSON file
        with open(path_to_save_JSON, "w") as file:
            json.dump(data, file)

    except Exception as e:
        raise CustomException(e, sys)


# ________________ MAKE GROUND TRUTH ________________ #
def make_annotated_images(
    path_to_images, path_to_save_annotated_images, metadata, dataset_dicts
):
    """
    Visualize annotations on test images and save the annotated images to a specified directory.

    Parameters:
    - path_to_test_images: Path to the directory containing test images.
    - path_to_save_annotated_images: Path to the directory where annotated images will be saved.
    - test_metadata: Metadata for the test dataset.
    - test_dataset_dicts: Dictionary containing annotations for the test images.

    Return:
    - None
    """

    try:
        if not os.path.exists(path_to_save_annotated_images):
            os.makedirs(path_to_save_annotated_images)

        # Loop over the images in the input folder
        for image_name in os.listdir(path_to_images):
            if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            single_image = next(
                (
                    d
                    for d in dataset_dicts
                    if os.path.basename(d["file_name"]) == image_name
                ),
                None,
            )

            # Read the image
            path_to_image = os.path.join(path_to_images, image_name)
            image = cv2.imread(path_to_image)

            # Create a Visualizer object
            visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1)

            # Draw the annotations on the image
            visual = visualizer.draw_dataset_dict(single_image)

            # Define the path to save the annotated image
            full_path_to_save_annotated_images = os.path.join(
                path_to_save_annotated_images, image_name
            )
            # Save the annotated image to the specified output path
            cv2.imwrite(
                full_path_to_save_annotated_images, visual.get_image()[:, :, ::-1]
            )

    except Exception as e:
        raise CustomException(e, sys)


def make_annotation_df(path_to_annotation_JSON, annotation_JSON_name):
    """
    Extract information from annotation COCO JSON file to make a DataFrame

    Parameters:
    - path_to_annotation_JSON: Path to the directory containing annotation COCO JSON file.
    - annotation_JSON_name: Name of the annotation COCO JSON file.
    Return:
    - DataFrame.

    """

    try:
        annotation_JSON = load_JSON(path_to_annotation_JSON, annotation_JSON_name)

        # Extract info from annotation_JSON
        annotations = annotation_JSON["annotations"]
        categories = {
            category["id"]: category["name"]
            for category in annotation_JSON["categories"]
        }
        images = {
            image["id"]: image["file_name"] for image in annotation_JSON["images"]
        }

        # Preprocess annotation_JSON
        records = []
        for annotation in annotations:
            image_id = annotation["image_id"]
            category_id = annotation["category_id"]
            bbox = annotation["bbox"]
            area = annotation["area"]
            segmentation = annotation["segmentation"][0]

            # Calculate centroid from segmentation annotation_JSON
            xs = segmentation[0::2]
            ys = segmentation[1::2]
            centroid_x = sum(xs) / len(xs)
            centroid_y = sum(ys) / len(ys)
            centroid = (centroid_x, centroid_y)

            # Add record to the list
            records.append(
                {
                    "File Name": images[image_id],
                    "Image ID": image_id,
                    "Class Name": categories[category_id],
                    "Object Number": annotation["id"],
                    "Area": area,
                    "Centroid": centroid,
                    "BoundingBox": bbox,
                }
            )

        # Create annotation dataframe
        df = pd.DataFrame(records)

        return df

    except Exception as e:
        raise CustomException(e, sys)


# ________________ MEASURE EXECUTION TIME ________________ #
def measure_execution_time(func, *args, **kwargs):
    from time import time

    start_time = time()
    func(*args, **kwargs)
    end_time = time()
    estimate = round(end_time - start_time, 2)

    return estimate
