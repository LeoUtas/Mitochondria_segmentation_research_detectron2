import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from exception import CustomException
import io, json, os, sys


def render_mpl_table(
    data,
    title,
    col_width=3.0,
    row_height=0.625,
    font_size=14,
    header_color="#40466e",
    row_colors="w",
    edge_color="#40466e",
    bbox=[0, 0, 1, 1],
    header_columns=0,
    ax=None,
    **kwargs,
):
    """Render DataFrame as a table in Matplotlib with a title and return the image."""

    try:
        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1.5])) * np.array(
                [col_width, row_height]
            )
            fig, ax = plt.subplots(figsize=size)
            ax.axis("off")

        mpl_table = ax.table(
            cellText=data.round(2).values, bbox=bbox, colLabels=data.columns, **kwargs
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)
        for k, cell in mpl_table._cells.items():
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight="bold", color="w")
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        ax.set_title(title, fontsize=16, weight="bold", pad=20)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2)
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        return img

    except Exception as e:
        raise CustomException(e, sys)


def render_info_img(COCOmetrics, col_width=4.0, **kwargs):
    try:
        """Render the 'info' section of COCOmetrics as an image table with visible lines."""
        info_data = COCOmetrics["info"].copy()
        title = info_data.pop("model", "Info")
        df_info = pd.DataFrame(list(info_data.items()), columns=["Parameter", "Value"])
        return render_mpl_table(df_info, title, col_width=col_width, **kwargs)

    except Exception as e:
        raise CustomException(e, sys)


def generate_COCOtable2(COCOmetrics, file_path, file_name):
    try:
        # Extract metrics from the loaded JSON data
        bbox_data = {
            metric: COCOmetrics["bbox"][metric]
            for metric in ["AP", "AP50", "AP75", "APs", "APm", "APl"]
        }
        segm_data = {
            metric: COCOmetrics["segm"][metric]
            for metric in ["AP", "AP50", "AP75", "APs", "APm", "APl"]
        }

        # Create dataframes
        df_bbox = pd.DataFrame([bbox_data])
        df_segm = pd.DataFrame([segm_data])

        bbox_img = render_mpl_table(df_bbox, "bbox", col_width=2.5)
        segm_img = render_mpl_table(df_segm, "segm", col_width=2.5)

        # Combine the images vertically
        final_img = Image.new(
            "RGB", (bbox_img.width, bbox_img.height + segm_img.height), (255, 255, 255)
        )
        final_img.paste(bbox_img, (0, 0))
        final_img.paste(segm_img, (0, bbox_img.height))

        # Save the combined image to a PNG file
        output_path = os.path.join(file_path, file_name)
        final_img.save(output_path)

    except Exception as e:
        raise CustomException(e, sys)


def generate_COCOtable(COCOmetrics, file_path, file_name):
    """Combine the 'info', 'bbox', and 'segm' tables and save them as a single PNG file."""

    # Render each table
    info_img = render_info_img(COCOmetrics, col_width=4.0)

    bbox_data = {
        metric: COCOmetrics["bbox"][metric]
        for metric in ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    }
    segm_data = {
        metric: COCOmetrics["segm"][metric]
        for metric in ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    }

    df_bbox = pd.DataFrame([bbox_data])
    df_segm = pd.DataFrame([segm_data])
    bbox_img = render_mpl_table(df_bbox, "bbox", col_width=2.5)
    segm_img = render_mpl_table(df_segm, "segm", col_width=2.5)

    # Combine the images vertically
    final_img = Image.new(
        "RGB",
        (
            max(info_img.width, bbox_img.width, segm_img.width),
            info_img.height + bbox_img.height + segm_img.height,
        ),
        (255, 255, 255),
    )
    final_img.paste(info_img, (0, 0))
    final_img.paste(bbox_img, (0, info_img.height))
    final_img.paste(segm_img, (0, info_img.height + bbox_img.height))

    # Save the combined image to a PNG file
    output_path = os.path.join(file_path, file_name)
    final_img.save(output_path)


# Complete code to plot and save the graph
def visualize_losses(loss_metrics, figsize=(10, 6)):
    """
    Plots and saves the training and validation losses from the provided loss metrics.

    Parameters:
    - loss_metrics: List of dictionaries containing iteration, total_train_loss, and total_val_loss.
    - save_path: Path where the plot will be saved.

    """

    try:
        # Extracting data from loss metrics
        iterations = [entry["iteration"] for entry in loss_metrics]
        train_losses = [entry["total_train_loss"] for entry in loss_metrics]
        train_time = [entry["train_time"] for entry in loss_metrics]
        total_time = sum(train_time)
        execution_time = [entry["execution_time"] for entry in loss_metrics]
        execution_time = np.mean(execution_time)
        # test_losses = [entry["total_test_loss"] for entry in loss_metrics]

        # Plotting
        plt.figure(figsize=figsize)
        plt.plot(iterations, train_losses, label="Train Loss", color="blue")
        # plt.plot(iterations, test_losses, label="test Loss", color="red")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Train losses over Iterations")

        text_position_x = iterations[-1] * 0.6
        text_position_y = max(train_losses) * 0.8
        plt.text(
            text_position_x,
            text_position_y,
            f"Execution time: {(execution_time/60):.2f} min",
            fontsize=12,
        )

        plt.legend()
        plt.grid(False)
        plt.tight_layout()

        return plt.gcf()

    except Exception as e:
        raise CustomException(e, sys)
