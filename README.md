<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li><a href="#technical-tools">Technical Tools</a></li>
    <li><a href="#data-source">Data source</a></li>
    <li><a href="#data-processing">Data processing</a></li>
    <li><a href="#the-design">The design</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#how-to-use-the-source-code">How to use the source code</a></li>
    <li><a href="#the-bottom-line">The Bottom Line</a></li>
    <li><a href="#reference">Reference</a></li>
  </ol>
</details>

### Introduction

This repository hosts the source code for an instance segmentation project, focusing on segmenting mitochondria in microscopic images. Detectron2 <a href="https://github.com/facebookresearch/detectron2/blob/main/README.md"> (Wu et al., 2019) </a> has been utilized as the primary tool for this task.

### Technical tools

-   Pytorch
-   Detectron2
-   Opencv
-   pandas
-   Docker
-   Github action

### Data source

This project employs a dataset comprising electron microscopy images from the CA1 region of the hippocampus, with annotated mitochondria by professionally trained researchers. The dataset is accessible at <a href="https://www.epfl.ch/labs/cvlab/data/data-em/"> Electron Microscopy Dataset</a>.

### Data processing

```python
save_image_from_tiff()
save_image_from_mask()
DataHandler
```

### The design

<img src="output/viz/code_structure.png" alt="" width="550">

The above diagram illustrates the fundamental architecture of this project. Below, you will find an example outline detailing the code setups necessary for experimenting with various configurations of Detectron2 for this specific task.

-   YOLOv8 - test1 configuration

```python

```

### Results

<h6 align="center">
  Visualization of bird species
</h6>

<p align="center">
  <img src="input/viz/image_visualization.jpg" alt="" width="690">
</p>

This project involved numerous experiments, but only the most significant results are highlighted here.

<br>

<h6 align="center">
Model performance metrics
</h6>

<p align="left">
MobileNet - test 20
</p>

<p align="center">
    <img src="output/viz/model_performance_plot_mobilenet_finetune_test20.jpg" alt="" width="550">
</p>

<p align="left">
YOLOv8 test0
</p>

<p align="center">
<img src="output/viz/results.png" alt="" width="550">
</p>

-   YOLOv8 accuracy on test data: 98.5%

The results of this project is pretty decent when compared to similar efforts addressing the task of classifying bird species using the <a href="https://www.kaggle.com/datasets/gpiosenka/100-bird-species/data"> birds 525 species- image classification dataset </a>. The highest accuracy achieved so far is approximately 98% on the test dataset (e.g., <a href="https://thesai.org/Downloads/Volume14No7/Paper_102-Bird_Detection_and_Species_Classification.pdf"> Vo, H.-T., et al. 2023</a>)

### How to use the source code

##### Using the source code for development

-   Fork this repository (https://github.com/LeoUtas/bird_classification_research.git).
-   First thing first, before proceeding, ensure that you are in the root directory of the project.
-   Get the docker container ready:

    -   Run docker build (it might take a while for installing all the required dependencies to your local docker image).

    ```cmd
    docker build -t <name of the docker image> .
    ```

    -   Run a docker container in an interactive mode (once the docker image is built, you can run a docker container).

    ```cmd
    docker run -it -v "$(PWD):/app" <name of the docker image> /bin/bash
    ```

    -   Now, it should be running inside the interactive mode of the docker container to explore the code functionalities.
    -   When you're done, you can simply type "exit" to escape the development environment

    ```
    exit
    ```

-   Also, stop running the container when you're done:

    ```cmd
    docker stop <name of the container>
    ```

### The bottom line

I'm excited to share this repository! Please feel free to explore its functionalities. Thank you for this far. Have a wonderful day ahead!

Best,
Hoang Ng

### Reference

Wu, Y., Kirillov, A., Massa, F., Lo, W.-Y., & Girshick, R. (2019). Detectron2. Retrieved from https://github.com/facebookresearch/detectron2
