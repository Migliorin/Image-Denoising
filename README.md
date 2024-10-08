<!-- vim-markdown-toc GitLab -->

* [1. Image-Denoising](#1-image-denoising)
* [2. Installation](#2-installation)
    * [2.1. Prerequisites](#21-prerequisites)
    * [2.2. Steps](#22-steps)
* [3. About Datasets](#3-about-datasets)
* [4. Annotations](#4-annotations)
* [5. Train results](#5-train-results)

<!-- vim-markdown-toc -->

---
# 1. Image-Denoising

Image denoising project is a research to explore new methods to remove the noise present in faces images.

# 2. Installation

To get started with the project, you'll need to set up a Conda environment using the provided `environment.yml` file.

## 2.1. Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/index.html)

## 2.2. Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/migliorin/Image-Denoising.git

# 3. About Datasets

The experiments were done using the following datasets:
 - All datasets follow the annotation
        ![alt text](imgs/annotation_afw.png)
 - [AFW](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
 - [HELEN](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
 - [300W](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
 - [LFPW](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)

# 4. Annotations

1. dataframe_v1.csv - Only salt and pepper noise with 8604 images. Train split is 70/15/15

# 5. Train results

The params and results are save at [results/README.md](results/README.md)
