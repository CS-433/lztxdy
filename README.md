# CS-433 Machine Learning Project 2

## Road Segmentation

Road Segmentation is a deep learning project designed to detect and mask roads in aerial satellite images. Using the UNet architecture, the model accurately identifies roads from the background. This finds applications in urban planning, navigation, geographic analysis or military purposes.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Architecture](#architecture)
- [Usage](#usage)
  - [Model Initialization](#model-initialization)
  - [Training and Validation](#training-and-validation)
  - [Performance Testing](#performance-testing)
- [Results](#results)
- [Samples Predicitions](#samples-predicitons)

---

## Overview

This project leverages the **UNet** architecture to perform semantic segmentation of roads in high-resolution aerial satellite images. The objective is to predict binary masks where road pixels are labeled as 1 and the background as 0. This technology has versatile use cases, from urban planning to autonomous navigation systems.

---

## Dataset

The dataset used for this project includes:
1. **Satellite Images**: High-resolution RGB satellite images.
2. **Ground Truth Masks**: Binary masks where road pixels are labeled as white (1) and background pixels as black (0).

Each training sample consists of an image and its corresponding ground truth mask. The dataset can be downloaded from [AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files).

---

## Installation

1. **Dataset Preparation**:
   - Download and extract the dataset.
   - You have to put all extracted folder into a new folder epfl_road-segmentation and the files has to be organized into the following structure:
     ```
     epfl_road-segmentation/
     |-- training/
     |-- test_set_images/
     ```
   - Please keep this folder in the same directory as your project.
   - Update data_root to the path to epfl_road-segmentation folder.

2. **Dependencies**:
   We trained our model using PyTorch. To display the images and plots we used matplotlib. We also used tqdm to have a little progress bar during the training phase.
   To install the required Python libraries please use the following commands:
   ```bash
   !pip install torch torchvision matplotlib tqdm
   ```

4. **Google Drive Access (Optional)**:
   We used the GPU from Google Colab, which allows faster computation speed than CPU.
   To use Google Colab, please upload the dataset to your Google Drive and mount it, and set the data_root as the path to the folder in your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   data_root = '/content/drive/MyDrive/.../epfl_road-segmentation'  # Update with your dataset path
   ```

---

## Architecture

The project utilizes the **UNet** architecture, a popular model for semantic segmentation tasks. Key features include:
- **Encoder-Decoder Structure**: Captures high-level features and reconstructs them into pixel-level predictions.
- **Skip Connections**: Preserve spatial information from the encoder for better localization.
- **Regularization**: Batch normalization and dropout layers to reduce overfitting.

---

## Usage

### Model Initialization
Run the cells in the **Data Preprocessing** section to prepare the dataset:
- The dataset will be split into training, validation, and test sets.
- Data augmentation (e.g., rotation, flipping, Gaussian blur) will be applied to the training set. Augmented images will be saved with the naming convention `satImage_*_augmented.png`.
- Please do not modify files name and ensure the images and their corresponding ground truth masks are correctly aligned.

### Training and Validation
Train the model by running the cells in the **Model Training** section:
- Monitor the training and validation loss, overall accuracy (OA), and F1-score after each epoch.
- The F1-score is calculated based on the Intersection over Union (IoU).

### Performance Testing
Run the cells in the **Model Testing** section to predict road masks for new images:
- The model will use unseen test images to generate their segmentation masks.
- It will also compare the predictions with the groundtruth and compute the F1-score.

---

## Results

After training, the model achieved the following metrics on the validation set:
- **IoU (over the whole validation set)**: 80%
- **Pixel Accuracy**: 95.5%
- **Validation Loss**: 0.12

### Samples Predictions
Below are some predicitons from the test set images.

![image1](https://github.com/user-attachments/assets/401553d5-0c4b-439a-87a7-676277841530)
![image2](https://github.com/user-attachments/assets/bb8baab3-4dd4-491e-87c3-72ebd3e58d6c)
![image3](https://github.com/user-attachments/assets/eb3cfb5a-4cad-4fad-b90b-d92e78aeef5b)
![image4](https://github.com/user-attachments/assets/ab35ebcd-66ec-4dd2-870d-607a79892598)
![image5](https://github.com/user-attachments/assets/66682518-72d1-4bdc-a90a-15634c85369f)


