# DFME2024
For CCAC MER Competition
## Project Description
This project focuses on enhancing videos using motion magnification and extracting optical flow features for micro-expression recognition. The project involves multiple steps including video magnification, optical flow extraction, dataset integration, and training/testing using MobileViT network.

## Table of Contents
- [Installation](#installation)
- [Running Instructions](#running-instructions)
  - [Step One: Motion Magnification](#step-one-motion-magnification)
  - [Step Two: Optical Flow Extraction](#step-two-optical-flow-extraction)
  - [Step Three: Dataset Integration](#step-three-dataset-integration)
  - [Step Four: Training and Testing](#step-four-training-and-testing)
- [Usage Examples](#usage-examples)

## Installation
1. Clone the repository to your local machine:
    ```sh
    git clone https://github.com/Zhang-ren/DFME2024.git
    cd DFME2024
    ```

2. Install dependencies:
    ```sh
    # If using Python
    pip install -r requirements.txt
    ```

## Running Instructions

### Step One: Motion Magnification
Magnify the motion in videos using the deep_motion_mag library.

- **Input Files**: Path to the video files.
- **Output Files**: Magnified video files.
- **Command**:
    ```sh
    cd deep_motion_mag
    python run_temporal_on_videos.py
    ```
    Refer to the deep_motion_mag repository [here](https://github.com/12dmodel/deep_motion_mag) for more details.

### Step Two: Optical Flow Extraction
Extract optical flow and aligned in optical flow field from the magnified videos. Follow the methodologies described in the papers "Beyond pixels: exploring new representations and applications for motion analysis" and "A main directional mean optical flow feature for spontaneous microexpression recognition".

- **Input Files**: Magnified video files.
- **Output Files**: Optical flow data.
- **Command**:
    ```sh
    cd Opticalflow
    python Prepare.py
    python Prepare_DFME.py
    # Repeat for other datasets: MMEW, SAMM, CAS(ME)^3, CAS(ME)^2, CK+
    ```
    Ensure you have the necessary datasets and follow the specific scripts for each dataset in the Opticalflow folder.

### Step Three: Dataset Integration
Integrate the extracted optical flow data from different datasets for training.

- **Input Files**: Optical flow data from various datasets.
- **Output Files**: Combined dataset ready for training.
- **Command**:
    ```sh
    cd mix_dataset
    python combine_txt.py
    ```

### Step Four: Training and Testing
Train and test the MobileViT network using the integrated dataset. Download the pre-trained weights for MobileViT from [here](https://github.com/wilile26811249/MobileViT).

- **Training Command**:
    ```sh
    cd Train
    python mvit_main.py
    ```
- **Testing Command**:
    ```sh
    python test_main.py
    ```

## Usage Examples
```sh
# Step One: Motion Magnification
cd deep_motion_mag
python run_temporal_on_videos.py

# Step Two: Optical Flow Extraction
cd Opticalflow
python Prepare.py
python Prepare_DFME.py
# Repeat for other datasets

# Step Three: Dataset Integration
cd mix_dataset
python combine_txt.py

# Step Four: Training and Testing
cd Train
python mvit_main.py
python test_main.py
