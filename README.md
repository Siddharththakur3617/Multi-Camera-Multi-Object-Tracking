# Multi-Camera Multi-Object Tracking

## Overview

This repository hosts a robust Multi-Camera Multi-Object Tracking (MCMOT) system designed to track multiple objects across various camera views in real-time or offline settings. By integrating advanced computer vision and deep learning techniques, the system ensures accurate object detection, tracking, and re-identification across multiple camera perspectives. It is ideal for applications such as surveillance, autonomous driving, and crowd analysis.

## Features

- **Multi-Camera Integration**: Tracks objects across multiple cameras with overlapping or non-overlapping fields of view.
- **Object Detection**: Employs state-of-the-art models (YOLO) for precise object detection.
- **Tracking Algorithms**: Uses robust tracking methods (BYTETrack) for consistent single-camera tracking.
- **Re-Identification (ReID)**: Maintains object identities across cameras using advanced re-identification techniques.
- **Real-Time Performance**: Optimized for real-time processing with GPU acceleration.
- **Modular Architecture**: Easily adaptable for various use cases and camera configurations.
- **Visualization Tools**: Includes utilities for visualizing tracking results and camera feeds.

## Installation

### Prerequisites

- Python 3.8 or higher  
- PyTorch 1.9 or higher (with CUDA support for GPU acceleration)  
- OpenCV 4.5 or higher  
- Standard Python libraries (NumPy, SciPy, etc.)  
- Pre-trained models for detection and re-identification  

### Setup Instructions

#### 1. Clone the Repository:

```bash
git clone https://github.com/your-username/Multi-Camera-Multi-Object-Tracking.git
cd Multi-Camera-Multi-Object-Tracking
```
#### 2. Set Up a Virtual Environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

#### 3. Install Dependencies:
```bash
pip install -r requirements.txt
```
#### 4. Download Pre-trained Models:
- Obtain pre-trained models for detection and re-identification (e.g., YOLO weights, ReID models).

- Place them in the assets/ directory.

#### 5. Congigure Tracking
Set paths for your camera videos and weights:

```bash
{
  "model_weights": "assets/custom_yolov11.pt",
  "video_paths": ["assets/static/broadcast.mp4", "assets/static/tacticam.mp4"],
  "output_dir": "ouput/",
  "save_log": true
}
```

### Command-Line Arguments
--config: Path to the configuration file
Default: config/config.json

--weights: Path to the YOLOv11 model weights
Example: assets/custom_yolov11.pt

--video0: Path to the first camera stream
Example: assets/static/broadcast.mp4

--video1: Path to the second camera stream
Example: assets/static/tacticam.mp4

--output: Directory to save tracking results (videos and logs)
Example: output/

#### Example Command
```bash
python src/main.py \
  --config config/config.json \
  --weights assets/custom_yolov11.pt \
  --video0 assets/static/broadcast.mp4 \
  --video1 assets/static/tacticam.mp4 \
  --output output/
```
#### Directory Structure
```bash
Multi-Camera-Multi-Object-Tracking:.
│   .gitattributes
│   DOCKERFILE
│   README.md
│   requirements.txt
│
├───assets
│   │   custom_yolov11.pt
│   │
│   └───static
│           broadcast.mp4
│           tacticam.mp4
│
├───config
│       config.json
│
├───ouput
│       output_camera_0.mp4
│       output_camera_1.mp4
│       tracker.log
│
└───src
        bytetracker.py
        config.py
        kalman_filter.py
        logger.py
        main.py
        multi_camera_tracker.py
        track.py
        __init__.py
```

### Configuration
The system is highly configurable through YAML files located in the config/ directory. Key parameters include:

Camera Settings: Define intrinsic and extrinsic parameters for each camera.

Detection Model: Specify model type (e.g., YOLOv5, Faster R-CNN) and confidence threshold.

Tracking Parameters: Set IoU threshold, maximum track age, etc.

ReID Settings: Configure re-identification model and distance metric.

Example Configuration (config/default.yaml)
yaml
Copy
Edit
cameras:
  - id: 0
    source: "rtsp://example.com/stream1"
    calibration: "config/calibration/camera0.yaml"
detector:
  model: "yolov5s"
  conf_threshold: 0.5
tracker:
  type: "deepsort"
  max_age: 30
reid:
  model: "osnet_x1_0"
  distance_metric: "cosine"

### Datasets
Sample datasets are provided in the data/ directory for testing purposes.

The system supports standard multi-camera datasets like DukeMTMC and MOTChallenge, as well as custom datasets.

To use custom datasets, place video files in the data/ directory and update the configuration file accordingly.

### Performance Optimization
GPU Acceleration: Install PyTorch with CUDA support for enhanced processing speed.

Batch Processing: Adjust batch size in the configuration file to optimize speed and memory usage.

Multi-Threading: Enable multi-threaded data loading for real-time applications.

### Contributing
We welcome contributions to enhance the project! To contribute:

Fork the repository.

Create a new branch (git checkout -b feature-branch)

Implement your changes and commit (git commit -m "Add feature")

Push to the branch (git push origin feature-branch)

Submit a Pull Request.

Please adhere to the coding style and include tests for new features.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgments
Built with open-source libraries including PyTorch, OpenCV, and YOLO.

Gratitude to the computer vision community for providing datasets and pre-trained models.

### Contact
For questions or issues, please open an issue on GitHub or contact the repository owner at:

📧 siddharththakur3617@example.com
