Real-Time Road Anomaly Detection

## Project Overview

This project implements a real-time road anomaly detection system designed for dashcam applications on Raspberry Pi hardware. The application utilizes a lightweight object detection model to identify and log potholes in real-time, providing both visual feedback and timestamped documentation of road hazards.

## Features

* **Real-Time Inference:** Processes video streams to detect anomalies with a live FPS counter displayed on the output window.
* **Automated Logging:** Records every detection event in a persistent log file (`log.txt`) with precise timestamps.
* **Visual Evidence Capture:** Automatically saves high-resolution JPEG images of the detected anomaly, including bounding boxes and confidence scores, for further review.
* **Hardware Optimized:** Configured for the Raspberry Pi architecture using ONNX Runtime for efficient model execution.

## Hardware Requirements

* Raspberry Pi 4 or 5
* Raspberry Pi Camera Module v2 (configured at 640x480 resolution)
* High-speed microSD card for logging and image storage

## Installation

### 1. Repository Setup

Clone this repository to your local Raspberry Pi directory:

```bash
git clone <repository-url>
cd <repository-directory>

```

### 2. Dependency Installation

Install the necessary Python libraries using the provided requirements file:

```bash
pip install -r requirements.txt

```

### 3. Model Deployment

Ensure the optimized model file is placed in the `model/` directory as `best_int8.onnx`.

## Usage

Run the detection script from the terminal:

```bash
python detect_pothole.py

```

### Controls

* **Q Key:** Safely terminate the application, close the camera stream, and release all resources.
