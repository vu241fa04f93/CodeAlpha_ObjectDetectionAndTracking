# OBJECT VISION
### Intelligent Object Detection & Multi-Object Tracking

🚀 **Live Demo:** https://object-vision-tracking.streamlit.app/

💻 **Source Code:** https://github.com/vu241fa04f93/CodeAlpha_ObjectDetectionAndTracking

# 🚀 OBJECT VISION
### Intelligent Object Detection & Multi-Object Tracking System

<p align="center">
  <b>Real-Time Computer Vision | YOLOv8 | OpenCV | SORT Tracking | Streamlit</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit" />
  <img src="https://img.shields.io/badge/Tracking-SORT-orange" />
  <img src="https://img.shields.io/badge/Status-Active-success" />
</p>

---

## 📌 Overview

**OBJECT VISION** is an intelligent real-time object detection and multi-object tracking application built using Python, OpenCV, Ultralytics YOLOv8, and the SORT (Simple Online and Realtime Tracking) algorithm.

The system detects multiple objects in video streams, identifies their classes, assigns unique tracking IDs, and maintains their identities across consecutive frames.

It integrates two pretrained YOLOv8 models:

- **YOLOv8n (Nano):** Lightweight model optimized for faster CPU inference.
- **YOLOv8s (Small):** Larger model offering an alternative detection configuration.

The application features a professional Streamlit dashboard with video processing, detection visualization, tracking statistics, model selection, and performance monitoring.

The primary objective is to provide an efficient and interactive computer vision system that can run on standard laptops without requiring dedicated GPU hardware.

---

## ✨ Key Features

### 1. Real-Time Object Detection

- Integration with Ultralytics YOLOv8.
- Support for YOLOv8n and YOLOv8s.
- Detection of multiple objects in a single frame.
- Class labels and confidence scores.
- Bounding box visualization.
- Full-frame detection without intentional center cropping.

### 2. Multi-Object Tracking

- Custom Class-Aware SORT tracking implementation.
- Unique tracking IDs for detected objects.
- Kalman Filter-based motion prediction.
- Hungarian Algorithm for data association.
- Intersection over Union (IoU)-based matching.
- Track management across consecutive frames.
- Duplicate track suppression.

### 3. Professional AI Dashboard

- Interactive Streamlit interface.
- Model selection through the sidebar.
- Video processing interface.
- Real-time object detection visualization.
- Object class and tracking ID information.
- Detection statistics.
- FPS and inference latency monitoring.
- Processing resolution and compute device information.

### 4. Performance Optimization

- Model caching to avoid unnecessary model loading.
- Frame-skipping optimization.
- Track propagation during skipped detection frames.
- Lightweight YOLOv8n option for CPU systems.
- Performance benchmarking between model variants.

### 5. Testing and Validation

- Automated system testing.
- Real-video tracking tests.
- Video pipeline testing.
- Model benchmarking.
- Memory and pipeline stability checks.

---

## 🧠 System Architecture

The project follows a modular architecture in which detection, tracking, visualization, and application management are handled separately.

```text
                 INPUT VIDEO
                      |
                      |
               VIDEO FRAME
                      |
                      |
              OPENCV PROCESSING
                      |
                      |
              YOLOv8 DETECTION
                /          \
               /            \
          YOLOv8n          YOLOv8s
               \            /
                \          /
                      |
                      |
               DETECTION BOXES
                      |
                      |
              CLASS-AWARE SORT
                      |
             -------------------
             |        |        |
          Kalman   Hungarian  IoU
          Filter   Matching   Matching
             |        |        |
             -------------------
                      |
                      |
               TRACKING IDS
                      |
                      |
             VISUALIZATION
                      |
                      |
              STREAMLIT UI
                      |
                      |
              FINAL OUTPUT
```

---

## 🔬 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Ultralytics YOLOv8 | Object detection |
| OpenCV | Video and image processing |
| PyTorch | Deep learning framework |
| NumPy | Numerical operations |
| SciPy | Hungarian assignment algorithm |
| FilterPy | Kalman Filter implementation |
| Streamlit | Interactive dashboard |
| Pandas | Data and statistics handling |

---

## ⚙️ Detection Models

The application supports two pretrained YOLOv8 model variants.

### YOLOv8n – Nano

- Lightweight architecture.
- Lower computational requirements.
- Suitable for CPU-based systems.
- Recommended for faster processing.

### YOLOv8s – Small

- Larger model architecture.
- Alternative detection configuration.
- Higher computational requirements.
- Suitable when additional processing resources are available.

### Model Comparison

The following are results from the project's reported CPU benchmarking experiment.

| Metric | YOLOv8n | YOLOv8s |
|---|---:|---:|
| Model Size | 6.25 MB | 22.5 MB |
| Average CPU Latency | 57.8 ms | 135.6 ms |
| Pure Inference FPS | 17.3 | 7.4 |
| Live Tracking FPS (N=3) | 54.2 | 21.8 |
| Average Detections/Frame | 3.16 | 3.48 |

**Benchmark configuration:**
- Video: `pedestrian_bus_demo.mp4`
- Resolution: 640 × 640
- Backend: CPU

*Note: These are machine-specific benchmark observations, not guaranteed performance on every device. Live FPS and inference FPS measure different processing stages.*

**Recommended Model:** YOLOv8n for CPU-based environments.

---

## 🎯 Object Tracking Methodology

OBJECT VISION uses a Class-Aware SORT implementation to maintain object identities across video frames.

### 1. Kalman Filter

The Kalman Filter predicts the future position of an object using its previous motion information.

The tracking state is represented as:

```text
x = [u, v, s, r, du, dv, ds]ᵀ
```

Where:

- `u`: Horizontal center position.
- `v`: Vertical center position.
- `s`: Bounding box scale.
- `r`: Bounding box aspect ratio.
- `du`: Horizontal velocity.
- `dv`: Vertical velocity.
- `ds`: Scale velocity.

### 2. Hungarian Algorithm

The Hungarian Algorithm performs optimal assignment between predicted tracks and newly detected objects.

It helps associate detections with existing object trajectories.

### 3. IoU Matching

Intersection over Union measures the overlap between two bounding boxes.

```text
IoU = Intersection Area / Union Area
```

It helps determine whether a detection corresponds to an existing track.

### 4. Class-Aware Matching

The custom tracking implementation restricts matching between objects of different classes.

This reduces incorrect associations between objects belonging to different categories.

### 5. Frame Skipping

To reduce computational overhead, the system can skip selected detection frames while propagating existing tracks.

This allows tracking visualization to continue between detection updates.

---

## 📁 Project Structure

```text
CodeAlpha_ObjectDetectionAndTracking/
│
├── models/
│   ├── yolov8n.pt
│   └── yolov8s.pt
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py
│   └── performance.py
│
├── sample_videos/
│   ├── pedestrian_bus_demo.mp4
│   ├── sample_traffic_test.mp4
│   └── test_bus.jpg
│
├── output/
│
├── app.py
├── detector.py
├── tracker.py
├── sort.py
├── config.py
│
├── benchmark_models.py
├── generate_sample_video.py
│
├── test_real_tracking.py
├── test_system.py
├── test_video_pipeline.py
│
├── requirements.txt
├── howToRun.txt
├── .gitignore
└── README.md
```

### Main Components

| File | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `detector.py` | YOLOv8 detection and model management |
| `tracker.py` | Tracking management and statistics |
| `sort.py` | Class-Aware SORT implementation |
| `config.py` | Centralized configuration |
| `visualization.py` | Bounding boxes and visual elements |
| `performance.py` | FPS and latency monitoring |
| `benchmark_models.py` | YOLOv8n and YOLOv8s benchmarking |
| `generate_sample_video.py` | Synthetic sample video generation |
| `test_system.py` | Automated system tests |
| `test_real_tracking.py` | Real-video tracking validation |
| `test_video_pipeline.py` | Video pipeline stability testing |

---

## 💻 Installation and Setup

### Prerequisites

Make sure you have installed:

- Python 3.10 or later.
- Git.
- VS Code (recommended).
- Basic knowledge of Python.

### Step 1: Clone the Repository

```bash
git clone https://github.com/vu241fa04f93/CodeAlpha_ObjectDetectionAndTracking.git
```

### Step 2: Navigate to the Project

```bash
cd CodeAlpha_ObjectDetectionAndTracking
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

**Windows:**

```powershell
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Launch the Application

```bash
streamlit run app.py
```

Open your browser and visit:

```text
http://localhost:8501
```

---

## 🧪 Testing and Benchmarking

The project includes multiple testing scripts to evaluate system functionality.

### Run System Tests

```bash
python test_system.py
```

### Run Real-Video Tracking Tests

```bash
python test_real_tracking.py
```

### Run Video Pipeline Tests

```bash
python test_video_pipeline.py
```

### Run Model Benchmarking

```bash
python benchmark_models.py
```

These scripts help evaluate detection, tracking, processing performance, and pipeline stability.

---

## 📊 Applications

OBJECT VISION demonstrates computer vision techniques that can be applied to:

- Smart traffic monitoring.
- Pedestrian detection.
- Vehicle tracking.
- Surveillance systems.
- Crowd monitoring.
- Intelligent video analytics.
- Real-time object monitoring.

---

## ⚠️ Current Limitations

- Detection accuracy depends on the pretrained model and input video quality.
- CPU-based processing can limit inference speed.
- Tracking performance depends on detection quality and object movement.
- Occlusion and rapid movement can affect tracking consistency.
- Remote hosting environments may have limited CPU, memory, and processing resources.
- Local webcam access does not automatically transfer to a remotely hosted application.

---

## 🔮 Future Improvements

- Browser-based webcam integration.
- GPU acceleration.
- Advanced tracking with Deep SORT or ByteTrack.
- Improved object re-identification.
- Custom dataset training and fine-tuning.
- Detection analytics and downloadable reports.
- Improved multi-object tracking under occlusion.
- Cloud-based video processing.

---

## 👨‍💻 Developer

**Abhishek Kumar**

B.Tech – Computer Science and Engineering

Vignan's Foundation for Science, Technology and Research (VFSTR)

### Connect With Me

- GitHub: [@vu241fa04f93](https://github.com/vu241fa04f93)

---

## 🏆 Project Information

**Project:** OBJECT VISION – Intelligent Object Detection & Multi-Object Tracking

**Organization:** CodeAlpha

**Domain:** Artificial Intelligence and Computer Vision

**Task:** Object Detection and Tracking

**Repository:** [CodeAlpha_ObjectDetectionAndTracking](https://github.com/vu241fa04f93/CodeAlpha_ObjectDetectionAndTracking)

---

<p align="center">
  <b>Built with Python, Computer Vision and Artificial Intelligence.</b>
  <br>
  ⭐ If you find this project interesting, consider giving it a star!
</p>
