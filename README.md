<div align="center">

# 🚦 Meta-YOLOv8: Meta-Learning-Enhanced YOLOv8 for Precise Traffic Light Color Detection in ADAS

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?style=for-the-badge&logo=ros&logoColor=white)](https://docs.ros.org/)
[![Docker](https://img.shields.io/badge/Docker-Containers-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/u/tammisetti)
[![ADAS](https://img.shields.io/badge/ADAS-Automotive-FF6F00?style=for-the-badge&logo=tesla&logoColor=white)](https://github.com/)
[![Meta-Learning](https://img.shields.io/badge/Meta--Learning-MAML-8B5CF6?style=for-the-badge)](https://github.com/)
[![F1 Score](https://img.shields.io/badge/F1_Score-93%25-brightgreen?style=for-the-badge)](https://github.com/)
[![Precision](https://img.shields.io/badge/Precision-97%25-brightgreen?style=for-the-badge)](https://github.com/)

**A meta-learning-enhanced YOLOv8 model that targets illuminated sections of traffic signals for precise color detection — outperforming SSD, Faster R-CNN, and DETR under challenging lighting and weather conditions.**

---

</div>

## 🐳 Quick Start — Docker & ROS2

This project is fully containerized using **Docker** and integrated with **ROS2** for real-time robotic/automotive middleware communication. Pre-built Docker images are available on [Docker Hub](https://hub.docker.com/u/tammisetti) for instant reproducibility.

### Available Docker Images

| Image | Description |
|-------|-------------|
| [`tammisetti/ros2-object-det_traff`](https://hub.docker.com/r/tammisetti/ros2-object-det_traff) | **⭐ This project — ROS2 traffic signal detection (Meta-YOLOv8)** |
| [`tammisetti/aithena`](https://hub.docker.com/r/tammisetti/aithena) | AI inference engine |
| [`tammisetti/ros2-object-tracker`](https://hub.docker.com/r/tammisetti/ros2-object-tracker) | ROS2 object tracking node |
| [`tammisetti/ros2-object-det-iris`](https://hub.docker.com/r/tammisetti/ros2-object-det-iris) | ROS2 object detection (IRIS) |
| [`tammisetti/ros2-object-detection1`](https://hub.docker.com/r/tammisetti/ros2-object-detection1) | ROS2 object detection node |
| [`tammisetti/car_sign_v4`](https://hub.docker.com/r/tammisetti/car_sign_v4) | Car signal detection model v4 |

### Pull & Run

```bash
# Pull the traffic signal detection container
docker pull tammisetti/ros2-object-det_traff

# Run with GPU support
docker run --gpus all -it tammisetti/ros2-object-det_traff
```

> **Reproducibility:** All dependencies, model weights, and ROS2 configurations are baked into the containers. Simply pull and run to replicate our results.

---

## 📋 Abstract

The accurate detection of traffic lights is crucial for the effectiveness and safety of **Advanced Driver Assistance Systems (ADAS)**. Meta-YOLOv8 is an enhancement of YOLOv8 using **meta-learning**, specifically designed to improve traffic light detection with a focus on **color recognition**.

Unlike conventional models, Meta-YOLOv8 targets the **illuminated sections** of traffic signals, improving accuracy and detection range even under challenging conditions. The model also reduces computational load by filtering out irrelevant data and employs an innovative labeling technique to handle weather-related detection issues.

> Leveraging meta-learning principles, Meta-YOLOv8 enhances detection reliability across varying lighting and weather conditions without requiring extensive datasets. Comparative assessments show that Meta-YOLOv8 **outperforms traditional models** like SSD, Faster R-CNN, and detection transformers, achieving an **F1 score of 93%** and **precision of 97%**.

---

## 🏁 Performance Comparison

| Model | F1 Score | Precision | Real-Time | Meta-Learning |
|-------|:--------:|:---------:|:---------:|:-------------:|
| Model | Meta-Learning | F1 Score | Precision | Real-Time Speed | Selected |
|-------|:---:|:---:|:---:|:---:|:---:|
| **Meta-YOLOv8 (Ours)** | ✅ | **93%** | **97%** | ✅ Very Fast | ✅ **Final** |
| Meta-SSD | ✅ | Lower | Lower | ✅ Fast | |
| Meta-Faster R-CNN | ✅ | Lower | Lower | ❌ Slow | |
| Meta-DETR | ✅ | Lower | Lower | ❌ Slow (transformer overhead) | |

> **All models were trained with our meta-learning (MAML) pipeline.** Meta-YOLOv8 was selected as the final model due to superior F1/precision scores combined with real-time inference speed — critical for resource-constrained ADAS deployment.

---

## ⚙️ Experimentation Procedure

### Installation

```bash
pip install ultralytics
```

> See the `model/` folder for detailed experimentation steps and code.

### Meta-Learning Pipeline

Initially, we trained a pre-trained YOLOv8 model on a relevant dataset, utilizing it as the **outer loop**. The trained weights were then transferred to a simpler YOLOv8 model (the **inner loop**), which was subsequently fine-tuned on task-specific data. This approach, grounded in meta-learning principles that optimize the learning process itself, resulted in superior performance in extended-range detection compared to SSD, DETR, and Faster R-CNN.

```
┌──────────────────────────────────────────────────────────┐
│                     OUTER LOOP                           │
│  Pre-trained YOLOv8 → Learn task similarities            │
│               ↓ (weight transfer)                        │
│                     INNER LOOP                           │
│  Simplified YOLOv8 → Task-specific fine-tuning           │
│               ↓ (second-order optimization)              │
│                   FINAL MODEL                            │
│  Tailored weights (Θ'i) optimized per class detection    │
└──────────────────────────────────────────────────────────┘
```

The system employs a **two-stage optimization strategy**:

1. **Stage 1 (Outer Loop)** — Focuses on learning task similarities. The base model is pre-trained using images of car turn signals and brake lights, which share common color features with traffic lights and are more readily available.
2. **Stage 2 (Inner Loop)** — The meta-learner refines previously generalized weights to align them with specific task requirements, involving second-order computations to effectively learn across tasks from the same distribution.

---

## 🧠 Meta Learner

<div align="center">

![Metalearner](https://github.com/user-attachments/assets/e7a31999-e7b8-4e08-bdd0-c78f5287269a)

*Fig 1: The base model is initialized with random weights θ and trained on similar tasks. A meta-learner refines these weights to Θ', aligning them with task-specific requirements until the model is fine-tuned with task-specific data, resulting in tailored weights (Θ'i) optimized for each class.*

</div>

---

## 🏷️ Data Preparation & Labeling

### Fusion Dataset

In the absence of specific public datasets tailored to our advanced requirements (high-quality labeled images covering various lighting, angles, and weather), we constructed a **bespoke fusion traffic dataset** from diverse public repositories:

| Dataset | Source | Access |
|---------|--------|--------|
| **KITTI** | [Link](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) | Accessed 10-07-2023 |
| **Kaggle** | [Link](https://www.kaggle.com/datasets/wjybuqi/traffic-light-detection-dataset?resource=download) | Accessed 10-07-2023 |
| **CARLA** | [Link](https://www.kaggle.com/datasets/sachsene/carla-traffic-lights-images) | Accessed 10-07-2023 |
| **LISA** | [Link](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset/code) | Accessed 10-07-2023 |
| **CityScapes** | [Link](https://www.cityscapes-dataset.com/login/) | Accessed 01-12-2023 |
| **EuroCity** | [Link](https://eurocity-dataset.tudelft.nl/eval/user/login?_next=/eval/downloads/detection) | Accessed 15-01-2024 |

> 📦 **Dataset available on Zenodo:** [Meta-YOLOv8 Dataset (v1, December 31, 2024)](https://zenodo.org/records/13969232)

### Novel Targeted Labeling

Traditional labeling methods label the entire traffic light housing, but **2/3 of the bounding box area does not impact the learning process**. Our novel approach focuses specifically on the **illuminated color regions**, improving model robustness and computational efficiency.

<div align="center">

| Conventional Labeling | Targeted Labeling (Ours) |
|:---:|:---:|
| ![Conventional](https://github.com/user-attachments/assets/9a70e057-29f6-467d-8a29-1a7ec6e12172) | ![Targeted](https://github.com/user-attachments/assets/bab97a36-ec8d-4c71-81ea-d90fac2315e7) |
| Bounding box covers entire housing — 2/3 of area is irrelevant | Focuses on illuminating regions — highest impact on learning |

</div>

Manual labeling of **315 images** ensured focus on salient features, crucial for traffic management and autonomous vehicle navigation.

---

## 📊 Results

<div align="center">

![Results 1](https://github.com/user-attachments/assets/90f14931-baa1-4a67-8e47-471c24c9feec)

![Results 2](https://github.com/user-attachments/assets/39207c75-dd46-4d7f-8886-04841d89c5e0)

![Results 3](https://github.com/user-attachments/assets/5a10338b-fa65-41d1-8b0e-8ee421475850)

</div>

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **97% Precision** | Outperforms SSD, Faster R-CNN, and DETR in traffic light detection |
| 🔬 **Targeted Labeling** | Novel approach focusing on illuminated regions, not full housing |
| ⚡ **Low Compute** | Filters irrelevant data, reducing computational load for edge devices |
| 🌧️ **Weather Robust** | Innovative technique handles adverse weather detection challenges |
| 🔄 **Few-Shot Ready** | Meta-learning enables rapid adaptation without extensive datasets |
| 🐳 **Containerized** | Docker + ROS2 for instant reproducibility |

---

## 📚 Citation

If you use this work, please cite the following:

```bibtex
@article{meta_yolov8_traffic,
  title={Meta-YOLOv8: Meta-Learning-Enhanced YOLOv8 for Precise Traffic Light Color Detection in ADAS},
  journal={Electronics},
  year={2024},
  url={https://www.mdpi.com/2079-9292/13/14/2771}
}
```

### Related Publications

- **IEEE Publication:** [https://ieeexplore.ieee.org/document/10533619](https://ieeexplore.ieee.org/document/10533619)
- **MAML (Finn et al., 2017):** [https://proceedings.mlr.press/v70/finn17a.html](https://proceedings.mlr.press/v70/finn17a.html)
- **MDPI Electronics:** [https://www.mdpi.com/2079-9292/13/14/2771](https://www.mdpi.com/2079-9292/13/14/2771)

---

<div align="center">

**Meta-YOLOv8 advances traffic light detection for ADAS with optimized feature weighting, reduced computational demands, and meta-learning adaptability — potentially reducing the risk of accidents in resource-constrained autonomous systems.**

---

Made with 🔬 for safer autonomous driving

</div>
