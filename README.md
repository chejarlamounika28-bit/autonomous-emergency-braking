# Autonomous Emergency Braking for Rear-End Collision Avoidance

## Project Overview

This project presents an autonomous emergency braking prototype designed to reduce the risk of rear-end collisions using computer vision and real-time object detection.

The system uses a camera feed to detect objects, estimate their distance, monitor lanes, and determine braking or lane-change actions based on the detected environment.

## Project Output

![AEB Object Detection](aeb_object_detection.png)

## Technologies Used

- Python
- OpenCV
- YOLOv8
- NumPy
- Computer Vision

## Key Features

- Real-time object detection using YOLOv8
- Approximate distance estimation using detected object size
- Lane monitoring and lane detection
- Emergency braking based on obstacle distance
- Speed and braking-status calculation
- Traffic-light detection
- Basic lane-change decision logic
- Real-time webcam processing

## Project Files

- `aeb_system.py` – Main AEB system with object detection, distance estimation, braking and lane-monitoring logic.
- `lane_detection.py` – Lane detection using Canny edge detection, Region of Interest (ROI), and Hough Line Transform.

## How It Works

1. Capture live video using a webcam.
2. Detect objects using YOLOv8.
3. Estimate the approximate distance of detected objects.
4. Monitor the current lane for obstacles.
5. Determine the appropriate braking response based on distance.
6. Detect traffic-light conditions.
7. Display lane, object, speed, and braking information in real time.

## Project Objective

To explore the practical application of computer vision and AI-based object detection in autonomous vehicle safety systems.
