# Facial Features Recognition

Face Mesh is a face geometry solution that estimates 468 3D face landmarks in real-time even on mobile devices. It employs machine learning (ML) to infer the 3D surface geometry, requiring only a single camera input without the need for a dedicated depth sensor. The solution is bundled with the Face Geometry module that bridges the gap between the face landmark estimation and useful real-time augmented reality (AR) applications.

Based on this technology, I have created a pipeline for detection, recognition and features understanding (age, race and gender, for now) on any input video or cam capture with few lines of code.

This approach has very positive points, such as the improved quality in face detection and context recognition, when compared to the traditional OpenCV models and the machine learning approaches.


## Used dataset

FairFace:
- https://github.com/joojs/fairface
- https://github.com/dchen236/FairFace

## Usage

>> python cam_detector.py

## Requirements

Developed using Python 3.7 in the Anaconda env and the followed libs:

- tensorflow
- mediapipe
- numpy
- opencv
