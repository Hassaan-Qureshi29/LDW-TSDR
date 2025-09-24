# LDW-TSDR
Lane Departure Warning with Traffic Sign Detection &amp; Recognition

This repository contains the trained models and resources developed for a real-time edge-based lane departure warning and traffic sign recognition system. The work combines three core components: lane detection for monitoring vehicle position, traffic sign detection using YOLOv8, and traffic sign classification using MobileNetV3 variants. The approach is designed to address the challenges of local driving environments, focusing on Pakistani road conditions while also testing cross-dataset generalization.

The weights folder includes four key model files:

yolov8s.pth is used for detecting traffic signs in real-time scenes and provides the bounding boxes for the classification stage.

mobilenetv3_base.pth is the initial classification model trained on an imbalanced Pakistani traffic sign dataset and serves as the baseline for further improvements.

mobilenetv3_v2.pth is trained on a balanced version of the Pakistani dataset and achieves 93.1% accuracy on the self-collected local dataset, demonstrating improved performance over the base model.

mobilenetv3_bangladesh.pth is obtained by fine-tuning the base model on the Bangladesh traffic sign dataset to evaluate cross-dataset generalization and reaches 95.7% accuracy on the Bangladesh test set.

These models can be used to reproduce the experiments and results described in the associated research paper, showcasing both regional adaptability and robustness of the proposed lane departure warning and traffic sign recognition framework.
