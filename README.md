# LDW-TSDR
Lane Departure Warning with Traffic Sign Detection &amp; Recognition

This repository contains the trained models and resources developed for a real-time edge-based lane departure warning and traffic sign recognition system. The work integrates three core components: lane detection to monitor vehicle position, traffic sign detection using YOLOv8, and traffic sign classification using MobileNetV3. The approach is tailored to Pakistani road conditions and further tested for cross-dataset generalization on a Bangladesh dataset.

The weights folder includes four key model files:

yolov8s.pth is used for detecting traffic signs in real-time scenes and provides the bounding boxes for the classification stage.

mobilenetv3_base.pth is the initial classification model trained on an imbalanced Pakistani traffic sign dataset and serves as the baseline for further improvements.

mobilenetv3_v2.pth is trained on a balanced version of the Pakistani dataset and achieves 93.1% accuracy on the self-collected local dataset, demonstrating improved performance over the base model.

mobilenetv3_bangladesh.pth is obtained by fine-tuning the base model on the Bangladesh traffic sign dataset to evaluate cross-dataset generalization and reaches 95.7% accuracy on the Bangladesh test set.

Datasets

The traffic sign classification models were trained and validated using the following publicly available datasets:

Pakistani Traffic Sign Recognition Dataset: https://www.kaggle.com/datasets/mexwell/pakistani-traffic-sign-recognition-dataset/data

Bangladesh Traffic Sign Dataset: https://www.kaggle.com/datasets/tusher7575/traffic-sign-in-bangladesh/data?select=bd_traffic_signs

Data Augmentation Profile

To handle class imbalance and improve model generalization, a custom augmentation and dataset balancing script was applied. Each class was balanced to 250 images through a combination of undersampling, oversampling, and targeted augmentation. The key augmentation steps included:

Random rotation between -15° and +15° to simulate different camera angles and minor sign orientation changes.

Random saturation adjustment by ±40% to make the model robust against varying color intensities.

Random brightness adjustment by ±30% to account for changes in ambient lighting conditions.

Random contrast (exposure) adjustment by ±13% to handle different levels of image contrast.

Addition of 20 Gaussian-blurred images per class (blur radius = 3) to enhance robustness against motion blur or defocused captures.

The augmentation pipeline ensured that each class contained a balanced mix of original and synthetically generated images while maintaining natural visual characteristics. This strategy significantly improved the classification model’s ability to generalize across varying lighting, color, and environmental conditions.

These datasets, trained weights, and augmentation strategy can be used to reproduce the experiments and results described in the associated research paper, demonstrating the adaptability and robustness of the proposed lane departure warning and traffic sign recognition framework.
