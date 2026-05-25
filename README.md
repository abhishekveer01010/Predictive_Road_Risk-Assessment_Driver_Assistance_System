# Predictive_Road_Risk-Assessment_Driver_Assistance_System
A two-module ML pipeline integrating geospatial road risk prediction with real-time traffic sign recognition, delivered through embedded hardware for active driver alerts.


SmartDriver: Predictive Road Risk Assessment & Driver Assistance System
Seeing a stop sign is useful, but knowing that a specific intersection has a history of accidents is what actually saves lives. Existing Advanced Driver Assistance Systems (ADAS) often ignore environmental context—a sharp curve on a rainy night is far more dangerous than the same curve on a sunny afternoon, yet standard systems treat them identically.
SmartDriver is a context-aware driver assistance system that combines a real-time traffic sign classifier with a geospatial machine learning risk map. The system knows which roads are historically dangerous, detects traffic signs the driver needs to respond to, and escalates physical alerts based on the fusion of both data points.  

🚀 Key Features
Context-Aware Alerts: The system produces contextually different outputs for the same sign depending on the road's historical danger level.
Geospatial Risk Mapping: Evaluates real road segments using OpenStreetMap and Mapillary imagery to score danger before the car even arrives.  
Real-Time Sign Recognition: Uses a CNN + Spatial Transformer Network (STN) trained on the GTSRB dataset (43 classes) to identify signs in poor lighting and bad angles.
Hardware Integration: Utilizes an ESP32-CAM for wireless video streaming and a secondary ESP32 controller to drive a color-coded LED and buzzer dashboard.

🧠 System Architecture
The project is divided into an offline data pipeline and a real-time inference loop, which merge at the integration layer.

Module 1: Offline Road Risk Assessment
This module generates a persistent digital risk map (Predictions.geojson).
Data Collection: Pulls street-level images from Mapillary within a bounding box (Magarpatta, Pune) and maps them to OpenStreetMap road segments.
Feature Extraction: Passes images through a fine-tuned Mask R-CNN (sv1.pth) to detect 124 semantic scene categories (e.g., pedestrians, barriers, road markings).
Risk Prediction: Converts object counts into a 124-dimensional feature vector, which is fed into a trained Support Vector Machine (RRE.sav) to output a binary High/Low risk level and a continuous risk score (0.02–7.97).

Module 2: Real-Time Traffic Sign Recognition
This module processes live camera feeds to classify traffic signs.
Architecture: A Convolutional Neural Network (CNN) augmented with a Spatial Transformer Network (STN) to geometrically correct image alignment before classification.
Training: Trained on the German Traffic Sign Recognition Benchmark (GTSRB) using CLAHE grayscale preprocessing to handle varied lighting.
Smoothing: Uses a 12-frame temporal window requiring a 7/12 majority vote to prevent UI flickering from single-frame misclassifications.

Module 3: Driver Assistance Integration (driver_assistance.py)
The fusion layer that drives the entire system.
Simulates GPS movement across the mapped Magarpatta road segments.
Cross-references the live sign detection with the active road's risk score.
Sends serial commands to the ESP32 hardware to trigger specific alerts.
