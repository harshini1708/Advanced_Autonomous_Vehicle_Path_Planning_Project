
# Advanced Autonomous Vehicle Path Planning Using Deep Neural Networks (DNNs)

## Project Overview:
This project uses a deep learning model (DNN) for autonomous vehicle path planning. 
The system leverages NVIDIA GPUs with CUDA for real-time performance improvements. 
The model is trained on simulated vehicle sensor data.

### Features:
- Advanced DNN architecture using convolutional and recurrent layers for decision making.
- NVIDIA CUDA acceleration for faster training and real-time inference.
- Real-time path planning using simulated sensor data.

### Technologies:
- **Python**
- **TensorFlow**
- **Keras**
- **CUDA for NVIDIA GPUs**

### Steps:
1. **Data Preparation**: The simulated dataset includes LIDAR, Camera, and vehicle state information.
2. **Model Training**: The model is trained using the simulated data to predict the optimal path.
3. **Real-Time Inference**: The trained model performs real-time path planning for an autonomous vehicle.

## How to Run:
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python train_advanced_model.py`
3. Perform inference: `python serve_advanced_model.py`
