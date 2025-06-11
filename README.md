# MNIST Digit Recognizer 🖌️🔢

## Overview 🌟
This project is a PyTorch-based digit recognition system that uses the MNIST dataset. It enables users to train a neural network to recognize handwritten digits, draw digits on a canvas in Google Colab, and upload MNIST images for prediction. Developed entirely in Google Colab, it includes a trained model, training loss visualization, and sample predictions. 🚀

## Features ✨
- Trains a neural network on the MNIST dataset. 🧠
- Provides a canvas interface to draw digits for real-time prediction. 🎨
- Supports uploading MNIST images for comparison. 📤
- Saves the raw canvas image for debugging. 💾
- Visualizes training loss and test set predictions. 📊

## Requirements 🛠️
- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- NumPy
- PIL (Python Imaging Library)
- ipycanvas
- OpenCV (cv2)
- Google Colab environment ☁️

## Installation ⚙️
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MNIST-Digit-Recognizer.git
   cd MNIST-Digit-Recognizer
   ```

## Usage 📋
1. Open the project in Google Colab. 📓
2. Run all cells to train the model, evaluate it, and load the interface. ▶️
3. Draw a digit on the canvas and click "Predict from Canvas" to see the prediction. ✍️
4. Upload an MNIST image and click "Upload and Predict Image" to test with known data. 🖼️
5. Click "Save Canvas Image" to download the raw drawing for inspection. 💻

## Note 📝
To avoid retraining the model, upload the `mnist_model.pth` file in Google Colab or place it in the same directory if running locally. ⏩