# Enable custom widget manager for ipycanvas in Colab
from google.colab import output
output.enable_custom_widget_manager()

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import io
from google.colab import files

# Install ipycanvas if not already installed
!pip install ipycanvas -q

from ipycanvas import Canvas
import ipywidgets as widgets
import cv2

# Set random seed for reproducibility
torch.manual_seed(42)

# Step 1: Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Step 2: Define the neural network
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# Step 3: Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DigitClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load model parameters if they exist, otherwise train
model_path = 'mnist_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Loaded pre-trained model parameters.")
else:
    # Step 4: Train the model with loss tracking
    losses = []
    def train_model(num_epochs=50):
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(trainloader)
            losses.append(epoch_loss)
            print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
    
    print("Training the model...")
    train_model()
    torch.save(model.state_dict(), model_path)
    print("Model parameters saved to 'mnist_model.pth'")

    # Visualize training progress
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    plt.close()

# Step 5: Evaluate the model
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# Step 6: Real-time digit recognition
def preprocess_canvas_image(canvas):
    # Get the canvas data
    image_data = canvas.get_image_data(x=0, y=0, width=canvas.width, height=canvas.height)
    print(f"Raw image data shape: {image_data.shape}, size: {image_data.size}")

    # Validate and reshape if needed
    height, width = canvas.height, canvas.width
    expected_shape = (height, width, 4)
    if image_data.shape != expected_shape:
        expected_size = height * width * 4
        if image_data.size == expected_size and image_data.ndim == 1:
            image_data = image_data.reshape(expected_shape)
        else:
            raise ValueError(f"Unexpected image data shape {image_data.shape}, expected {expected_shape}")

    # Convert to grayscale (using RGB only)
    image = np.mean(image_data[:, :, :3], axis=2).astype(np.uint8)
    print(f"Grayscale image shape: {image.shape}, min: {image.min()}, max: {image.max()}")
    plt.imsave('raw_canvas_image.png', image, cmap='gray')
    print("Saved raw grayscale image as 'raw_canvas_image.png'")

    # Resize to 28x28
    resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA).astype(np.float32)
    print(f"Resized image shape: {resized_image.shape}, min: {resized_image.min()}, max: {resized_image.max()}")

    # Apply MNIST normalization
    image = (resized_image - 0.1307) / 0.3081
    print(f"Preprocessed image min: {image.min()}, max: {image.max()}, mean: {image.mean()}, std: {image.std()}")

    # Convert to tensor
    image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
    return image

def preprocess_uploaded_image(file_content, filename):
    try:
        print(f"Processing uploaded file: {filename}")
        image = Image.open(io.BytesIO(file_content))
        print(f"Image mode: {image.mode}, size: {image.size}")
        image = image.convert('L')
        image = image.resize((28, 28), Image.LANCZOS)
        image = np.array(image).astype(np.float32)
        print(f"Image array shape: {image.shape}, min: {image.min()}, max: {image.max()}")
        image = (image - 0.1307) / 0.3081
        print(f"Normalized image min: {image.min()}, max: {image.max()}, mean: {image.mean()}, std: {image.std()}")
        image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        return image
    except Exception as e:
        print(f"Error processing uploaded image: {str(e)}")
        return None

def predict_digit(canvas_or_image):
    model.eval()
    with torch.no_grad():
        if isinstance(canvas_or_image, Canvas):
            image = preprocess_canvas_image(canvas_or_image)
        else:
            image = canvas_or_image
            if image is None:
                output_label.value = "Error: Failed to process uploaded image."
                return None
        image = image.to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Create canvas for drawing
canvas = Canvas(width=150, height=150)  # Reduced from 200x200 to 150x150
canvas.sync_image_data = True
canvas.fill_style = 'black'
canvas.fill_rect(0, 0, 150, 150)

# Global flag to track mouse state
mouse_down = False

# Drawing logic
def handle_mouse_down(x, y):
    global mouse_down
    mouse_down = True
    canvas.begin_path()
    canvas.move_to(x, y)

def handle_mouse_up(x, y):
    global mouse_down
    mouse_down = False

def handle_mouse_move(x, y):
    global mouse_down
    if mouse_down:
        canvas.line_to(x, y)
        canvas.stroke_style = 'white'
        canvas.line_width = 5
        canvas.stroke()
        canvas.move_to(x, y)

def handle_clear_button(b):
    global mouse_down
    mouse_down = False
    canvas.fill_style = 'black'
    canvas.fill_rect(0, 0, 150, 150)
    output_label.value = ''

# Connect events
canvas.on_mouse_down(handle_mouse_down)
canvas.on_mouse_up(handle_mouse_up)
canvas.on_mouse_move(handle_mouse_move)

# Clear button
clear_button = widgets.Button(description="Clear")
clear_button.on_click(handle_clear_button)

# Output label for prediction
output_label = widgets.Label(value='')

# Predict button for canvas
predict_button = widgets.Button(description="Predict from Canvas")
def on_predict_button_clicked(b):
    prediction = predict_digit(canvas)
    if prediction is not None:
        output_label.value = f'Predicted Digit: {prediction}'

predict_button.on_click(on_predict_button_clicked)

# Upload and predict button
upload_predict_button = widgets.Button(description="Upload and Predict Image")
def on_upload_predict_button_clicked(b):
    try:
        print("Initiating file upload...")
        uploaded = files.upload()
        if uploaded:
            filename = list(uploaded.keys())[0]
            file_content = uploaded[filename]
            print(f"Uploaded file: {filename}, size: {len(file_content)} bytes")
            image = preprocess_uploaded_image(file_content, filename)
            prediction = predict_digit(image)
            if prediction is not None:
                output_label.value = f'Predicted Digit: {prediction}'
        else:
            output_label.value = "No file uploaded. Please select a file."
    except Exception as e:
        print(f"Error during upload or processing: {str(e)}")
        output_label.value = "Error: Failed to upload or process image."

upload_predict_button.on_click(on_upload_predict_button_clicked)

# Save canvas image button
save_canvas_image_button = widgets.Button(description="Save Canvas Image")
def on_save_canvas_image_button_clicked(b):
    image_data = canvas.get_image_data(x=0, y=0, width=canvas.width, height=canvas.height)
    height, width = canvas.height, canvas.width
    if image_data.shape != (height, width, 4):
        expected_size = height * width * 4
        if image_data.size == expected_size and image_data.ndim == 1:
            image_data = image_data.reshape(height, width, 4)
    image = np.mean(image_data[:, :, :3], axis=2).astype(np.uint8)
    plt.imsave('raw_canvas_image.png', image, cmap='gray')
    files.download('raw_canvas_image.png')

save_canvas_image_button.on_click(on_save_canvas_image_button_clicked)

# Display the interface with centered layout
display(widgets.VBox([
    widgets.HBox([canvas], layout=widgets.Layout(justify_content='center')),
    widgets.HBox([clear_button, predict_button], layout=widgets.Layout(justify_content='center')),
    widgets.HBox([upload_predict_button], layout=widgets.Layout(justify_content='center')),
    widgets.HBox([save_canvas_image_button], layout=widgets.Layout(justify_content='center')),
    output_label
]))

# Step 8: Visualize a few predictions from test set
def visualize_predictions():
    model.eval()
    images, labels = next(iter(testloader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    plt.figure(figsize=(12, 2))
    for i in range(6):
        plt.subplot(1, 6, i+1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f'Pred: {predicted[i].item()}\nTrue: {labels[i].item()}')
        plt.axis('off')
    plt.savefig('predictions.png')
    plt.show()
    plt.close()

# Run the project
print("\nEvaluating the model...")
evaluate_model()
print("\nGenerating sample predictions...")
visualize_predictions()
print("Predictions saved as 'predictions.png'")