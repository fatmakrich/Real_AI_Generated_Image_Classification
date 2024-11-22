import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=False)  # We load the architecture without pre-trained weights

# Modify the final layer to have 2 output units (binary classification)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 output classes: "REAL" vs. "AI Generated"

# Load the model weights (assuming they are compatible with this architecture)
model.load_state_dict(torch.load("model/trained_resnet50.pth", map_location=device))  # Load custom-trained model weights
model.to(device)
model.eval()  # Set model to evaluation mode

# Transformation pipeline for preprocessing the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to fit ResNet50 input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image based on ImageNet stats
])

# Streamlit app title
st.title("AI Image Classifier")

# Image uploader
img = st.file_uploader("Upload your Image")

if img and st.button("Check"):
    # Open and display the image
    image = Image.open(img)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Apply transformations to the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Move the image to the appropriate device (GPU/CPU)
    image = image.to(device)
    
    # Perform prediction
    with torch.no_grad():  # No gradients are needed for inference
        outputs = model(image)  # Get model predictions
        _, predicted = torch.max(outputs, 1)  # Get the index of the class with the highest score

    # Display the result
    if predicted.item() == 1:
        st.write("The given image is Real.")
    else:
        st.write("The given image is AI Generated.")
