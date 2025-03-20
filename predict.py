import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import streamlit as st
import os

# ✅ Load the trained model
def load_model():
    model = models.densenet121()
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(class_names))  # Modify classifier for crew members
    model.load_state_dict(torch.load("models/densenet121.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()  # Set model to evaluation mode
    return model.to(device)

# ✅ Define the same image transformations used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Load class names (crew member names)
data_dir = "augmented_data"
class_names = sorted(os.listdir(data_dir))  # Gets folder names as class labels

# ✅ Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load the trained model
model = load_model()

# ✅ Function to predict image
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)  # Convert image to tensor and add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
    
    predicted_class = class_names[predicted.item()]
    
    # If confidence is low, assume it's not a crew member
    if confidence < 0.7:  
        return "The person in the pic uploaded is not a member of the Straw Hat crew."
    
    return f"This is {predicted_class.capitalize()}! (Confidence: {confidence*100:.2f}%)"

# ✅ Streamlit UI
st.title("Straw Hat Crew Image Classifier 🏴‍☠️")
st.write("Upload an image, and I'll tell you which crew member it is! If it's not a member, I'll let you know.")

# ✅ File upload in Streamlit
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ✅ Predict the character
    prediction = predict_image(image)
    st.write("## 🔍 Prediction:")
    st.write(prediction)
