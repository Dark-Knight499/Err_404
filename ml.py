import skimage.io
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image
import io
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
import pickle

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition
class BrainTumorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)
    
    def forward(self, x):
        return self.model(x)

# Load model
def load_model():
    with open('brain_tumor_model.pkl', 'rb') as f:
        model_dict = pickle.load(f)
    
    model = BrainTumorModel()
    model.load_state_dict(model_dict['state_dict'])
    model.to(device)
    model.eval()
    return model

# Load model at startup
model = load_model()

def bytes_to_skimage(file: UploadFile):
    """
    Converts an uploaded image file to a skimage-compatible NumPy array.
    """
    image_bytes = file.file.read()
    image_array = skimage.io.imread(io.BytesIO(image_bytes))
    return image_array

def preprocess_image(img):
    """
    Preprocesses the input image for the model.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert numpy array to PIL Image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    img_tensor = transform(img)
    img_tensor = img_tensor.to(device)
    return img_tensor

def predict_tumor(model, img):
    """
    Predicts tumor presence using the given model.
    """
    img = preprocess_image(img)
    
    with torch.no_grad():
        output = model(img.unsqueeze(0))
        probability = torch.sigmoid(output).item()
        prediction = 1 if probability > 0.5 else 0
        
    return {
        'prediction': prediction,
        'probability': round(probability * 100, 2),
        'tumor_detected': bool(prediction)
    }

async def predict_brain_tumor(file: UploadFile):
    """
    FastAPI endpoint to predict brain tumor from uploaded image.
    """
    try:
        if not file.filename.lower().endswith(('.jpeg', '.jpg', '.png')):
            raise HTTPException(status_code=400, detail="Unsupported image format")
        
        img = bytes_to_skimage(file)
        result = predict_tumor(model, img)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

app = FastAPI()
