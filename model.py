from flask import Flask, request, jsonify, render_template, render_template_string
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import pickle
import io
import base64

app = Flask(__name__)

# Model definition (same as training)
class BrainTumorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)
    
    def forward(self, x):
        return self.model(x)

# Load the saved model
def load_model():
    with open('brain_tumor_model.pkl', 'rb') as f:
        model_dict = pickle.load(f)
    
    model = BrainTumorModel()
    model.load_state_dict(model_dict['state_dict'])
    model.eval()
    return model

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Load model at startup
model = load_model()

# Create HTML template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Brain Tumor Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            text-align: center;
        }
        .result {
            margin-top: 20px;
            padding: 10px;
            border-radius: 5px;
        }
        .positive {
            background-color: #ffebee;
            color: #c62828;
        }
        .negative {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        #preview {
            max-width: 300px;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Brain Tumor Detection</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" onchange="previewImage(this)">
            <br><br>
            <img id="preview" style="display: none;">
            <br>
            <input type="submit" value="Predict">
        </form>
        {% if prediction is not none %}
        <div class="result {% if prediction == 1 %}positive{% else %}negative{% endif %}">
            <h2>Prediction Result:</h2>
            <p>{% if prediction == 1 %}
                Tumor Detected (Probability: {{ probability }}%)
               {% else %}
                No Tumor Detected (Probability: {{ probability }}%)
               {% endif %}
            </p>
        </div>
        {% endif %}
    </div>

    <script>
        function previewImage(input) {
            var preview = document.getElementById('preview');
            if (input.files && input.files[0]) {
                var reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                }
                reader.readAsDataURL(input.files[0]);
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the POST request
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        
        # Preprocess the image
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability > 0.5 else 0
            probability_percent = round(probability * 100, 2)
        
        return render_template_string(
            html_template, 
            prediction=prediction,
            probability=probability_percent
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint for programmatic access
@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        # Get the image from the POST request
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        
        # Preprocess the image
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability > 0.5 else 0
        
        return jsonify({
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'tumor_detected': bool(prediction)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)