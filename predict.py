import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import argparse


DEFAULT_MODEL_PATH = os.path.join("quantum", "best_quantum_lung_cancer_model.pth")
# Default to None; we'll auto-discover a test image under Data/test if not provided
DEFAULT_IMAGE_PATH = None
DEFAULT_BATCH_FOLDER = os.path.join("Data", "test")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} | Capability: {torch.cuda.get_device_capability(0)}")
    torch.backends.cudnn.benchmark = True

num_qubits = 6
num_layers = 6

dev = qml.device("default.qubit", wires=num_qubits)

def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(num_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

weight_shapes = {"weights": (num_layers, num_qubits, 3)}
qnode = qml.QNode(quantum_circuit, dev, interface="torch")

class EnhancedQuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        scaled_input = x * self.scale
        return self.q_layer(scaled_input)

class QuantumHybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            self.backbone = models.resnet50(weights=None)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_qubits)
        )
        
        self.quantum_layer = EnhancedQuantumLayer()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_qubits, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        quantum_input = self.dim_reduction(features)
        quantum_output = self.quantum_layer(quantum_input)
        logits = self.classifier(quantum_output)
        return logits

def load_model(model_path=DEFAULT_MODEL_PATH):
    """Load the saved model from checkpoint"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        class_names = checkpoint['class_names']
        num_classes = len(class_names)
        
        model = QuantumHybridModel(num_classes).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Classes: {class_names}")
        
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def preprocess_image(image_path=DEFAULT_IMAGE_PATH):
    """Preprocess an image for prediction"""
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  
        return image_tensor, image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None

def predict(model, image_tensor, class_names):
    """Make a prediction on a preprocessed image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class_idx = torch.argmax(probabilities).item()
        
        predicted_class = class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx].item()
        
        top_probs, top_indices = torch.topk(probabilities, min(3, len(class_names)))
        top_predictions = [
            (class_names[idx], prob.item()) 
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        return predicted_class, confidence, top_predictions

def visualize_prediction(image, predicted_class, confidence, top_predictions):
    """Visualize the prediction results"""
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    classes = [p[0] for p in top_predictions]
    confidences = [p[1] for p in top_predictions]
    
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, confidences)
    plt.yticks(y_pos, classes)
    plt.xlabel('Confidence')
    plt.title('Top Predictions')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()

def _find_first_test_image(root=os.path.join("Data", "test")):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    if not os.path.isdir(root):
        return None
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                return os.path.join(dirpath, f)
    return None


def predict_single_image(model_path=DEFAULT_MODEL_PATH, image_path=DEFAULT_IMAGE_PATH, visualize=True):
    """Function to predict a single image with hard-coded paths"""
    if image_path is None:
        image_path = _find_first_test_image()
        if image_path is None:
            print("Error: --image not provided and no image found under 'Data/test'.")
            return
    print(f"Predicting image: {image_path}")
    print(f"Using model: {model_path}")
    
    model, class_names = load_model(model_path)
    if model is None:
        return
    
    image_tensor, original_image = preprocess_image(image_path)
    if image_tensor is None:
        return
    
    predicted_class, confidence, top_predictions = predict(model, image_tensor, class_names)
    
    print(f"\nPrediction Results for {image_path}:")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print("\nTop Predictions:")
    for i, (class_name, prob) in enumerate(top_predictions, 1):
        print(f"{i}. {class_name}: {prob:.2%}")
    
    if visualize and original_image is not None:
        visualize_prediction(original_image, predicted_class, confidence, top_predictions)
        print("\nVisualization saved as 'prediction_result.png'")
        
    return predicted_class, confidence, top_predictions

def predict_batch(model_path=DEFAULT_MODEL_PATH, image_folder=DEFAULT_BATCH_FOLDER, output_file='batch_predictions.txt'):
    """Make predictions on all images in a folder using hard-coded paths"""
    print(f"Batch processing images from: {image_folder}")
    print(f"Using model: {model_path}")
    
    if not os.path.exists(image_folder):
        print(f"Error: Folder {image_folder} does not exist")
        return
    
    model, class_names = load_model(model_path)
    if model is None:
        return
    
    results = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    with open(output_file, 'w') as f:
        f.write(f"Image,Predicted Class,Confidence\n")
        
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                    predicted_idx = torch.argmax(probs).item()
                    predicted_class = class_names[predicted_idx]
                    confidence = probs[predicted_idx].item()
                
                results.append((img_file, predicted_class, confidence))
                f.write(f"{img_file},{predicted_class},{confidence:.4f}\n")
                print(f"Image: {img_file} - Prediction: {predicted_class} ({confidence:.2%})")
            
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                f.write(f"{img_file},ERROR,0.0\n")
    
    print(f"Batch prediction completed. Results saved to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Predict lung cancer type from CT scan')
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE_PATH, 
                        help='Path to a CT image. If omitted, the first image under Data/test is used if available.')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, 
                        help=f'Path to the trained model checkpoint (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--batch', action='store_true', help='Process a batch of images')
    parser.add_argument('--batch_folder', type=str, default=DEFAULT_BATCH_FOLDER,
                        help=f'Folder containing images to process in batch mode (default: {DEFAULT_BATCH_FOLDER})')
    parser.add_argument('--no_vis', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    if args.batch:
        predict_batch(args.model, args.batch_folder)
    else:
        predict_single_image(args.model, args.image, not args.no_vis)


if __name__ == "__main__":
    main()
    